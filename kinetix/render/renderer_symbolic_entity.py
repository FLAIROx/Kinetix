from typing import Callable
import jax
import jax.numpy as jnp
from flax import struct
from jax2d.engine import get_pairwise_interaction_indices

from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.environment.utils import create_empty_env
from kinetix.render import inverse_renderer_common
from kinetix.render.renderer_symbolic_common import (
    make_circle_features,
    make_joint_features,
    make_polygon_features,
    make_thruster_features,
)
from kinetix.render.renderer_symbolic_flat import _get_symbolic_features
from jax2d.engine import calculate_collision_matrix


@struct.dataclass
class EntityObservation:
    circles: jnp.ndarray
    polygons: jnp.ndarray
    joints: jnp.ndarray
    thrusters: jnp.ndarray

    circle_mask: jnp.ndarray
    polygon_mask: jnp.ndarray
    joint_mask: jnp.ndarray
    thruster_mask: jnp.ndarray
    attention_mask: jnp.ndarray
    # collision_mask: jnp.ndarray

    joint_indexes: jnp.ndarray
    thruster_indexes: jnp.ndarray


def make_render_entities(env_params, static_params, ignore_attention_mask=False):
    _, _, _, circle_circle_pairs, circle_rect_pairs, rect_rect_pairs = get_pairwise_interaction_indices(static_params)
    circle_rect_pairs = circle_rect_pairs.at[:, 0].add(static_params.num_polygons)
    circle_circle_pairs = circle_circle_pairs + static_params.num_polygons

    def render_entities(state: EnvState):
        state = jax.tree.map(lambda x: jnp.nan_to_num(x), state)

        joint_features, joint_indexes, joint_mask = make_joint_features(state, env_params, static_params)
        thruster_features, thruster_indexes, thruster_mask = make_thruster_features(state, env_params, static_params)

        poly_nodes, poly_mask = make_polygon_features(state, env_params, static_params)
        circle_nodes, circle_mask = make_circle_features(state, env_params, static_params)

        def _add_grav(nodes):
            return jnp.concatenate(
                [nodes, jnp.zeros((nodes.shape[0], 1)) + state.gravity[1] / 10], axis=-1
            )  # add gravity to each shape's embedding

        poly_nodes = _add_grav(poly_nodes)
        circle_nodes = _add_grav(circle_nodes)

        # Shape of something like (NPoly + NCircle + 2 * NJoint + NThruster )
        mask_flat_shapes = jnp.concatenate([poly_mask, circle_mask], axis=0)
        num_shapes = static_params.num_polygons + static_params.num_circles

        def make_n_squared_mask(val):
            # val has shape N of bools.
            N = val.shape[0]
            A = jnp.eye(N, N, dtype=bool)  # also have things attend to themselves
            # Make the shapes fully connected
            full_mask = A.at[:num_shapes, :num_shapes].set(jnp.ones((num_shapes, num_shapes), dtype=bool))

            one_hop_connected = jnp.zeros((N, N), dtype=bool)
            one_hop_connected = one_hop_connected.at[joint_indexes[:, 0], joint_indexes[:, 1]].set(True)
            one_hop_connected = one_hop_connected.at[0, 0].set(False)  # invalid joints have indices of (0, 0)

            multi_hop_connected = jnp.logical_not(state.collision_matrix)

            collision_mask = state.collision_matrix

            # where val is false, we want to mask out the row and column.
            full_mask = full_mask & (val[:, None]) & (val[None, :])
            collision_mask = collision_mask & (val[:, None]) & (val[None, :])
            multi_hop_connected = multi_hop_connected & (val[:, None]) & (val[None, :])
            one_hop_connected = one_hop_connected & (val[:, None]) & (val[None, :])
            collision_manifold_mask = jnp.zeros_like(collision_mask)

            def _set(collision_manifold_mask, pairs, active):
                return collision_manifold_mask.at[
                    pairs[:, 0],
                    pairs[:, 1],
                ].set(active)

            collision_manifold_mask = _set(
                collision_manifold_mask,
                rect_rect_pairs,
                jnp.logical_or(state.acc_rr_manifolds.active[..., 0], state.acc_rr_manifolds.active[..., 1]),
            )

            collision_manifold_mask = _set(collision_manifold_mask, circle_rect_pairs, state.acc_cr_manifolds.active)
            collision_manifold_mask = _set(collision_manifold_mask, circle_circle_pairs, state.acc_cc_manifolds.active)
            collision_manifold_mask = collision_manifold_mask & (val[:, None]) & (val[None, :])

            return jnp.concatenate(
                [full_mask[None], multi_hop_connected[None], one_hop_connected[None], collision_manifold_mask[None]],
                axis=0,
            )

        if ignore_attention_mask:
            mask_n_squared = None
        else:
            mask_n_squared = make_n_squared_mask(mask_flat_shapes)

        return EntityObservation(
            circles=circle_nodes,
            polygons=poly_nodes,
            joints=joint_features,
            thrusters=thruster_features,
            circle_mask=circle_mask,
            polygon_mask=poly_mask,
            joint_mask=joint_mask,
            thruster_mask=thruster_mask,
            attention_mask=mask_n_squared,
            joint_indexes=joint_indexes,
            thruster_indexes=thruster_indexes,
        )

    return render_entities


def make_inverse_render_entity(
    env_state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams
) -> Callable[[jnp.ndarray], EnvState]:
    """This creates an inverse renderer, which takes in an observation and returns an EnvState.

    Args:
        env_state (EnvState): This is a dummy env state, and is used to get the shapes of the various arrays.

    Returns:
        Callable[[jnp.ndarray], EnvState]: Maps symbolic entity observation to Env
    """

    (
        nshapes,
        *_,
    ) = _get_symbolic_features(env_state, env_params, static_env_params)

    def symbolic_entity_obs_to_env_state(obs: EntityObservation) -> EnvState:

        env_state = create_empty_env(static_env_params)
        polygons, polygon_shape_roles, polygon_densities = inverse_renderer_common.features_to_rigidbody(
            env_params, static_env_params, obs.polygons, static_env_params.num_polygons, is_circle=False
        )

        polygons = polygons.replace(
            active=obs.polygon_mask,
            collision_mode=polygons.collision_mode.at[: static_env_params.num_static_fixated_polys].set(2),
        )

        circles, circle_shape_roles, circle_densities = inverse_renderer_common.features_to_rigidbody(
            env_params, static_env_params, obs.circles, static_env_params.num_circles, is_circle=True
        )
        circles = circles.replace(active=obs.circle_mask)

        env_state = env_state.replace(
            polygon=polygons,
            circle=circles,
            polygon_shape_roles=polygon_shape_roles,
            circle_shape_roles=circle_shape_roles,
            polygon_densities=polygon_densities,
            circle_densities=circle_densities,
        )

        joints, motor_bindings = inverse_renderer_common.features_to_joint(
            static_env_params,
            env_state,
            obs.joints[: static_env_params.num_joints],
            nshapes,
            joint_idxs_override=obs.joint_indexes[: static_env_params.num_joints],
        )

        joints = joints.replace(
            active=obs.joint_mask[: static_env_params.num_joints],
        )

        thrusters, thruster_bindings = inverse_renderer_common.features_to_thruster(
            static_env_params, env_state, obs.thrusters, nshapes, thruster_idxs_override=obs.thruster_indexes
        )
        thrusters = thrusters.replace(active=obs.thruster_mask, object_index=obs.thruster_indexes)

        env_state = env_state.replace(
            joint=joints, thruster=thrusters, motor_bindings=motor_bindings, thruster_bindings=thruster_bindings
        )

        return env_state.replace(
            collision_matrix=calculate_collision_matrix(static_env_params, env_state.joint),
        )

    return symbolic_entity_obs_to_env_state
