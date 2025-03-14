import jax
import jax.numpy as jnp
from flax import struct
from jax2d.engine import get_pairwise_interaction_indices

from kinetix.environment.env_state import EnvState
from kinetix.render.renderer_symbolic_common import (
    make_circle_features,
    make_joint_features,
    make_polygon_features,
    make_thruster_features,
)


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
        state = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), state)

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
