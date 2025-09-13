from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np

from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.render.renderer_symbolic_common import (
    make_circle_features,
    make_joint_features,
    make_polygon_features,
    make_thruster_features,
)
from kinetix.environment.utils import create_empty_env
from kinetix.render.inverse_renderer_common import features_to_joint, features_to_rigidbody, features_to_thruster
from jax2d.engine import calculate_collision_matrix


def _get_symbolic_features(state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams):
    nshapes = static_env_params.num_polygons + static_env_params.num_circles

    polygon_features, polygon_mask = make_polygon_features(state, env_params, static_env_params)
    mask_to_ignore_walls_ceiling = np.ones(static_env_params.num_polygons, dtype=bool)
    mask_to_ignore_walls_ceiling[np.array([i for i in range(1, static_env_params.num_static_fixated_polys)])] = False

    polygon_features = polygon_features[mask_to_ignore_walls_ceiling]
    polygon_mask = polygon_mask[mask_to_ignore_walls_ceiling]

    circle_features, circle_mask = make_circle_features(state, env_params, static_env_params)
    joint_features, joint_idxs, joint_mask = make_joint_features(state, env_params, static_env_params)
    thruster_features, thruster_idxs, thruster_mask = make_thruster_features(state, env_params, static_env_params)

    two_J = joint_features.shape[0]
    J = two_J // 2  # for symbolic only have the one
    joint_features = jnp.concatenate(
        [
            joint_features[:J],  # shape (2 * J, K)
            (jax.nn.one_hot(joint_idxs[:J, 0], nshapes)),  # shape (2 * J, N)
            (jax.nn.one_hot(joint_idxs[:J, 1], nshapes)),  # shape (2 * J, N)
        ],
        axis=1,
    )
    thruster_features = jnp.concatenate(
        [
            thruster_features,
            (jax.nn.one_hot(thruster_idxs, nshapes)),
        ],
        axis=1,
    )

    polygon_features = jnp.where(polygon_mask[:, None], polygon_features, 0.0)
    circle_features = jnp.where(circle_mask[:, None], circle_features, 0.0)
    joint_features = jnp.where(joint_mask[:J, None], joint_features, 0.0)
    thruster_features = jnp.where(thruster_mask[:, None], thruster_features, 0.0)

    return nshapes, polygon_features, circle_features, joint_features, thruster_features, joint_idxs, thruster_idxs


def make_render_symbolic(
    env_params: EnvParams, static_env_params: StaticEnvParams, padded: bool = False, clip: bool = True
) -> Callable[[EnvState], jnp.ndarray]:
    def render_symbolic(state: EnvState) -> jnp.ndarray:
        _, polygon_features, circle_features, joint_features, thruster_features, _, _ = _get_symbolic_features(
            state, env_params, static_env_params
        )
        if padded:
            # pad final dimension with zeros to make all length max_width
            max_width = max(
                polygon_features.shape[-1],
                circle_features.shape[-1],
                joint_features.shape[-1],
                thruster_features.shape[-1],
            )
            polygon_features = jnp.pad(
                polygon_features, ((0, 0), (0, max_width - polygon_features.shape[-1]))
            )  # (2, 36)
            circle_features = jnp.pad(circle_features, ((0, 0), (0, max_width - circle_features.shape[-1])))  # (2, 36)
            joint_features = jnp.pad(joint_features, ((0, 0), (0, max_width - joint_features.shape[-1])))  # (1, 36)
            thruster_features = jnp.pad(
                thruster_features, ((0, 0), (0, max_width - thruster_features.shape[-1]))
            )  # (1, 36)
            # stack
            obs = jnp.concatenate(
                [polygon_features, circle_features, joint_features, thruster_features], axis=0
            )  # (6, 36)
            # add one-hot encoding of the shape type (+4 to feature dim)
            n_polys, n_circles, n_joints, n_thrusters = (
                polygon_features.shape[0],
                circle_features.shape[0],
                joint_features.shape[0],
                thruster_features.shape[0],
            )
            object_types = jnp.array([0] * n_polys + [1] * n_circles + [2] * n_joints + [3] * n_thrusters)  # (6,)
            one_hot_object_types = jax.nn.one_hot(object_types, 4)  # (6, 4)
            obs = jnp.concatenate([obs, one_hot_object_types], axis=-1)  # (6, 40)
            # add gravity (+1 to feature dim)
            gravity = jnp.full((obs.shape[0], 1), state.gravity[1] / 10)  # (6, 1)
            obs = jnp.concatenate([obs, gravity], axis=-1)  # (6, 41)
            # clip
            if clip:
                obs = jnp.clip(obs, min=-10.0, max=10.0)
            obs = jnp.nan_to_num(obs)
            return obs

        else:
            obs = jnp.concatenate(
                [
                    polygon_features.flatten(),
                    circle_features.flatten(),
                    joint_features.flatten(),
                    thruster_features.flatten(),
                    jnp.array([state.gravity[1]]) / 10,
                ],
                axis=0,
            )
            if clip:
                obs = jnp.clip(obs, min=-10.0, max=10.0)
            obs = jnp.nan_to_num(obs)
            return obs

    return render_symbolic


def make_inverse_render_symbolic(
    env_state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams
) -> Callable[[jnp.ndarray], EnvState]:
    """This creates an inverse renderer, which takes in an observation and returns an EnvState.

        Note: Since the symbolic observation clips values to be between -10 and 10, the inverse renderer will not be able to
        perfectly reconstruct the original EnvState.
    Args:
        env_state (EnvState): This is a dummy env state, and is used to get the shapes of the various arrays.

    Returns:
        Callable[[jnp.ndarray], EnvState]: Maps symbolic observation to EnvState.
    """
    (
        nshapes,
        polygon_features_dummy,
        circle_features_dummy,
        joint_features_dummy,
        thruster_features_dummy,
        _,
        _,
    ) = _get_symbolic_features(env_state, env_params, static_env_params)

    def symbolic_flat_obs_to_env_state(obs: jnp.ndarray) -> EnvState:
        # polygons
        start_index = 0
        end_index = polygon_features_dummy.shape[0] * polygon_features_dummy.shape[1]
        polygon_features = jnp.reshape(obs[start_index:end_index], polygon_features_dummy.shape)

        # circles
        start_index = end_index
        end_index += circle_features_dummy.shape[0] * circle_features_dummy.shape[1]
        circle_features = jnp.reshape(obs[start_index:end_index], circle_features_dummy.shape)

        # joints
        start_index = end_index
        end_index += joint_features_dummy.shape[0] * joint_features_dummy.shape[1]
        joint_features = jnp.reshape(obs[start_index:end_index], joint_features_dummy.shape)

        # thrusters
        start_index = end_index
        end_index += thruster_features_dummy.shape[0] * thruster_features_dummy.shape[1]
        thruster_features = jnp.reshape(obs[start_index:end_index], thruster_features_dummy.shape)

        env_state = create_empty_env(static_env_params)

        # Polygon & Circle
        circle, circle_shape_roles, circle_densities = features_to_rigidbody(
            env_params, static_env_params, circle_features, static_env_params.num_circles, is_circle=True
        )
        # polygon

        polygon, polygon_shape_roles, polygon_densities = features_to_rigidbody(
            env_params, static_env_params, polygon_features, static_env_params.num_polygons - 3, is_circle=False
        )

        def _add_walls_ceiling(small_array, dummy_large_array):
            return jnp.concatenate(
                [small_array[:1], dummy_large_array[1 : static_env_params.num_static_fixated_polys], small_array[1:]],
                axis=0,
            )

        (polygon, polygon_shape_roles, polygon_densities) = jax.tree.map(
            _add_walls_ceiling,
            (polygon, polygon_shape_roles, polygon_densities),
            (env_state.polygon, env_state.polygon_shape_roles, env_state.polygon_densities),
        )
        env_state = env_state.replace(
            circle=circle,
            polygon=polygon,
            circle_shape_roles=circle_shape_roles,
            polygon_shape_roles=polygon_shape_roles,
            circle_densities=circle_densities,
            polygon_densities=polygon_densities,
        )

        # Thruster

        thrusters, thruster_bindings = features_to_thruster(static_env_params, env_state, thruster_features, nshapes)

        env_state = env_state.replace(
            thruster=thrusters,
            thruster_bindings=thruster_bindings,
        )

        # Joint
        joint, motor_bindings = features_to_joint(static_env_params, env_state, joint_features, nshapes)
        env_state = env_state.replace(
            joint=joint,
            motor_bindings=motor_bindings,
        )

        return env_state.replace(
            collision_matrix=calculate_collision_matrix(static_env_params, env_state.joint),
        )

    return symbolic_flat_obs_to_env_state
