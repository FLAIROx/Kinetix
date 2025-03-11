import jax
import jax.numpy as jnp
import numpy as np

from kinetix.environment.env_state import StaticEnvParams
from kinetix.render.renderer_symbolic_common import (
    make_circle_features,
    make_joint_features,
    make_polygon_features,
    make_thruster_features,
)


def make_render_symbolic(env_params, static_params: StaticEnvParams, padded: bool = False):
    def render_symbolic(state):
        nshapes = static_params.num_polygons + static_params.num_circles

        polygon_features, polygon_mask = make_polygon_features(state, env_params, static_params)
        mask_to_ignore_walls_ceiling = np.ones(static_params.num_polygons, dtype=bool)
        mask_to_ignore_walls_ceiling[np.array([i for i in range(1, static_params.num_static_fixated_polys)])] = False

        polygon_features = polygon_features[mask_to_ignore_walls_ceiling]
        polygon_mask = polygon_mask[mask_to_ignore_walls_ceiling]

        circle_features, circle_mask = make_circle_features(state, env_params, static_params)
        joint_features, joint_idxs, joint_mask = make_joint_features(state, env_params, static_params)
        thruster_features, thruster_idxs, thruster_mask = make_thruster_features(state, env_params, static_params)

        two_J = joint_features.shape[0]
        J = two_J // 2  # for symbolic only have the one
        joint_features = jnp.concatenate(
            [
                joint_features[:J],  # shape (2 * J, K)
                jax.nn.one_hot(joint_idxs[:J, 0], nshapes),  # shape (2 * J, N)
                jax.nn.one_hot(joint_idxs[:J, 1], nshapes),  # shape (2 * J, N)
            ],
            axis=1,
        )
        thruster_features = jnp.concatenate(
            [
                thruster_features,
                jax.nn.one_hot(thruster_idxs, nshapes),
            ],
            axis=1,
        )

        polygon_features = jnp.where(polygon_mask[:, None], polygon_features, 0.0)
        circle_features = jnp.where(circle_mask[:, None], circle_features, 0.0)
        joint_features = jnp.where(joint_mask[:J, None], joint_features, 0.0)
        thruster_features = jnp.where(thruster_mask[:, None], thruster_features, 0.0)

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
            obs = jnp.clip(obs, a_min=-10.0, a_max=10.0)
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
            obs = jnp.clip(obs, a_min=-10.0, a_max=10.0)
            obs = jnp.nan_to_num(obs)
            return obs

    return render_symbolic
