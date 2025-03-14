import jax
import jax.numpy as jnp
from jax2d.sim_state import RigidBody

from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams


def _get_base_shape_features(
    density: jnp.ndarray, roles: jnp.ndarray, shapes: RigidBody, env_params: EnvParams
) -> jnp.ndarray:
    cos = jnp.cos(shapes.rotation)
    sin = jnp.sin(shapes.rotation)
    return jnp.concatenate(
        [
            shapes.position,
            shapes.velocity,
            jnp.expand_dims(shapes.inverse_mass, axis=1),
            jnp.expand_dims(shapes.inverse_inertia, axis=1),
            jnp.expand_dims(density, axis=1),
            jnp.expand_dims(jnp.tanh(shapes.angular_velocity / 10), axis=1),
            jax.nn.one_hot(roles, env_params.num_shape_roles),
            jnp.expand_dims(sin, axis=1),
            jnp.expand_dims(cos, axis=1),
            jnp.expand_dims(shapes.friction, axis=1),
            jnp.expand_dims(shapes.restitution, axis=1),
        ],
        axis=1,
    )


def add_circle_features(
    base_features: jnp.ndarray, shapes: RigidBody, env_params: EnvParams, static_env_params: StaticEnvParams
):
    return jnp.concatenate(
        [
            base_features,
            shapes.radius[:, None],
            jnp.ones_like(base_features[:, :1]),  # one for circle
        ],
        axis=1,
    )


def make_circle_features(
    state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams
) -> tuple[jnp.ndarray, jnp.ndarray]:
    base_features = _get_base_shape_features(state.circle_densities, state.circle_shape_roles, state.circle, env_params)
    node_features = add_circle_features(base_features, state.circle, env_params, static_env_params)
    return node_features, state.circle.active


def add_polygon_features(
    base_features: jnp.ndarray, shapes: RigidBody, env_params: EnvParams, static_env_params: StaticEnvParams
):
    vertices = jnp.where(
        jnp.arange(static_env_params.max_polygon_vertices)[None, :, None] < shapes.n_vertices[:, None, None],
        shapes.vertices,
        jnp.zeros_like(shapes.vertices) - 1,
    )

    return jnp.concatenate(
        [
            base_features,
            jnp.zeros_like(base_features[:, :1]),  # zero for polygon
            vertices.reshape((vertices.shape[0], -1)),
            jnp.expand_dims((shapes.n_vertices <= 3), axis=1),
        ],
        axis=1,
    )


def make_polygon_features(
    state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams
) -> tuple[jnp.ndarray, jnp.ndarray]:
    base_features = _get_base_shape_features(
        state.polygon_densities, state.polygon_shape_roles, state.polygon, env_params
    )
    node_features = add_polygon_features(base_features, state.polygon, env_params, static_env_params)
    return node_features, state.polygon.active


def make_unified_shape_features(
    state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams
) -> tuple[jnp.ndarray, jnp.ndarray]:
    base_p = _get_base_shape_features(state.polygon_densities, state.polygon_shape_roles, state.polygon, env_params)
    base_c = _get_base_shape_features(state.circle_densities, state.circle_shape_roles, state.circle, env_params)
    base_p = add_polygon_features(base_p, state.polygon, env_params, static_env_params)
    base_p = add_circle_features(base_p, state.polygon, env_params, static_env_params)

    base_c = add_polygon_features(base_c, state.circle, env_params, static_env_params)
    base_c = add_circle_features(base_c, state.circle, env_params, static_env_params)

    return jnp.concatenate([base_p, base_c], axis=0), jnp.concatenate(
        [state.polygon.active, state.circle.active], axis=0
    )


def make_joint_features(
    state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Returns joint_features, indexes, mask, of shape:
    # (2 * J, K), (2 * J, 2), (2 * J,)
    def _create_joint_features(joints):
        # 2, J, A
        J = joints.active.shape[0]

        def _create_1way_joint_features(direction):
            from_pos = jax.lax.select(direction, joints.a_relative_pos, joints.b_relative_pos)
            to_pos = jax.lax.select(direction, joints.b_relative_pos, joints.a_relative_pos)

            rotation_sin, rotation_cos = jnp.sin(joints.rotation), jnp.cos(joints.rotation)
            rotation_max_sin = jnp.sin(joints.max_rotation) * joints.motor_has_joint_limits
            rotation_max_cos = jnp.cos(joints.max_rotation) * joints.motor_has_joint_limits
            rotation_min_sin = jnp.sin(joints.min_rotation) * joints.motor_has_joint_limits
            rotation_min_cos = jnp.cos(joints.min_rotation) * joints.motor_has_joint_limits

            rotation_diff_max = (joints.max_rotation - joints.rotation) * joints.motor_has_joint_limits
            rotation_diff_min = (joints.min_rotation - joints.rotation) * joints.motor_has_joint_limits

            base_features = jnp.concatenate(
                [
                    (joints.active * 1.0)[:, None],
                    (joints.is_fixed_joint * 1.0)[:, None],  # J, 1
                    from_pos,
                    to_pos,
                    rotation_sin[:, None],
                    rotation_cos[:, None],
                ],
                axis=1,
            )
            rjoint_features = (
                jnp.concatenate(
                    [
                        joints.motor_speed[:, None],
                        joints.motor_power[:, None],
                        (joints.motor_on * 1.0)[:, None],
                        (joints.motor_has_joint_limits * 1.0)[:, None],
                        jax.nn.one_hot(state.motor_bindings, num_classes=static_env_params.num_motor_bindings),
                        rotation_min_sin[:, None],
                        rotation_min_cos[:, None],
                        rotation_max_sin[:, None],
                        rotation_max_cos[:, None],
                        rotation_diff_min[:, None],
                        rotation_diff_max[:, None],
                    ],
                    axis=1,
                )
                * (1.0 - (joints.is_fixed_joint * 1.0))[:, None]
            )

            return jnp.concatenate([base_features, rjoint_features], axis=1)

        # 2, J, A
        joint_features = jax.vmap(_create_1way_joint_features)(jnp.array([False, True]))

        # J, 2
        indexes_from = jnp.concatenate([joints.b_index[:, None], joints.a_index[:, None]], axis=1)
        indexes_to = jnp.concatenate([joints.a_index[:, None], joints.b_index[:, None]], axis=1)

        indexes_from = jnp.where(joints.active[:, None], indexes_from, jnp.zeros_like(indexes_from))
        indexes_to = jnp.where(joints.active[:, None], indexes_to, jnp.zeros_like(indexes_to))

        indexes = jnp.concatenate([indexes_from, indexes_to], axis=0)
        mask = jnp.concatenate([joints.active, joints.active], axis=0)

        return joint_features.reshape((2 * J, -1)), indexes, mask

    return _create_joint_features(state.joint)


def make_thruster_features(
    state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Returns thruster_features, indexes, mask, of shape:
    # (T, K), (T,), (T,)
    def _create_thruster_features(thrusters):
        cos = jnp.cos(thrusters.rotation)
        sin = jnp.sin(thrusters.rotation)
        return jnp.concatenate(
            [
                (thrusters.active * 1.0)[:, None],
                (thrusters.relative_position),
                jax.nn.one_hot(state.thruster_bindings, num_classes=static_env_params.num_thruster_bindings),
                sin[:, None],
                cos[:, None],
                thrusters.power[:, None],
            ],
            axis=1,
        )

    return _create_thruster_features(state.thruster), state.thruster.object_index, state.thruster.active
