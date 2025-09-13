from gymnax import EnvParams
import jax
import jax.numpy as jnp
from jax2d.engine import select_shape
from jax2d.maths import rmat
from jax2d.sim_state import RigidBody, Thruster, Joint

from kinetix.environment.env_state import EnvState, StaticEnvParams


def features_to_rigidbody(
    env_params: EnvParams, static_env_params: StaticEnvParams, features: jnp.ndarray, nshapes: int, is_circle: bool
) -> tuple[RigidBody, jnp.ndarray, jnp.ndarray]:
    """Given the rigidbody features from a symbolic observation, convert it to a RigidBody object."""
    nr = env_params.num_shape_roles

    # Circle + Poly

    # shapes.position, 0:2
    # shapes.velocity, 2:4
    # jnp.expand_dims(shapes.inverse_mass, axis=1), 4
    # jnp.expand_dims(shapes.inverse_inertia, axis=1), 5
    # jnp.expand_dims(density, axis=1), 6
    # jnp.expand_dims(jnp.tanh(shapes.angular_velocity / 10), axis=1), 7
    # jax.nn.one_hot(roles, env_params.num_shape_roles), 8:8+roles
    # jnp.expand_dims(sin, axis=1), 8+roles
    # jnp.expand_dims(cos, axis=1), 9+roles
    # jnp.expand_dims(shapes.friction, axis=1), 10+roles
    # jnp.expand_dims(shapes.restitution, axis=1), 11+roles

    # Circle:
    # radius: 12+roles

    # Poly:
    # dummy one hot poly vs circle: 12 + roles
    # vertices: 13+roles : 13+roles + 2 * max_polygon_vertices
    # n_vertices: 13+roles + 2 * max_polygon_vertices

    sin = features[:, 8 + nr]
    cos = features[:, 9 + nr]
    rotation = jnp.atan2(sin, cos)
    rotation = jnp.where(rotation < 0, rotation + 2 * jnp.pi, rotation)

    rb_shape_roles = jnp.argmax(features[:, 8 : 8 + nr], axis=1)
    rb_densities = features[:, 6]

    rb = RigidBody(
        position=features[:, :2],
        velocity=features[:, 2:4],
        inverse_mass=features[:, 4],
        inverse_inertia=features[:, 5],
        angular_velocity=jnp.atanh(features[:, 7]) * 10.0,
        rotation=rotation,
        friction=features[:, 10 + nr],
        restitution=features[:, 11 + nr],
        collision_mode=jnp.ones(nshapes, dtype=jnp.int32),
        active=jnp.zeros(nshapes, dtype=jnp.bool),
        n_vertices=jnp.zeros(nshapes, dtype=jnp.int32),
        vertices=jnp.zeros((nshapes, static_env_params.max_polygon_vertices, 2)),
        radius=jnp.zeros(nshapes),
    )

    if is_circle:
        rb = rb.replace(
            radius=features[:, 12 + nr],
            active=features[:, 12 + nr] > 0.0,
        )
    else:

        mpv = static_env_params.max_polygon_vertices
        has_three_vertices = features[:, 13 + nr + 2 * mpv]

        rb = rb.replace(
            vertices=features[:, 13 + nr : 13 + nr + 2 * mpv].reshape((-1, mpv, 2)),
            n_vertices=jnp.where(
                has_three_vertices,
                jnp.ones_like(has_three_vertices, dtype=jnp.int32) * 3,
                jnp.ones_like(has_three_vertices, dtype=jnp.int32) * 4,
            ),
            collision_mode=rb.collision_mode.at[..., 0].set(2),
        )
        rb = rb.replace(active=jnp.abs(rb.vertices.reshape(rb.vertices.shape[0], -1)).sum(axis=1) > 0.0)
    return rb, rb_shape_roles, rb_densities


def features_to_thruster(
    static_env_params: StaticEnvParams,
    env_state: EnvState,
    thruster_features: jnp.ndarray,
    nshapes: int,
    thruster_idxs_override=None,
):
    # (thrusters.active * 1.0)[:, None], 0
    # (thrusters.relative_position), 1:3
    # jax.nn.one_hot(state.thruster_bindings, num_classes=static_env_params.num_thruster_bindings), 3:3+ntb
    # sin[:, None], 3+ntb
    # cos[:, None], 4+ntb
    # thrusters.power[:, None], 5+ntb
    # jax.nn.one_hot(thruster_idxs, nshapes) 5+ntb:5+ntb+nshapes

    ntb = static_env_params.num_thruster_bindings
    if thruster_idxs_override is None:
        thruster_idxs = jnp.argmax(thruster_features[:, 6 + ntb : 6 + ntb + nshapes], axis=1)
    else:
        thruster_idxs = thruster_idxs_override

    sin = thruster_features[:, 3 + ntb]
    cos = thruster_features[:, 4 + ntb]
    rotation = jnp.atan2(sin, cos)
    rotation = jnp.where(rotation < 0, rotation + 2 * jnp.pi, rotation)

    thruster_bindings = jnp.argmax(thruster_features[:, 3 : 3 + ntb], axis=1)

    thruster_shapes = jax.vmap(select_shape, in_axes=(None, 0, None))(env_state, thruster_idxs, static_env_params)

    active = thruster_features[:, 0] > 0.0

    thruster_relative_position = thruster_features[:, 1:3]
    thruster_global_position = thruster_shapes.position + jax.vmap(jnp.matmul)(
        jax.vmap(rmat)(thruster_shapes.rotation), thruster_relative_position
    )

    thruster_global_position = jnp.where(
        active[:, None], thruster_global_position, jnp.zeros_like(thruster_global_position)
    )
    return (
        Thruster(
            active=active,
            relative_position=thruster_relative_position,
            rotation=rotation,
            power=thruster_features[:, 5 + ntb],
            global_position=thruster_global_position,
            object_index=thruster_idxs,
        ),
        thruster_bindings,
    )


def features_to_joint(
    static_env_params: StaticEnvParams,
    env_state: EnvState,
    joint_features: jnp.ndarray,
    nshapes: int,
    joint_idxs_override=None,
):
    # (joints.active * 1.0)[:, None] 0
    # (joints.is_fixed_joint * 1.0)[:, None], 1
    # from_pos 2:4
    # to_pos 4:6
    # rotation_sin[:, None] 6
    # rotation_cos[:, None] 7
    # joints.motor_speed[:, None], 8
    # joints.motor_power[:, None], 9
    # (joints.motor_on * 1.0)[:, None], 10
    # (joints.motor_has_joint_limits * 1.0)[:, None], 11
    # jax.nn.one_hot(state.motor_bindings, num_classes=static_env_params.num_motor_bindings), 12:12+nmb
    # rotation_min_sin[:, None], 12+nmb
    # rotation_min_cos[:, None], 13+nmb
    # rotation_max_sin[:, None], 14+nmb
    # rotation_max_cos[:, None], 15+nmb
    # rotation_diff_min[:, None], 16+nmb
    # rotation_diff_max[:, None], 17+nmb
    # jax.nn.one_hot(joint_idxs[:J, 0], nshapes),  18+nmb:18+nmb+nshapes
    # jax.nn.one_hot(joint_idxs[:J, 1], nshapes),  18+nmb+nshapes:18+nmb+2*nshapes
    nmb = static_env_params.num_motor_bindings

    if joint_idxs_override is None:
        b_index = jnp.argmax(joint_features[:, 18 + nmb : 18 + nmb + nshapes], axis=1)
        a_index = jnp.argmax(joint_features[:, 18 + nmb + nshapes : 18 + nmb + nshapes * 2], axis=1)
    else:
        b_index = joint_idxs_override[:, 0]
        a_index = joint_idxs_override[:, 1]

    b_relative_pos = joint_features[:, 2:4]
    a_relative_pos = joint_features[:, 4:6]
    joint_rotation = jnp.arctan2(joint_features[:, 6], joint_features[:, 7])
    joint_rotation = jnp.where(joint_rotation < 0, joint_rotation + 2 * jnp.pi, joint_rotation)

    joint_shapes = jax.vmap(select_shape, in_axes=(None, 0, None))(env_state, a_index, static_env_params)
    active = joint_features[:, 0] > 0.0

    joint_global_position = joint_shapes.position + jax.vmap(jnp.matmul)(
        jax.vmap(rmat)(joint_shapes.rotation), a_relative_pos
    )
    joint_global_position = jnp.where(active[:, None], joint_global_position, jnp.zeros_like(joint_global_position))

    return (
        Joint(
            active=active,
            is_fixed_joint=joint_features[:, 1] > 0.0,
            a_index=a_index,
            b_index=b_index,
            a_relative_pos=a_relative_pos,
            b_relative_pos=b_relative_pos,
            rotation=joint_rotation,
            motor_speed=joint_features[:, 8],
            motor_power=joint_features[:, 9],
            motor_on=joint_features[:, 10] > 0.0,
            motor_has_joint_limits=joint_features[:, 11] > 0.0,
            min_rotation=jnp.arctan2(joint_features[:, 12 + nmb], joint_features[:, 13 + nmb]),
            max_rotation=jnp.arctan2(joint_features[:, 14 + nmb], joint_features[:, 15 + nmb]),
            global_position=joint_global_position,
            acc_r_impulse=jnp.zeros((static_env_params.num_joints), dtype=jnp.float32),
            acc_impulse=jnp.zeros((static_env_params.num_joints, 2), dtype=jnp.float32),
        ),
        jnp.argmax(joint_features[:, 12 : 12 + nmb], axis=1),
    )
