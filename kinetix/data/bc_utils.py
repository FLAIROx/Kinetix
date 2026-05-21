from functools import partial
import jax
import jax.numpy as jnp
from jax2d.engine import CollisionManifold, Joint, RigidBody, Thruster
import numpy as np

from kinetix.data.bc_types import ActionEnvStateMask
from kinetix.environment.env_state import EnvState, StaticEnvParams
from kinetix.data.expanding_utils import expand_env_state_numpy, expand_env_state_numpy_batched


def map_raw_dict_to_action_env_state(raw_data: dict) -> ActionEnvStateMask:
    return ActionEnvStateMask(
        action=raw_data["action"],
        mask=raw_data["mask"],
        action_mask=raw_data["action_mask"],
        env_state=EnvState(
            timestep=raw_data["env_state/timestep"],
            last_distance=raw_data["env_state/last_distance"],
            gravity=raw_data["env_state/gravity"],
            collision_matrix=raw_data["env_state/collision_matrix"],
            # Manifolds
            acc_cc_manifolds=CollisionManifold(
                acc_impulse_normal=raw_data["env_state/acc_cc_manifolds/acc_impulse_normal"],
                acc_impulse_tangent=raw_data["env_state/acc_cc_manifolds/acc_impulse_tangent"],
                active=raw_data["env_state/acc_cc_manifolds/active"],
                collision_point=raw_data["env_state/acc_cc_manifolds/collision_point"],
                normal=raw_data["env_state/acc_cc_manifolds/normal"],
                penetration=raw_data["env_state/acc_cc_manifolds/penetration"],
                restitution_velocity_target=raw_data["env_state/acc_cc_manifolds/restitution_velocity_target"],
            ),
            acc_cr_manifolds=CollisionManifold(
                acc_impulse_normal=raw_data["env_state/acc_cr_manifolds/acc_impulse_normal"],
                acc_impulse_tangent=raw_data["env_state/acc_cr_manifolds/acc_impulse_tangent"],
                active=raw_data["env_state/acc_cr_manifolds/active"],
                collision_point=raw_data["env_state/acc_cr_manifolds/collision_point"],
                normal=raw_data["env_state/acc_cr_manifolds/normal"],
                penetration=raw_data["env_state/acc_cr_manifolds/penetration"],
                restitution_velocity_target=raw_data["env_state/acc_cr_manifolds/restitution_velocity_target"],
            ),
            acc_rr_manifolds=CollisionManifold(
                acc_impulse_normal=raw_data["env_state/acc_rr_manifolds/acc_impulse_normal"],
                acc_impulse_tangent=raw_data["env_state/acc_rr_manifolds/acc_impulse_tangent"],
                active=raw_data["env_state/acc_rr_manifolds/active"],
                collision_point=raw_data["env_state/acc_rr_manifolds/collision_point"],
                normal=raw_data["env_state/acc_rr_manifolds/normal"],
                penetration=raw_data["env_state/acc_rr_manifolds/penetration"],
                restitution_velocity_target=raw_data["env_state/acc_rr_manifolds/restitution_velocity_target"],
            ),
            # Rigid Bodies
            polygon=RigidBody(
                active=raw_data["env_state/polygon/active"],
                angular_velocity=raw_data["env_state/polygon/angular_velocity"],
                collision_mode=raw_data["env_state/polygon/collision_mode"],
                friction=raw_data["env_state/polygon/friction"],
                inverse_inertia=raw_data["env_state/polygon/inverse_inertia"],
                inverse_mass=raw_data["env_state/polygon/inverse_mass"],
                n_vertices=raw_data["env_state/polygon/n_vertices"],
                position=raw_data["env_state/polygon/position"],
                radius=raw_data["env_state/polygon/radius"],
                restitution=raw_data["env_state/polygon/restitution"],
                rotation=raw_data["env_state/polygon/rotation"],
                velocity=raw_data["env_state/polygon/velocity"],
                vertices=raw_data["env_state/polygon/vertices"],
            ),
            circle=RigidBody(
                active=raw_data["env_state/circle/active"],
                angular_velocity=raw_data["env_state/circle/angular_velocity"],
                collision_mode=raw_data["env_state/circle/collision_mode"],
                friction=raw_data["env_state/circle/friction"],
                inverse_inertia=raw_data["env_state/circle/inverse_inertia"],
                inverse_mass=raw_data["env_state/circle/inverse_mass"],
                n_vertices=raw_data["env_state/circle/n_vertices"],
                position=raw_data["env_state/circle/position"],
                radius=raw_data["env_state/circle/radius"],
                restitution=raw_data["env_state/circle/restitution"],
                rotation=raw_data["env_state/circle/rotation"],
                velocity=raw_data["env_state/circle/velocity"],
                vertices=raw_data["env_state/circle/vertices"],
            ),
            # Joints
            joint=Joint(
                a_index=raw_data["env_state/joint/a_index"],
                a_relative_pos=raw_data["env_state/joint/a_relative_pos"],
                acc_impulse=raw_data["env_state/joint/acc_impulse"],
                acc_r_impulse=raw_data["env_state/joint/acc_r_impulse"],
                active=raw_data["env_state/joint/active"],
                b_index=raw_data["env_state/joint/b_index"],
                b_relative_pos=raw_data["env_state/joint/b_relative_pos"],
                global_position=raw_data["env_state/joint/global_position"],
                is_fixed_joint=raw_data["env_state/joint/is_fixed_joint"],
                max_rotation=raw_data["env_state/joint/max_rotation"],
                min_rotation=raw_data["env_state/joint/min_rotation"],
                motor_has_joint_limits=raw_data["env_state/joint/motor_has_joint_limits"],
                motor_on=raw_data["env_state/joint/motor_on"],
                motor_power=raw_data["env_state/joint/motor_power"],
                motor_speed=raw_data["env_state/joint/motor_speed"],
                rotation=raw_data["env_state/joint/rotation"],
            ),
            # Thrusters
            thruster=Thruster(
                active=raw_data["env_state/thruster/active"],
                global_position=raw_data["env_state/thruster/global_position"],
                object_index=raw_data["env_state/thruster/object_index"],
                power=raw_data["env_state/thruster/power"],
                relative_position=raw_data["env_state/thruster/relative_position"],
                rotation=raw_data["env_state/thruster/rotation"],
            ),
            # Meta properties
            circle_densities=raw_data["env_state/circle_densities"],
            circle_highlighted=raw_data["env_state/circle_highlighted"],
            circle_shape_roles=raw_data["env_state/circle_shape_roles"],
            polygon_densities=raw_data["env_state/polygon_densities"],
            polygon_highlighted=raw_data["env_state/polygon_highlighted"],
            polygon_shape_roles=raw_data["env_state/polygon_shape_roles"],
            motor_auto=raw_data["env_state/motor_auto"],
            motor_bindings=raw_data["env_state/motor_bindings"],
            thruster_bindings=raw_data["env_state/thruster_bindings"],
        ),
    )


@partial(jax.jit, static_argnums=(1,))
def get_valid_action_mask(env_state: EnvState, static_env_params: StaticEnvParams, action: jax.Array):
    if hasattr(env_state, "env_state"):
        env_state = env_state.env_state
    assert len(action.shape) == 2  # (transitions, action dim)

    mask = jnp.zeros_like(action, dtype=bool)
    arange = np.arange(len(mask))
    joint_motor_bindings = env_state.motor_bindings
    assert joint_motor_bindings.shape[-1] == static_env_params.num_joints
    for i in range(static_env_params.num_joints):
        current = mask[arange, joint_motor_bindings[:, i]]

        mask = mask.at[arange, joint_motor_bindings[:, i]].set(
            jnp.logical_or(
                current, env_state.joint.active[:, i] * jnp.logical_not(env_state.joint.is_fixed_joint[:, i])
            )
        )
    thruster_bindings = env_state.thruster_bindings
    assert thruster_bindings.shape[-1] == static_env_params.num_thrusters
    for i in range(static_env_params.num_thrusters):
        current = mask[
            arange,
            thruster_bindings[:, i] + static_env_params.num_motor_bindings,
        ]
        mask = mask.at[
            arange,
            thruster_bindings[:, i] + static_env_params.num_motor_bindings,
        ].set(jnp.logical_or(current, env_state.thruster.active[:, i]))

    return mask


def get_valid_action_mask_np(env_state, static_env_params: StaticEnvParams, action: np.ndarray) -> np.ndarray:
    """Pure-numpy version of bc_utils.get_valid_action_mask — avoids any GPU/JAX dispatch."""
    if hasattr(env_state, "env_state"):
        env_state = env_state.env_state
    assert len(action.shape) == 2  # (transitions, action_dim)

    mask = np.zeros_like(action, dtype=bool)
    arange = np.arange(len(mask))
    joint_motor_bindings = np.asarray(env_state.motor_bindings)
    for i in range(static_env_params.num_joints):
        idx = joint_motor_bindings[:, i]
        mask[arange, idx] = np.logical_or(
            mask[arange, idx],
            np.asarray(env_state.joint.active[:, i]) & ~np.asarray(env_state.joint.is_fixed_joint[:, i]),
        )
    thruster_bindings = np.asarray(env_state.thruster_bindings)
    for i in range(static_env_params.num_thrusters):
        idx = thruster_bindings[:, i] + static_env_params.num_motor_bindings
        mask[arange, idx] = np.logical_or(
            mask[arange, idx],
            np.asarray(env_state.thruster.active[:, i]),
        )
    return mask
