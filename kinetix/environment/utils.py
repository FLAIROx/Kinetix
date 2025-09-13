from functools import partial
from typing import Optional, Union

import chex
import jax
import jax.numpy as jnp
from jax2d.engine import calculate_collision_matrix, create_empty_sim
from jax2d.sim_state import SimState

from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams


@partial(jax.jit, static_argnums=(0,))
def create_empty_env(static_env_params):
    sim_state = create_empty_sim(static_env_params)
    return EnvState(
        timestep=0,
        last_distance=-1.0,
        thruster_bindings=jnp.zeros(static_env_params.num_thrusters, dtype=jnp.int32),
        motor_bindings=jnp.zeros(static_env_params.num_joints, dtype=jnp.int32),
        motor_auto=jnp.zeros(static_env_params.num_joints, dtype=bool),
        polygon_shape_roles=jnp.zeros(static_env_params.num_polygons, dtype=jnp.int32),
        circle_shape_roles=jnp.zeros(static_env_params.num_circles, dtype=jnp.int32),
        polygon_highlighted=jnp.zeros(static_env_params.num_polygons, dtype=bool),
        circle_highlighted=jnp.zeros(static_env_params.num_circles, dtype=bool),
        polygon_densities=jnp.ones(static_env_params.num_polygons, dtype=jnp.float32),
        circle_densities=jnp.ones(static_env_params.num_circles, dtype=jnp.float32),
        **sim_state.__dict__,
    )


@jax.jit
def index_motor_actions(
    action: jnp.ndarray,
    state: EnvState,
    clip_min=None,
    clip_max=None,
):
    # Expand the motor actions to all joints with the same colour
    return jnp.clip(action[state.motor_bindings], clip_min, clip_max)


@jax.jit
def index_thruster_actions(
    action: jnp.ndarray,
    state: EnvState,
    clip_min=None,
    clip_max=None,
):
    # Expand the thruster actions to all joints with the same colour
    return jnp.clip(action[state.thruster_bindings], clip_min, clip_max)


@partial(jax.jit, static_argnums=(2,))
def convert_continuous_actions(
    action: jnp.ndarray, state: SimState, static_env_params: StaticEnvParams, env_params: EnvParams
):
    action_motor = action[: static_env_params.num_motor_bindings]
    action_thruster = action[static_env_params.num_motor_bindings :]
    action_motor = index_motor_actions(action_motor, state, -1, 1)
    action_thruster = index_thruster_actions(action_thruster, state, 0, 1)

    action_motor = jnp.where(state.motor_auto, jnp.ones_like(action_motor), action_motor)

    action_to_perform = jnp.concatenate([action_motor, action_thruster], axis=0)
    return action_to_perform


@partial(jax.jit, static_argnums=(2,))
def convert_discrete_actions(action: int, state: SimState, static_env_params: StaticEnvParams, env_params: EnvParams):
    # so, we have
    # 0 to NJC * 2 - 1: Joint Actions
    # NJC * 2: No-op
    # NJC * 2 + 1 to NJC * 2 + 1 + NTC - 1: Thruster Actions
    # action here is a categorical action
    which_idx = action // 2
    which_dir = action % 2
    actions = (
        jnp.zeros(static_env_params.num_motor_bindings + static_env_params.num_thruster_bindings)
        .at[which_idx]
        .set(which_dir * 2 - 1)
    )
    actions = actions * (
        1 - (action >= static_env_params.num_motor_bindings * 2)
    )  # if action is the last one, set it to zero, i.e., a no-op. Alternatively, if the action is larger than NJC * 2, then it is a thruster action and we shouldn't control the joints.

    actions = jax.lax.select(
        action > static_env_params.num_motor_bindings * 2,
        actions.at[action - static_env_params.num_motor_bindings * 2 - 1 + static_env_params.num_motor_bindings].set(1),
        actions,
    )

    action_motor = index_motor_actions(actions[: static_env_params.num_motor_bindings], state, -1, 1)
    action_motor = jnp.where(state.motor_auto, jnp.ones_like(action_motor), action_motor)
    action_thruster = index_thruster_actions(actions[static_env_params.num_motor_bindings :], state, 0, 1)
    action_to_perform = jnp.concatenate([action_motor, action_thruster], axis=0)
    return action_to_perform


@partial(jax.jit, static_argnums=(2,))
def convert_multi_discrete_actions(
    action: jnp.ndarray, state: SimState, static_env_params: StaticEnvParams, env_params: EnvParams
):
    # Comes in with each action being in {0,1,2} for joints and {0,1} for thrusters
    # Convert to [-1., 1.] for joints and [0., 1.] for thrusters

    def _single_motor_action(act):
        return jax.lax.switch(
            act,
            [lambda: 0.0, lambda: 1.0, lambda: -1.0],
        )

    def _single_thruster_act(act):
        return jax.lax.select(
            act == 0,
            0.0,
            1.0,
        )

    action_motor = jax.vmap(_single_motor_action)(action[: static_env_params.num_motor_bindings])
    action_thruster = jax.vmap(_single_thruster_act)(action[static_env_params.num_motor_bindings :])

    action_motor = index_motor_actions(action_motor, state, -1, 1)
    action_thruster = index_thruster_actions(action_thruster, state, 0, 1)

    action_motor = jnp.where(state.motor_auto, jnp.ones_like(action_motor), action_motor)

    action_to_perform = jnp.concatenate([action_motor, action_thruster], axis=0)
    return action_to_perform


@partial(jax.jit, static_argnums=(2,))
def permute_state(rng: chex.PRNGKey, env_state: EnvState, static_env_params: StaticEnvParams):
    idxs_circles = jnp.arange(static_env_params.num_circles)
    idxs_polygons = jnp.arange(static_env_params.num_polygons)
    idxs_joints = jnp.arange(static_env_params.num_joints)
    idxs_thrusters = jnp.arange(static_env_params.num_thrusters)

    rng, *_rngs = jax.random.split(rng, 5)
    idxs_circles_permuted = jax.random.permutation(_rngs[0], idxs_circles, independent=True)
    idxs_polygons_permuted = idxs_polygons.at[static_env_params.num_static_fixated_polys :].set(
        jax.random.permutation(_rngs[1], idxs_polygons[static_env_params.num_static_fixated_polys :], independent=True)
    )

    idxs_joints_permuted = jax.random.permutation(_rngs[2], idxs_joints, independent=True)
    idxs_thrusters_permuted = jax.random.permutation(_rngs[3], idxs_thrusters, independent=True)

    combined = jnp.concatenate([idxs_polygons_permuted, idxs_circles_permuted + static_env_params.num_polygons])
    # Change the ordering of the shapes, and also remember to change the indices associated with the joints

    inverse_permutation = jnp.argsort(combined)

    env_state = env_state.replace(
        polygon_shape_roles=env_state.polygon_shape_roles[idxs_polygons_permuted],
        circle_shape_roles=env_state.circle_shape_roles[idxs_circles_permuted],
        polygon_highlighted=env_state.polygon_highlighted[idxs_polygons_permuted],
        circle_highlighted=env_state.circle_highlighted[idxs_circles_permuted],
        polygon_densities=env_state.polygon_densities[idxs_polygons_permuted],
        circle_densities=env_state.circle_densities[idxs_circles_permuted],
        polygon=jax.tree.map(lambda x: x[idxs_polygons_permuted], env_state.polygon),
        circle=jax.tree.map(lambda x: x[idxs_circles_permuted], env_state.circle),
        joint=env_state.joint.replace(
            a_index=inverse_permutation[env_state.joint.a_index],
            b_index=inverse_permutation[env_state.joint.b_index],
        ),
        thruster=env_state.thruster.replace(
            object_index=inverse_permutation[env_state.thruster.object_index],
        ),
    )

    # And now permute the thrusters and joints
    env_state = env_state.replace(
        thruster_bindings=env_state.thruster_bindings[idxs_thrusters_permuted],
        motor_bindings=env_state.motor_bindings[idxs_joints_permuted],
        motor_auto=env_state.motor_auto[idxs_joints_permuted],
        joint=jax.tree.map(lambda x: x[idxs_joints_permuted], env_state.joint),
        thruster=jax.tree.map(lambda x: x[idxs_thrusters_permuted], env_state.thruster),
    )
    # and collision matrix
    env_state = env_state.replace(collision_matrix=calculate_collision_matrix(static_env_params, env_state.joint))
    return env_state
