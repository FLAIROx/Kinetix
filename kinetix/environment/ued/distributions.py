from functools import partial

import jax
import jax.numpy as jnp
from flax.serialization import to_state_dict
from jax2d.engine import PhysicsEngine

from kinetix.environment.env_state import EnvParams, StaticEnvParams
from kinetix.environment.ued.mutators import (
    mutate_add_connected_shape,
    mutate_add_connected_shape_proper,
    mutate_add_shape,
    mutate_add_thruster,
)
from kinetix.environment.ued.ued_state import UEDParams
from kinetix.environment.utils import create_empty_env, permute_state


# @partial(jax.jit, static_argnums=(1, 3, 5, 6, 7, 8, 9, 10))
def create_vmapped_filtered_distribution(
    rng,
    level_sampler,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    ued_params: UEDParams,
    n_samples: int,
    env,
    do_filter_levels: bool,
    level_filter_sample_ratio: int,
    env_size_name: str,
    level_filter_n_steps: int,
):

    if do_filter_levels and level_filter_n_steps > 0:
        sample_ratio = level_filter_sample_ratio
        n_unfiltered_samples = sample_ratio * n_samples
        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, n_unfiltered_samples)

        # unfiltered_levels = jax.vmap(level_sampler, in_axes=(0, None, None, None, None))(
        #     _rngs, env_params, static_env_params, ued_params, env_size_name
        # )
        unfiltered_levels = jax.vmap(level_sampler, in_axes=(0,))(_rngs)
        #

        # No-op filtering

        def _noop_step(carry, rng):
            states, start_state = carry
            rng, _rng = jax.random.split(rng)
            _rngs = jax.random.split(_rng, n_unfiltered_samples)

            action = jnp.zeros((n_unfiltered_samples, *env.action_space(env_params).shape), dtype=jnp.int32)

            obs, states, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None, 0))(
                _rngs, states, action, env_params, start_state
            )

            return (states, start_state), (done, reward)

        # Wrap levels
        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, n_unfiltered_samples)
        obsv, unfiltered_levels_wrapped = jax.vmap(env.reset, in_axes=(0, None, 0))(
            _rngs, env_params, unfiltered_levels
        )

        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, level_filter_n_steps)
        _, (done, rewards) = jax.lax.scan(_noop_step, (unfiltered_levels_wrapped, unfiltered_levels_wrapped), xs=_rngs)

        done_indexes = jnp.argmax(done, axis=0)
        done_rewards = rewards[done_indexes, jnp.arange(n_unfiltered_samples)]

        noop_solved_indexes = done_rewards > 0.5
        p = noop_solved_indexes * 0.001 + (1 - noop_solved_indexes) * 1.0
        p /= p.sum()

        rng, _rng = jax.random.split(rng)
        level_indexes = jax.random.choice(
            _rng, jnp.arange(n_unfiltered_samples), shape=(n_samples,), replace=False, p=p
        )

        levels = jax.tree.map(lambda x: x[level_indexes], unfiltered_levels)

    else:
        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, n_samples)

        levels = jax.vmap(level_sampler, in_axes=(0,))(_rngs)

    return levels


@partial(jax.jit, static_argnums=(1, 3, 4))
def sample_kinetix_level(
    rng,
    engine: PhysicsEngine,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    ued_params: UEDParams,
):
    rng, _rng = jax.random.split(rng)
    _rngs = jax.random.split(_rng, 12)

    small_force_no_fixate = static_env_params.num_polygons + static_env_params.num_circles <= 7

    add_shape_n_proposals = ued_params.add_shape_n_proposals
    if static_env_params.num_polygons + static_env_params.num_circles <= 7:
        add_shape_n_proposals = 1  # otherwise we get a very weird XLA bug.

    # Start with empty state
    state = create_empty_env(static_env_params)

    # Set the floor
    prob_of_floor_colour = jnp.array(
        [
            ued_params.floor_prob_normal,
            ued_params.floor_prob_green,
            ued_params.floor_prob_blue,
            ued_params.floor_prob_red,
        ]
    )
    floor_colour = jax.random.choice(_rngs[0], jnp.arange(4), p=prob_of_floor_colour)
    state = state.replace(polygon_shape_roles=state.polygon_shape_roles.at[0].set(floor_colour))

    # When we add shapes we don't want them to collide with already existing shapes
    def _choose_proposal_with_least_collisions(proposals, bias=None):
        rr, cr, cc = jax.vmap(engine.calculate_collision_manifolds)(proposals)

        rr_collisions = jnp.sum(jnp.sum(rr.active.astype(jnp.int32), axis=-1), axis=-1)
        cr_collisions = jnp.sum(cr.active.astype(jnp.int32), axis=-1)
        cc_collisions = jnp.sum(cc.active.astype(jnp.int32), axis=-1)

        all_collisions = jnp.concatenate(
            [rr_collisions[:, None], cr_collisions[:, None], cc_collisions[:, None]], axis=1
        )
        num_collisions = jnp.sum(all_collisions, axis=-1)
        if bias is not None:
            num_collisions = num_collisions + bias

        chosen_addition_idx = jnp.argmin(num_collisions)

        return jax.tree.map(lambda x: x[chosen_addition_idx], proposals)

    def _add_filtered_shape(rng, state, force_no_fixate=False):
        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, add_shape_n_proposals)
        proposed_additions = jax.vmap(mutate_add_shape, in_axes=(0, None, None, None, None, None))(
            _rngs,
            state,
            env_params,
            static_env_params,
            ued_params,
            jnp.logical_or(force_no_fixate, small_force_no_fixate),
        )

        return _choose_proposal_with_least_collisions(proposed_additions)

    def _add_filtered_connected_shape(rng, state, force_rjoint=False):
        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, add_shape_n_proposals)

        proposed_additions, valid = jax.vmap(mutate_add_connected_shape, in_axes=(0, None, None, None, None, None))(
            _rngs, state, env_params, static_env_params, ued_params, force_rjoint
        )

        bias = (jnp.ones(add_shape_n_proposals) - 1 * valid) * ued_params.connect_no_visibility_bias

        return _choose_proposal_with_least_collisions(proposed_additions, bias=bias)

    # Add green and blue - make sure they're not both fixated
    force_green_no_fixate = (jax.random.uniform(_rngs[1]) < 0.5) | (state.polygon_shape_roles[0] == 2)
    state = _add_filtered_shape(_rngs[2], state, force_green_no_fixate)
    state = _add_filtered_shape(_rngs[3], state, ~force_green_no_fixate)

    # Forced controls
    forced_control = jnp.array([[0, 1], [1, 0], [1, 1]])[jax.random.randint(_rngs[4], (), 0, 3)]
    force_thruster, force_motor = forced_control[0], forced_control[1]

    # Forced motor
    state = jax.lax.cond(
        force_motor,
        lambda: _add_filtered_connected_shape(_rngs[5], state, force_rjoint=True),  # force the rjoint
        lambda: _add_filtered_shape(_rngs[6], state),
    )

    # Forced thruster
    state = jax.lax.cond(
        force_thruster,
        lambda: mutate_add_thruster(_rngs[7], state, env_params, static_env_params, ued_params),
        lambda: state,
    )

    # Add rest of shapes
    n_shapes_to_add = (
        static_env_params.num_polygons + static_env_params.num_circles - 3 - static_env_params.num_static_fixated_polys
    )

    def _add_shape(state, rng):
        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, 3)
        shape_add_type = jax.random.choice(
            _rngs[0],
            jnp.arange(3),
            p=jnp.array(
                [ued_params.add_connected_shape_chance, ued_params.add_shape_chance, ued_params.add_no_shape_chance]
            ),
        )

        state = jax.lax.switch(
            shape_add_type,
            [
                lambda: _add_filtered_connected_shape(_rngs[1], state),
                lambda: _add_filtered_shape(_rngs[2], state),
                lambda: state,
            ],
        )

        return state, None

    state, _ = jax.lax.scan(_add_shape, state, jax.random.split(_rngs[8], n_shapes_to_add))

    # Add thrusters
    n_thrusters_to_add = static_env_params.num_thrusters - 1

    def _add_thruster(state, rng):
        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, 3)
        state = jax.lax.cond(
            jax.random.uniform(_rngs[0]) < ued_params.add_thruster_chance,
            lambda: mutate_add_thruster(_rngs[1], state, env_params, static_env_params, ued_params),
            lambda: state,
        )

        return state, None

    state, _ = jax.lax.scan(_add_thruster, state, jax.random.split(_rngs[9], n_thrusters_to_add))

    # Randomly swap green and blue to remove left-right bias
    def _swap_roles(do_swap_roles, roles):
        role1 = roles == 1
        role2 = roles == 2

        swapped_roles = roles * ~(role1 | role2) + role1.astype(int) * 2 + role2.astype(int) * 1
        return jax.lax.select(do_swap_roles, swapped_roles, roles)

    do_swap_roles = jax.random.uniform(_rngs[10], shape=()) < 0.5
    # Don't want to swap if floor is non-standard
    do_swap_roles &= state.polygon_shape_roles[0] == 0
    state = state.replace(
        polygon_shape_roles=_swap_roles(do_swap_roles, state.polygon_shape_roles),
        circle_shape_roles=_swap_roles(do_swap_roles, state.circle_shape_roles),
    )

    return permute_state(_rngs[11], state, static_env_params)


@partial(jax.jit, static_argnums=(2, 4, 5))
def create_random_starting_distribution(
    rng,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    ued_params: UEDParams,
    env_size_name: str,
    controllable=True,
):
    rng, _rng = jax.random.split(rng)
    _rngs = jax.random.split(_rng, 15)
    d = to_state_dict(ued_params)
    ued_params = UEDParams(
        **(
            d
            | dict(
                goal_body_size_factor=2.0,
                thruster_power_multiplier=2.0,
                max_shape_size=0.5,
            )
        ),
    )

    prob_of_large_shapes = 0.05

    ued_params_large_shapes = ued_params.replace(
        max_shape_size=static_env_params.max_shape_size * 1.0, goal_body_size_factor=1.0
    )

    state = create_empty_env(env_params, static_env_params)

    def _get_ued_params(rng):
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        large_shapes = jax.random.uniform(_rng) < prob_of_large_shapes
        params_to_use = jax.tree.map(
            lambda x, y: jax.lax.select(large_shapes, x, y), ued_params_large_shapes, ued_params
        )
        return params_to_use

    def _my_add_shape(rng, state):
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        return mutate_add_shape(_rng, state, env_params, static_env_params, _get_ued_params(_rng2))

    def _my_add_connected_shape(rng, state, **kwargs):
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        return mutate_add_connected_shape_proper(
            _rng, state, env_params, static_env_params, _get_ued_params(_rng2), **kwargs
        )

    # Add the green thing and blue thing
    state = _my_add_shape(_rngs[0], state)
    state = _my_add_shape(_rngs[1], state)
    if controllable:
        # Forced controls
        forced_control = jnp.array([[0, 1], [1, 0], [1, 1]])[jax.random.randint(_rngs[2], (), 0, 3)]
        force_thruster, force_motor = forced_control[0], forced_control[1]

        # Forced motor
        state = jax.lax.cond(
            force_motor,
            lambda: _my_add_connected_shape(_rngs[3], state, force_rjoint=True),  # force the rjoint
            lambda: state,
        )

        # Forced thruster
        state = jax.lax.cond(
            force_thruster,
            lambda: mutate_add_thruster(_rngs[4], state, env_params, static_env_params, ued_params),
            lambda: state,
        )
    return permute_state(_rngs[7], state, static_env_params)
