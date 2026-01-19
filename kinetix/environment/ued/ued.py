from enum import Enum
from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
from jax2d.engine import PhysicsEngine

from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.environment.ued.distributions import create_vmapped_filtered_distribution, sample_kinetix_level
from kinetix.environment.ued.mutators import (
    make_mutate_change_shape_rotation,
    make_mutate_change_shape_size,
    mutate_add_connected_shape,
    mutate_add_connected_shape_proper,
    mutate_add_shape,
    mutate_add_thruster,
    mutate_change_gravity,
    mutate_change_shape_location,
    mutate_remove_joint,
    mutate_remove_shape,
    mutate_remove_thruster,
    mutate_swap_role,
    mutate_toggle_fixture,
)
from kinetix.environment.ued.ued_state import UEDParams
from kinetix.environment.utils import create_empty_env
from kinetix.util.saving import load_evaluation_levels


def make_mutate_env(static_env_params: StaticEnvParams, env_params: EnvParams, ued_params: UEDParams):
    mutate_size = make_mutate_change_shape_size(env_params, static_env_params)
    mutate_rot = make_mutate_change_shape_rotation(env_params, static_env_params)

    def mutate_level(rng, level: EnvState, n=1):
        def inner(carry: tuple[chex.PRNGKey, EnvState], _):
            rng, level = carry
            rng, _rng, _rng2 = jax.random.split(rng, 3)

            any_rects_left = jnp.logical_not(level.polygon.active).sum() > 0
            any_circles_left = jnp.logical_not(level.circle.active).sum() > 0
            any_joints_left = jnp.logical_not(level.joint.active).sum() > 0
            any_thrust_left = jnp.logical_not(level.thruster.active).sum() > 0
            has_any_thursters = level.thruster.active.sum() > 0

            can_do_add_shape = any_rects_left | any_circles_left
            can_do_add_joint = can_do_add_shape & any_joints_left

            all_mutations = [
                mutate_add_shape,
                mutate_add_connected_shape_proper,
                mutate_remove_joint,
                mutate_remove_shape,
                mutate_swap_role,
                mutate_add_thruster,
                mutate_remove_thruster,
                mutate_toggle_fixture,
                mutate_size,
                mutate_change_shape_location,
                mutate_rot,
            ]

            def mypartial(f):
                def inner(rng, level):
                    return f(rng, level, env_params, static_env_params, ued_params)

                return inner

            probs = jnp.array(
                [
                    can_do_add_shape * 1.0,
                    can_do_add_joint * 1.0,
                    0.0,
                    0.0,
                    1.0,
                    any_thrust_left * 1.0,
                    has_any_thursters * 1.0,
                    0.1,
                    1.0,
                    1.0,
                    1.0,
                ]
            )

            all_mutations = [mypartial(i) for i in all_mutations]
            index = jax.random.choice(_rng, jnp.arange(len(all_mutations)), (), p=probs)
            level = jax.lax.switch(index, all_mutations, _rng2, level)

            return (rng, level), None

        (_, level), _ = jax.lax.scan(inner, (rng, level), None, length=n)
        return level

    return mutate_level


def make_reset_fn_sample_kinetix_level(
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    ued_params: UEDParams = None,
    physics_engine: PhysicsEngine = None,
):

    ued_params = ued_params or UEDParams()
    physics_engine = physics_engine or PhysicsEngine(static_env_params)

    def reset(rng):
        sampled_level = sample_kinetix_level(rng, physics_engine, env_params, static_env_params, ued_params)

        return sampled_level

    return reset


def make_reset_fn_sample_empty_level(env_params: EnvParams, static_env_params: StaticEnvParams):
    def reset(rng):
        return create_empty_env(static_env_params)

    return reset


def make_vmapped_filtered_level_sampler(
    level_sampler, env_params: EnvParams, static_env_params: StaticEnvParams, config, env, ued_params: UEDParams = None
):
    ued_params = ued_params or UEDParams()

    @partial(jax.jit, static_argnums=(1,))
    def reset(rng, n_samples):
        inner = create_vmapped_filtered_distribution(
            rng,
            level_sampler,
            env_params,
            static_env_params,
            ued_params,
            n_samples,
            env,
            config["filter_levels"],
            config["level_filter_sample_ratio"],
            config["env_size_name"],
            config["level_filter_n_steps"],
        )
        return inner

    return reset


def make_reset_fn_list_of_levels(levels, static_env_params):
    assert len(levels) > 0, "Need to provide at least one level to train on"
    levels_to_reset_to, _ = load_evaluation_levels(levels, static_env_params_override=static_env_params)

    def reset(rng):
        rng, _rng = jax.random.split(rng)
        level_idx = jax.random.randint(_rng, (), 0, len(levels))
        sampled_level = jax.tree.map(lambda x: x[level_idx], levels_to_reset_to)

        return sampled_level

    return reset


ALL_MUTATION_FNS = [
    mutate_add_shape,
    mutate_add_connected_shape,
    mutate_remove_joint,
    mutate_swap_role,
    mutate_toggle_fixture,
    mutate_add_thruster,
    mutate_remove_thruster,
    mutate_remove_shape,
    mutate_change_gravity,
]


def make_reset_fn_from_config(
    config,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    physics_engine: PhysicsEngine = None,
    ued_params: UEDParams = None,
):
    if config["train_level_mode"] == "list":
        reset_fn = make_reset_fn_list_of_levels(config["train_levels_list"], static_env_params)
    elif config["train_level_mode"] == "random":
        reset_fn = make_reset_fn_sample_kinetix_level(env_params, static_env_params, ued_params, physics_engine)
    elif config["train_level_mode"] == "dummy":
        reset_fn = make_reset_fn_sample_empty_level(env_params, static_env_params)
    else:
        raise ValueError("Invalid Reset Function Provided")

    return reset_fn


def test_ued():

    env_params = EnvParams()
    static_env_params = StaticEnvParams()
    ued_params = UEDParams()
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    state = create_empty_env(env_params, static_env_params)
    state = mutate_add_shape(_rng, state, env_params, static_env_params, ued_params)
    state = mutate_add_connected_shape(_rng, state, env_params, static_env_params, ued_params)
    state = mutate_remove_shape(_rng, state, env_params, static_env_params, ued_params)
    state = mutate_remove_joint(_rng, state, env_params, static_env_params, ued_params)
    state = mutate_swap_role(_rng, state, env_params, static_env_params, ued_params)
    mutate_toggle_fixture(_rng, state, env_params, static_env_params, ued_params)

    print("Successfully did this")


if __name__ == "__main__":
    test_ued()
