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
from kinetix.util.config import generate_params_from_config, generate_ued_params_from_config
from kinetix.util.learning import get_eval_levels
from kinetix.util.saving import load_world_state_pickle


class ResetMode(Enum):
    RANDOM = "random"
    LIST = "list"


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


def make_create_eval_env():
    eval_level1 = load_world_state_pickle("worlds/eval/eval_0610_car1")
    eval_level2 = load_world_state_pickle("worlds/eval/eval_0610_car2")
    eval_level3 = load_world_state_pickle("worlds/eval/eval_0628_ball_left")
    eval_level4 = load_world_state_pickle("worlds/eval/eval_0628_ball_right")
    eval_level5 = load_world_state_pickle("worlds/eval/eval_0628_hard_car_obstacle")
    eval_level6 = load_world_state_pickle("worlds/eval/eval_0628_swingup")

    def _create_eval_env(rng, env_params, static_env_params, index):
        return jax.lax.switch(
            index,
            [
                lambda: eval_level1,
                lambda: eval_level2,
                lambda: eval_level3,
                lambda: eval_level4,
                lambda: eval_level5,
                lambda: eval_level6,
            ],
        )
        return jax.tree.map(lambda x, y: jax.lax.select(index == 0, x, y), eval_level1, eval_level2)

    return _create_eval_env


def make_reset_train_function_with_mutations(
    engine: PhysicsEngine, env_params: EnvParams, static_env_params: StaticEnvParams, config
):

    ued_params = generate_ued_params_from_config(config)

    def reset(rng):
        inner = sample_kinetix_level(
            rng, engine, env_params, static_env_params, ued_params, env_size_name=config["env_size_name"]
        )

        return inner

    return reset


def make_vmapped_filtered_level_sampler(
    level_sampler, env_params: EnvParams, static_env_params: StaticEnvParams, config, env
):
    ued_params = generate_ued_params_from_config(config)

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


def make_reset_train_function_with_list_of_levels(config, levels, static_env_params, is_loading_train_levels=False):
    assert len(levels) > 0, "Need to provide at least one level to train on"
    if is_loading_train_levels:
        v = get_eval_levels(levels, static_env_params)
    else:
        _, static_env_params = generate_params_from_config(
            config["eval_env_size_true"] | {"frame_skip": config["frame_skip"]}
        )
        v = get_eval_levels(levels, static_env_params)

    def reset(rng):
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        idx = jax.random.randint(_rng, (), 0, len(levels))
        state_to_return = jax.tree.map(lambda x: x[idx], v)

        return state_to_return

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


def make_reset_func_from_config(
    config, env_params: EnvParams, static_env_params: StaticEnvParams, physics_engine: PhysicsEngine = None
):
    if config["train_level_mode"] == "list":
        reset_func = make_reset_train_function_with_list_of_levels(
            config, config["train_levels_list"], static_env_params, is_loading_train_levels=True
        )
    elif config["train_level_mode"] == "random":
        reset_func = make_reset_train_function_with_mutations(
            physics_engine or PhysicsEngine(static_env_params), env_params, static_env_params, config
        )
    else:
        raise ValueError("Invalid Reset Function Provided")

    return reset_func


def make_reset_func(
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    reset_mode: ResetMode,
    list_of_train_levels: list[str] = None,
    physics_engine: PhysicsEngine = None,
    env_size_name: str = "custom",
) -> Callable[[chex.PRNGKey], EnvState]:
    """This creates a reset function for the environment

    Args:
        env_params (EnvParams):
        static_env_params (StaticEnvParams):
        reset_mode (ResetMode): RANDOM or LIST
        list_of_train_levels (list[str], optional): If mode is LIST, this needs to be given, and controls the list of levels we can reset to. Defaults to None.
        physics_engine (PhysicsEngine, optional): If not given, we instantiate a new physics engine object. Defaults to None.
        env_size_name (str, optional): The size of the environment. Defaults to "custom".

    Returns:
        Callable[[chex.PRNGKey], EnvState]: Reset function
    """
    if reset_mode == ResetMode.LIST:
        assert list_of_train_levels is not None, "List of train levels must be provided if using list reset mode"
    return make_reset_func_from_config(
        {
            "train_level_mode": reset_mode.value,
            "train_levels_list": list_of_train_levels,
            "env_size_name": env_size_name,
        },
        env_params,
        static_env_params,
        physics_engine=physics_engine,
    )


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
    state = mutate_toggle_fixture(_rng, state, env_params, static_env_params, ued_params)

    print("Successfully did this")


if __name__ == "__main__":
    test_ued()
