"""
Based on PureJaxRL Implementation of PPO
"""

import os
import sys
import time
import typing
from functools import partial
from typing import NamedTuple

import chex
import hydra
import jax
import jax.experimental
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.training.train_state import TrainState
from kinetix.environment.ued.ued import make_reset_train_function_with_mutations, make_vmapped_filtered_level_sampler
from kinetix.environment.ued.ued import (
    make_reset_train_function_with_list_of_levels,
    make_reset_train_function_with_mutations,
)
from kinetix.util.config import (
    generate_ued_params_from_config,
    init_wandb,
    normalise_config,
    generate_params_from_config,
    get_eval_level_groups,
)
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from omegaconf import OmegaConf
from PIL import Image
from flax.serialization import to_state_dict

import wandb
from kinetix.environment.env import make_kinetix_env_from_name
from kinetix.environment.wrappers import (
    AutoReplayWrapper,
    DenseRewardWrapper,
    LogWrapper,
    UnderspecifiedToGymnaxWrapper,
)
from kinetix.models import make_network_from_config
from kinetix.models.actor_critic import ScannedRNN
from kinetix.render.renderer_pixels import make_render_pixels
from kinetix.util.learning import general_eval, get_eval_levels
from kinetix.util.saving import (
    load_train_state_from_wandb_artifact_path,
    save_model_to_wandb,
)

sys.path.append("ued")
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import load_file, save_file


def save_params(params: typing.Dict, filename: typing.Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)


def load_params(filename: typing.Union[str, os.PathLike]) -> typing.Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class RolloutBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    targets: jnp.ndarray
    advantages: jnp.ndarray
    # carry: jnp.ndarray
    mask: jnp.ndarray


def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
    keep_states=True,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """This runs the RNN on the environment, given an initial state and observation, and returns (states, rewards, episode_lengths)

    Args:
        rng (chex.PRNGKey):
        env (UnderspecifiedEnv):
        env_params (EnvParams):
        train_state (TrainState):
        init_hstate (chex.ArrayTree): Shape (num_levels, )
        init_obs (Observation): Shape (num_levels, )
        init_env_state (EnvState): Shape (num_levels, )
        max_episode_length (int):

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: (States, rewards, episode lengths) ((NUM_STEPS, NUM_LEVELS), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree.map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, hstate, x)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(rng_step, num_levels), state, action, env_params
        )

        next_mask = mask & ~done
        episode_length += mask

        if keep_states:
            return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward, info)
        else:
            return (rng, hstate, obs, next_state, done, next_mask, episode_length), (None, reward, info)

    (_, _, _, _, _, _, episode_lengths), (states, rewards, infos) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths, infos


@hydra.main(version_base=None, config_path="../configs", config_name="sfl")
def main(config):
    time_start = time.time()
    config = OmegaConf.to_container(config)
    config = normalise_config(config, "SFL" if config["ued"]["sampled_envs_ratio"] > 0 else "SFL-DR")
    env_params, static_env_params = generate_params_from_config(config)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)
    run = init_wandb(config, "SFL")

    rng = jax.random.PRNGKey(config["seed"])

    config["num_envs_from_sampled"] = int(config["num_train_envs"] * config["sampled_envs_ratio"])
    config["num_envs_to_generate"] = int(config["num_train_envs"] * (1 - config["sampled_envs_ratio"]))
    assert (config["num_envs_from_sampled"] + config["num_envs_to_generate"]) == config["num_train_envs"]

    def make_env(static_env_params):
        env = make_kinetix_env_from_name(config["env_name"], static_env_params=static_env_params)
        env = AutoReplayWrapper(env)
        env = UnderspecifiedToGymnaxWrapper(env)
        env = DenseRewardWrapper(env, dense_reward_scale=config["dense_reward_scale"])
        env = LogWrapper(env)
        return env

    env = make_env(static_env_params)

    if config["train_level_mode"] == "list":
        sample_random_level = make_reset_train_function_with_list_of_levels(
            config, config["train_levels"], static_env_params, make_pcg_state=False, is_loading_train_levels=True
        )
    elif config["train_level_mode"] == "random":
        sample_random_level = make_reset_train_function_with_mutations(
            env.physics_engine, env_params, static_env_params, config, make_pcg_state=False
        )
    else:
        raise ValueError(f"Unknown train_level_mode: {config['train_level_mode']}")

    sample_random_levels = make_vmapped_filtered_level_sampler(
        sample_random_level, env_params, static_env_params, config, make_pcg_state=False, env=env
    )
    _, eval_static_env_params = generate_params_from_config(
        config["eval_env_size_true"] | {"frame_skip": config["frame_skip"]}
    )
    eval_env = make_env(eval_static_env_params)
    ued_params = generate_ued_params_from_config(config)

    def make_render_fn(static_env_params):
        render_fn_inner = make_render_pixels(env_params, static_env_params)
        render_fn = lambda x: render_fn_inner(x).transpose(1, 0, 2)[::-1]
        return render_fn

    render_fn = make_render_fn(static_env_params)
    render_fn_eval = make_render_fn(eval_static_env_params)

    NUM_EVAL_DR_LEVELS = 200
    key_to_sample_dr_eval_set = jax.random.PRNGKey(100)
    DR_EVAL_LEVELS = sample_random_levels(key_to_sample_dr_eval_set, NUM_EVAL_DR_LEVELS)

    print("Hello here num steps is ", config["num_steps"])
    print("CONFIG is ", config)

    config["total_timesteps"] = config["num_updates"] * config["num_steps"] * config["num_train_envs"]
    config["minibatch_size"] = config["num_train_envs"] * config["num_steps"] // config["num_minibatches"]
    config["clip_eps"] = config["clip_eps"]

    config["env_name"] = config["env_name"]
    network = make_network_from_config(env, env_params, config)

    def linear_schedule(count):
        count = count // (config["num_minibatches"] * config["update_epochs"])
        frac = 1.0 - count / config["num_updates"]
        return config["lr"] * frac

    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    train_envs = 32  # To not run out of memory, the initial sample size does not matter.
    obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
    obs = jax.tree.map(
        lambda x: jnp.repeat(jnp.repeat(x[None, ...], train_envs, axis=0)[None, ...], 256, axis=0),
        obs,
    )
    init_x = (obs, jnp.zeros((256, train_envs)))
    init_hstate = ScannedRNN.initialize_carry(train_envs)
    network_params = network.init(_rng, init_hstate, init_x)
    if config["anneal_lr"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(config["lr"], eps=1e-5),
        )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    if config["load_from_checkpoint"] != None:
        print("LOADING from", config["load_from_checkpoint"], "with only params =", config["load_only_params"])
        train_state = load_train_state_from_wandb_artifact_path(
            train_state,
            config["load_from_checkpoint"],
            load_only_params=config["load_only_params"],
            legacy=config["load_legacy_checkpoint"],
        )

    rng, _rng = jax.random.split(rng)

    # INIT ENV
    rng, _rng, _rng2 = jax.random.split(rng, 3)
    rng_reset = jax.random.split(_rng, config["num_train_envs"])

    new_levels = sample_random_levels(_rng2, config["num_train_envs"])
    obsv, env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(rng_reset, new_levels, env_params)

    start_state = env_state
    init_hstate = ScannedRNN.initialize_carry(config["num_train_envs"])

    @jax.jit
    def log_buffer_learnability(rng, train_state, instances):
        BATCH_SIZE = config["num_to_save"]
        BATCH_ACTORS = BATCH_SIZE

        def _batch_step(unused, rng):
            def _env_step(runner_state, unused):
                env_state, start_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = last_obs
                ac_in = (
                    jax.tree.map(lambda x: x[np.newaxis, :], obs_batch),
                    last_done[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng).squeeze()
                log_prob = pi.log_prob(action)
                env_act = action

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["num_to_save"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, env_act, env_params
                )
                done_batch = done

                transition = Transition(
                    done,
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    reward,
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (env_state, start_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            @partial(jax.vmap, in_axes=(None, 1, 1, 1))
            @partial(jax.jit, static_argnums=(0,))
            def _calc_outcomes_by_agent(max_steps: int, dones, returns, info):
                idxs = jnp.arange(max_steps)

                @partial(jax.vmap, in_axes=(0, 0))
                def __ep_outcomes(start_idx, end_idx):
                    mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
                    r = jnp.sum(returns * mask)
                    goal_r = info["GoalR"]  # (returns > 0) * 1.0
                    success = jnp.sum(goal_r * mask)
                    l = end_idx - start_idx
                    return r, success, l

                done_idxs = jnp.argwhere(dones, size=50, fill_value=max_steps).squeeze()
                mask_done = jnp.where(done_idxs == max_steps, 0, 1)
                ep_return, success, length = __ep_outcomes(
                    jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs
                )

                return {
                    "ep_return": ep_return.mean(where=mask_done),
                    "num_episodes": mask_done.sum(),
                    "success_rate": success.mean(where=mask_done),
                    "ep_len": length.mean(where=mask_done),
                }

            # sample envs
            rng, _rng, _rng2 = jax.random.split(rng, 3)
            rng_reset = jax.random.split(_rng, config["num_to_save"])
            rng_levels = jax.random.split(_rng2, config["num_to_save"])
            # obsv, env_state = jax.vmap(sample_random_level, in_axes=(0,))(reset_rng)
            # new_levels = jax.vmap(sample_random_level)(rng_levels)
            obsv, env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(rng_reset, instances, env_params)
            # env_instances = new_levels
            init_hstate = ScannedRNN.initialize_carry(
                BATCH_ACTORS,
            )

            runner_state = (env_state, env_state, obsv, jnp.zeros((BATCH_ACTORS), dtype=bool), init_hstate, rng)
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["rollout_steps"])
            done_by_env = traj_batch.done.reshape((-1, config["num_to_save"]))
            reward_by_env = traj_batch.reward.reshape((-1, config["num_to_save"]))
            # info_by_actor = jax.tree.map(lambda x: x.swapaxes(2, 1).reshape((-1, BATCH_ACTORS)), traj_batch.info)
            o = _calc_outcomes_by_agent(config["rollout_steps"], traj_batch.done, traj_batch.reward, traj_batch.info)
            success_by_env = o["success_rate"].reshape((1, config["num_to_save"]))
            learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=0)
            return None, (learnability_by_env, success_by_env.sum(axis=0))

        rngs = jax.random.split(rng, 1)
        _, (learnability, success_by_env) = jax.lax.scan(_batch_step, None, rngs, 1)
        return learnability[0], success_by_env[0]

    num_eval_levels = len(config["eval_levels"])
    all_eval_levels = get_eval_levels(config["eval_levels"], eval_env.static_env_params)

    eval_group_indices = get_eval_level_groups(config["eval_levels"])
    print("group indices", eval_group_indices)

    @jax.jit
    def get_learnability_set(rng, network_params):

        BATCH_ACTORS = config["batch_size"]

        def _batch_step(unused, rng):
            def _env_step(runner_state, unused):
                env_state, start_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = last_obs
                ac_in = (
                    jax.tree.map(lambda x: x[np.newaxis, :], obs_batch),
                    last_done[np.newaxis, :],
                )
                hstate, pi, value = network.apply(network_params, hstate, ac_in)
                action = pi.sample(seed=_rng).squeeze()
                log_prob = pi.log_prob(action)
                env_act = action

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["batch_size"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, env_act, env_params
                )
                done_batch = done

                transition = Transition(
                    done,
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    reward,
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (env_state, start_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            @partial(jax.vmap, in_axes=(None, 1, 1, 1))
            @partial(jax.jit, static_argnums=(0,))
            def _calc_outcomes_by_agent(max_steps: int, dones, returns, info):
                idxs = jnp.arange(max_steps)

                @partial(jax.vmap, in_axes=(0, 0))
                def __ep_outcomes(start_idx, end_idx):
                    mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
                    r = jnp.sum(returns * mask)
                    goal_r = info["GoalR"]  # (returns > 0) * 1.0
                    success = jnp.sum(goal_r * mask)
                    l = end_idx - start_idx
                    return r, success, l

                done_idxs = jnp.argwhere(dones, size=50, fill_value=max_steps).squeeze()
                mask_done = jnp.where(done_idxs == max_steps, 0, 1)
                ep_return, success, length = __ep_outcomes(
                    jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs
                )

                return {
                    "ep_return": ep_return.mean(where=mask_done),
                    "num_episodes": mask_done.sum(),
                    "success_rate": success.mean(where=mask_done),
                    "ep_len": length.mean(where=mask_done),
                }

            # sample envs
            rng, _rng, _rng2 = jax.random.split(rng, 3)
            rng_reset = jax.random.split(_rng, config["batch_size"])
            new_levels = sample_random_levels(_rng2, config["batch_size"])
            obsv, env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(rng_reset, new_levels, env_params)
            env_instances = new_levels
            init_hstate = ScannedRNN.initialize_carry(
                BATCH_ACTORS,
            )

            runner_state = (env_state, env_state, obsv, jnp.zeros((BATCH_ACTORS), dtype=bool), init_hstate, rng)
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["rollout_steps"])
            done_by_env = traj_batch.done.reshape((-1, config["batch_size"]))
            reward_by_env = traj_batch.reward.reshape((-1, config["batch_size"]))
            # info_by_actor = jax.tree.map(lambda x: x.swapaxes(2, 1).reshape((-1, BATCH_ACTORS)), traj_batch.info)
            o = _calc_outcomes_by_agent(config["rollout_steps"], traj_batch.done, traj_batch.reward, traj_batch.info)
            success_by_env = o["success_rate"].reshape((1, config["batch_size"]))
            learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=0)
            return None, (learnability_by_env, success_by_env.sum(axis=0), env_instances)

        if config["sampled_envs_ratio"] == 0.0:
            print("Not doing any rollouts because sampled_envs_ratio is 0.0")
            # Here we have zero envs, so we can literally just sample random ones because there is no point.
            top_instances = sample_random_levels(_rng, config["num_to_save"])
            top_success = top_learn = learnability = success_rates = jnp.zeros(config["num_to_save"])
        else:
            rngs = jax.random.split(rng, config["num_batches"])
            _, (learnability, success_rates, env_instances) = jax.lax.scan(
                _batch_step, None, rngs, config["num_batches"]
            )

            flat_env_instances = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), env_instances)
            learnability = learnability.flatten() + success_rates.flatten() * 0.001
            top_1000 = jnp.argsort(learnability)[-config["num_to_save"] :]

            top_1000_instances = jax.tree.map(lambda x: x.at[top_1000].get(), flat_env_instances)
            top_learn, top_instances = learnability.at[top_1000].get(), top_1000_instances
            top_success = success_rates.at[top_1000].get()

        if config["put_eval_levels_in_buffer"]:
            top_instances = jax.tree.map(
                lambda all, new: jnp.concatenate([all[:-num_eval_levels], new], axis=0),
                top_instances,
                all_eval_levels.env_state,
            )

        log = {
            "learnability/learnability_sampled_mean": learnability.mean(),
            "learnability/learnability_sampled_median": jnp.median(learnability),
            "learnability/learnability_sampled_min": learnability.min(),
            "learnability/learnability_sampled_max": learnability.max(),
            "learnability/learnability_selected_mean": top_learn.mean(),
            "learnability/learnability_selected_median": jnp.median(top_learn),
            "learnability/learnability_selected_min": top_learn.min(),
            "learnability/learnability_selected_max": top_learn.max(),
            "learnability/solve_rate_sampled_mean": top_success.mean(),
            "learnability/solve_rate_sampled_median": jnp.median(top_success),
            "learnability/solve_rate_sampled_min": top_success.min(),
            "learnability/solve_rate_sampled_max": top_success.max(),
            "learnability/solve_rate_selected_mean": success_rates.mean(),
            "learnability/solve_rate_selected_median": jnp.median(success_rates),
            "learnability/solve_rate_selected_min": success_rates.min(),
            "learnability/solve_rate_selected_max": success_rates.max(),
        }

        return top_learn, top_instances, log

    def eval(rng: chex.PRNGKey, train_state: TrainState, keep_states=True):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["eval_levels"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        num_levels = len(config["eval_levels"])
        # eval_levels = get_eval_levels(config["eval_levels"], eval_env.static_env_params)
        return general_eval(
            rng,
            eval_env,
            env_params,
            train_state,
            all_eval_levels,
            env_params.max_timesteps,
            num_levels,
            keep_states=keep_states,
            return_trajectories=True,
        )

    def eval_on_dr_levels(rng: chex.PRNGKey, train_state: TrainState, keep_states=False):
        return general_eval(
            rng,
            env,
            env_params,
            train_state,
            DR_EVAL_LEVELS,
            env_params.max_timesteps,
            NUM_EVAL_DR_LEVELS,
            keep_states=keep_states,
        )

    def eval_on_top_learnable_levels(rng: chex.PRNGKey, train_state: TrainState, levels, keep_states=True):
        N = 5
        return general_eval(
            rng,
            env,
            env_params,
            train_state,
            jax.tree.map(lambda x: x[:N], levels),
            env_params.max_timesteps,
            N,
            keep_states=keep_states,
        )

    # TRAIN LOOP
    def train_step(runner_state_instances, unused):
        # COLLECT TRAJECTORIES
        runner_state, instances = runner_state_instances
        num_env_instances = instances.polygon.position.shape[0]

        def _env_step(runner_state, unused):
            train_state, env_state, start_state, last_obs, last_done, hstate, update_steps, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_batch = last_obs
            ac_in = (
                jax.tree.map(lambda x: x[np.newaxis, :], obs_batch),
                last_done[np.newaxis, :],
            )
            hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
            action = pi.sample(seed=_rng).squeeze()
            log_prob = pi.log_prob(action)
            env_act = action

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["num_train_envs"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                rng_step, env_state, env_act, env_params
            )
            done_batch = done
            transition = Transition(
                done,
                last_done,
                action.squeeze(),
                value.squeeze(),
                reward,
                log_prob.squeeze(),
                obs_batch,
                info,
            )
            runner_state = (train_state, env_state, start_state, obsv, done_batch, hstate, update_steps, rng)
            return runner_state, (transition)

        initial_hstate = runner_state[-3]
        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["num_steps"])

        # CALCULATE ADVANTAGE
        train_state, env_state, start_state, last_obs, last_done, hstate, update_steps, rng = runner_state
        last_obs_batch = last_obs  # batchify(last_obs, env.agents, config["num_train_envs"])
        ac_in = (
            jax.tree.map(lambda x: x[np.newaxis, :], last_obs_batch),
            last_done[np.newaxis, :],
        )
        _, _, last_val = network.apply(train_state.params, hstate, ac_in)
        last_val = last_val.squeeze()

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition: Transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.global_done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["gamma"] * next_value * (1 - done) - value
                gae = delta + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn_masked(params, init_hstate, traj_batch, gae, targets):

                    # RERUN NETWORK
                    _, pi, value = network.apply(
                        params,
                        jax.tree.map(lambda x: x.transpose(), init_hstate),
                        (traj_batch.obs, traj_batch.done),
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config["clip_eps"], config["clip_eps"]
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped)
                    critic_loss = config["vf_coef"] * value_loss.mean()

                    # CALCULATE ACTOR LOSS
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(logratio)
                    # if env.do_sep_reward: gae = gae.sum(axis=-1)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["clip_eps"],
                            1.0 + config["clip_eps"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    approx_kl = jax.lax.stop_gradient(((ratio - 1) - logratio).mean())
                    clipfrac = jax.lax.stop_gradient((jnp.abs(ratio - 1) > config["clip_eps"]).mean())

                    total_loss = loss_actor + critic_loss - config["ent_coef"] * entropy
                    return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clipfrac)

                grad_fn = jax.value_and_grad(_loss_fn_masked, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state
            rng, _rng = jax.random.split(rng)

            init_hstate = jax.tree.map(lambda x: jnp.reshape(x, (256, config["num_train_envs"])), init_hstate)
            batch = (
                init_hstate,
                traj_batch,
                advantages.squeeze(),
                targets.squeeze(),
            )
            permutation = jax.random.permutation(_rng, config["num_train_envs"])

            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )

            train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
            # total_loss = jax.tree.map(lambda x: x.mean(), total_loss)
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, total_loss

        # init_hstate = initial_hstate[None, :].squeeze().transpose()
        init_hstate = jax.tree.map(lambda x: x[None, :].squeeze().transpose(), initial_hstate)
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["update_epochs"])
        train_state = update_state[0]
        metric = traj_batch.info
        metric = jax.tree.map(
            lambda x: x.reshape((config["num_steps"], config["num_train_envs"])),  # , env.num_agents
            traj_batch.info,
        )
        rng = update_state[-1]

        def callback(metric):
            dones = metric["dones"]
            wandb.log(
                {
                    "episode_return": (metric["returned_episode_returns"] * dones).sum() / jnp.maximum(1, dones.sum()),
                    "episode_solved": (metric["returned_episode_solved"] * dones).sum() / jnp.maximum(1, dones.sum()),
                    "episode_length": (metric["returned_episode_lengths"] * dones).sum() / jnp.maximum(1, dones.sum()),
                    "timing/num_env_steps": int(
                        int(metric["update_steps"]) * int(config["num_train_envs"]) * int(config["num_steps"])
                    ),
                    "timing/num_updates": metric["update_steps"],
                    **metric["loss_info"],
                }
            )

        loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
        metric["loss_info"] = {
            "loss/total_loss": loss_info[0],
            "loss/value_loss": loss_info[1][0],
            "loss/policy_loss": loss_info[1][1],
            "loss/entropy_loss": loss_info[1][2],
        }
        metric["dones"] = traj_batch.done
        metric["update_steps"] = update_steps
        jax.experimental.io_callback(callback, None, metric)

        # SAMPLE NEW ENVS
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        rng_reset = jax.random.split(_rng, config["num_envs_to_generate"])

        new_levels = sample_random_levels(_rng2, config["num_envs_to_generate"])
        obsv_gen, env_state_gen = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(rng_reset, new_levels, env_params)

        rng, _rng, _rng2 = jax.random.split(rng, 3)
        sampled_env_instances_idxs = jax.random.randint(_rng, (config["num_envs_from_sampled"],), 0, num_env_instances)
        sampled_env_instances = jax.tree.map(lambda x: x.at[sampled_env_instances_idxs].get(), instances)
        myrng = jax.random.split(_rng2, config["num_envs_from_sampled"])
        obsv_sampled, env_state_sampled = jax.vmap(env.reset_to_level, in_axes=(0, 0))(myrng, sampled_env_instances)

        obsv = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), obsv_gen, obsv_sampled)
        env_state = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), env_state_gen, env_state_sampled)

        start_state = env_state
        hstate = ScannedRNN.initialize_carry(config["num_train_envs"])

        update_steps = update_steps + 1
        runner_state = (
            train_state,
            env_state,
            start_state,
            obsv,
            jnp.zeros((config["num_train_envs"]), dtype=bool),
            hstate,
            update_steps,
            rng,
        )
        return (runner_state, instances), metric

    def log_buffer(learnability, levels, epoch):
        num_samples = levels.polygon.position.shape[0]
        states = levels
        rows = 2
        fig, axes = plt.subplots(rows, int(num_samples / rows), figsize=(20, 10))
        axes = axes.flatten()
        all_imgs = jax.vmap(render_fn)(states)
        for i, ax in enumerate(axes):
            # ax.imshow(train_state.plr_buffer.get_sample(i))
            score = learnability[i]
            ax.imshow(all_imgs[i] / 255.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"learnability: {score:.3f}")
            ax.set_aspect("equal", "box")

        plt.tight_layout()
        fig.canvas.draw()
        im = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()
        return {"maps": wandb.Image(im)}

    @jax.jit
    def train_and_eval_step(runner_state, eval_rng):

        learnability_rng, eval_singleton_rng, eval_sampled_rng, _rng = jax.random.split(eval_rng, 4)
        # TRAIN
        learnabilty_scores, instances, test_metrics = get_learnability_set(learnability_rng, runner_state[0].params)

        if config["log_learnability_before_after"]:
            learn_scores_before, success_score_before = log_buffer_learnability(
                learnability_rng, runner_state[0], instances
            )

        print("instance size", sum(x.size for x in jax.tree_util.tree_leaves(instances)))

        runner_state_instances = (runner_state, instances)
        runner_state_instances, metrics = jax.lax.scan(train_step, runner_state_instances, None, config["eval_freq"])

        if config["log_learnability_before_after"]:
            learn_scores_after, success_score_after = log_buffer_learnability(
                learnability_rng, runner_state_instances[0][0], instances
            )

        # EVAL
        rng, rng_eval = jax.random.split(eval_singleton_rng)
        (states, cum_rewards, _, episode_lengths, eval_infos), (eval_dones, eval_rewards) = jax.vmap(eval, (0, None))(
            jax.random.split(rng_eval, config["eval_num_attempts"]), runner_state_instances[0][0]
        )
        all_eval_eplens = episode_lengths

        # Collect Metrics
        eval_returns = cum_rewards.mean(axis=0)  # (num_eval_levels,)
        eval_solves = (eval_infos["returned_episode_solved"] * eval_dones).sum(axis=1) / jnp.maximum(
            1, eval_dones.sum(axis=1)
        )
        eval_solves = eval_solves.mean(axis=0)
        # just grab the first run
        states, episode_lengths = jax.tree_util.tree_map(
            lambda x: x[0], (states, episode_lengths)
        )  # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        # And one attempt
        states = jax.tree_util.tree_map(lambda x: x[:, :], states)
        episode_lengths = episode_lengths[:]
        images = jax.vmap(jax.vmap(render_fn_eval))(
            states.env_state.env_state.env_state
        )  # (num_steps, num_eval_levels, ...)
        frames = images.transpose(
            0, 1, 4, 2, 3
        )  # WandB expects color channel before image dimensions when dealing with animations for some reason

        test_metrics["update_count"] = runner_state[-2]
        test_metrics["eval_returns"] = eval_returns
        test_metrics["eval_ep_lengths"] = episode_lengths
        test_metrics["eval_animation"] = (frames, episode_lengths)

        # Eval on sampled
        dr_states, dr_cum_rewards, _, dr_episode_lengths, dr_infos = jax.vmap(eval_on_dr_levels, (0, None))(
            jax.random.split(rng_eval, config["eval_num_attempts"]), runner_state_instances[0][0]
        )

        eval_dr_returns = dr_cum_rewards.mean(axis=0).mean()
        eval_dr_eplen = dr_episode_lengths.mean(axis=0).mean()

        test_metrics["eval/mean_eval_return_sampled"] = eval_dr_returns
        my_eval_dones = dr_infos["returned_episode"]
        eval_dr_solves = (dr_infos["returned_episode_solved"] * my_eval_dones).sum(axis=1) / jnp.maximum(
            1, my_eval_dones.sum(axis=1)
        )

        test_metrics["eval/mean_eval_solve_rate_sampled"] = eval_dr_solves
        test_metrics["eval/mean_eval_eplen_sampled"] = eval_dr_eplen

        # Collect Metrics
        eval_returns = cum_rewards.mean(axis=0)  # (num_eval_levels,)

        log_dict = {}

        log_dict["to_remove"] = {
            "eval_return": eval_returns,
            "eval_solve_rate": eval_solves,
            "eval_eplen": all_eval_eplens,
        }

        for i, name in enumerate(config["eval_levels"]):
            log_dict[f"eval_avg_return/{name}"] = eval_returns[i]
            log_dict[f"eval_avg_solve_rate/{name}"] = eval_solves[i]

        log_dict.update({"eval/mean_eval_return": eval_returns.mean()})
        log_dict.update({"eval/mean_eval_solve_rate": eval_solves.mean()})
        log_dict.update({"eval/mean_eval_eplen": all_eval_eplens.mean()})

        test_metrics.update(log_dict)

        runner_state, _ = runner_state_instances
        test_metrics["update_count"] = runner_state[-2]

        top_instances = jax.tree.map(lambda x: x.at[-5:].get(), instances)

        # Eval on top learnable levels
        tl_states, tl_cum_rewards, _, tl_episode_lengths, tl_infos = jax.vmap(
            eval_on_top_learnable_levels, (0, None, None)
        )(jax.random.split(rng_eval, config["eval_num_attempts"]), runner_state_instances[0][0], top_instances)

        # just grab the first run
        states, episode_lengths = jax.tree_util.tree_map(
            lambda x: x[0], (tl_states, tl_episode_lengths)
        )  # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        # And one attempt
        states = jax.tree_util.tree_map(lambda x: x[:, :], states)
        episode_lengths = episode_lengths[:]
        images = jax.vmap(jax.vmap(render_fn))(
            states.env_state.env_state.env_state
        )  # (num_steps, num_eval_levels, ...)
        frames = images.transpose(
            0, 1, 4, 2, 3
        )  # WandB expects color channel before image dimensions when dealing with animations for some reason

        test_metrics["top_learnable_animation"] = (frames, episode_lengths, tl_cum_rewards)

        if config["log_learnability_before_after"]:

            def single(x, name):
                return {
                    f"{name}_mean": x.mean(),
                    f"{name}_std": x.std(),
                    f"{name}_min": x.min(),
                    f"{name}_max": x.max(),
                    f"{name}_median": jnp.median(x),
                }

            test_metrics["learnability_log_v2/"] = {
                **single(learn_scores_before, "learnability_before"),
                **single(learn_scores_after, "learnability_after"),
                **single(success_score_before, "success_score_before"),
                **single(success_score_after, "success_score_after"),
            }

        return runner_state, (learnabilty_scores.at[-20:].get(), top_instances), test_metrics

    rng, _rng = jax.random.split(rng)
    runner_state = (
        train_state,
        env_state,
        start_state,
        obsv,
        jnp.zeros((config["num_train_envs"]), dtype=bool),
        init_hstate,
        0,
        _rng,
    )

    def log_eval(stats):
        log_dict = {}

        to_remove = stats["to_remove"]
        del stats["to_remove"]

        def _aggregate_per_size(values, name):
            to_return = {}
            for group_name, indices in eval_group_indices.items():
                to_return[f"{name}_{group_name}"] = values[indices].mean()
            return to_return

        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        env_steps_delta = config["eval_freq"] * config["num_train_envs"] * config["num_steps"]
        time_now = time.time()
        log_dict = {
            "timing/num_updates": stats["update_count"],
            "timing/num_env_steps": env_steps,
            "timing/sps": env_steps_delta / stats["time_delta"],
            "timing/sps_agg": env_steps / (time_now - time_start),
        }
        log_dict.update(_aggregate_per_size(to_remove["eval_return"], "eval_aggregate/return"))
        log_dict.update(_aggregate_per_size(to_remove["eval_solve_rate"], "eval_aggregate/solve_rate"))

        for i in range((len(config["eval_levels"]))):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[:episode_length])
            log_dict.update(
                {
                    f"media/eval_video_{config['eval_levels'][i]}": wandb.Video(
                        frames.astype(np.uint8), fps=15, caption=f"(len {episode_length})"
                    )
                }
            )

        for j in range(5):
            frames, episode_length, cum_rewards = (
                stats["top_learnable_animation"][0][:, j],
                stats["top_learnable_animation"][1][j],
                stats["top_learnable_animation"][2][:, j],
            )  # num attempts
            rr = "|".join([f"{r:<.2f}" for r in cum_rewards])
            frames = np.array(frames[:episode_length])
            log_dict.update(
                {
                    f"media/tl_animation_{j}": wandb.Video(
                        frames.astype(np.uint8), fps=15, caption=f"(len {episode_length})\n{rr}"
                    )
                }
            )

        stats.update(log_dict)
        wandb.log(stats, step=stats["update_count"])

    checkpoint_steps = config["checkpoint_save_freq"]
    assert config["num_updates"] % config["eval_freq"] == 0, "num_updates must be divisible by eval_freq"

    for eval_step in range(int(config["num_updates"] // config["eval_freq"])):
        start_time = time.time()
        rng, eval_rng = jax.random.split(rng)
        runner_state, instances, metrics = train_and_eval_step(runner_state, eval_rng)
        curr_time = time.time()
        metrics.update(log_buffer(*instances, metrics["update_count"]))
        metrics["time_delta"] = curr_time - start_time
        metrics["steps_per_section"] = (config["eval_freq"] * config["num_steps"] * config["num_train_envs"]) / metrics[
            "time_delta"
        ]
        log_eval(metrics)
        if ((eval_step + 1) * config["eval_freq"]) % checkpoint_steps == 0:
            if config["save_path"] is not None:
                steps = int(metrics["update_count"]) * int(config["num_train_envs"]) * int(config["num_steps"])
                # save_params_to_wandb(runner_state[0].params, steps, config)
                save_model_to_wandb(runner_state[0], steps, config)

    if config["save_path"] is not None:
        # save_params_to_wandb(runner_state[0].params, config["total_timesteps"], config)
        save_model_to_wandb(runner_state[0], config["total_timesteps"], config)


if __name__ == "__main__":
    # with jax.disable_jit():
    #     main()
    main()
