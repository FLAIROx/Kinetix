"""
Based on PureJaxRL Implementation of PPO
"""

import os
import pickle
import time
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
from flax.jax_utils import replicate, unreplicate
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from PIL import Image

import wandb
from kinetix.environment import (
    LogWrapper,
    make_kinetix_env,
    make_reset_fn_from_config,
    make_vmapped_filtered_level_sampler,
)
from kinetix.models import ScannedRNN, make_network_from_config
from kinetix.render import make_render_pixels
from kinetix.util import (
    general_eval,
    generate_params_from_config,
    get_eval_level_groups,
    get_eval_levels,
    init_wandb,
    load_train_state_from_wandb_artifact_path,
    normalise_config,
    save_model,
)

os.environ["WANDB_DISABLE_SERVICE"] = "True"


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


@hydra.main(version_base=None, config_path="../configs", config_name="sfl")
def main(config):
    time_start = time.time()
    config = OmegaConf.to_container(config)
    config = normalise_config(config, "SFL" if config["ued"]["sampled_envs_ratio"] > 0 else "SFL-DR")
    env_params, static_env_params = generate_params_from_config(config)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)
    config["num_gpus"] = jax.device_count()
    print("USING", config["num_gpus"], "GPUS; #Updates = ", config["num_updates"])

    run = init_wandb(config, "SFL")
    # Do this after wandb init so we have consistent params for multi and single gpu runs
    print(
        f"CONFIG BEFORE: UPDATES = {config['num_updates']}, BATCHES = {config['num_batches']}, TRAIN_ENVS = {config['num_train_envs']}, NUM_TO_SAVE = {config['num_to_save']}"
    )
    config["num_updates"] = config["num_updates"] // config["num_gpus"]
    config["num_to_save"] = config["num_to_save"] // config["num_gpus"]
    config["num_train_envs"] = config["num_train_envs"] // config["num_gpus"]
    config["num_batches"] = config["num_batches"] // config["num_gpus"]

    print(
        f"CONFIG AFTER: UPDATES = {config['num_updates']}, BATCHES = {config['num_batches']}, TRAIN_ENVS = {config['num_train_envs']}, NUM_TO_SAVE = {config['num_to_save']}"
    )
    rng = jax.random.PRNGKey(config["seed"])

    config["num_envs_from_sampled"] = int(config["num_train_envs"] * config["sampled_envs_ratio"])
    config["num_envs_to_generate"] = int(config["num_train_envs"] * (1 - config["sampled_envs_ratio"]))
    assert (config["num_envs_from_sampled"] + config["num_envs_to_generate"]) == config["num_train_envs"]

    def make_env(static_env_params):
        env = LogWrapper(
            make_kinetix_env(
                config["action_type"],
                config["observation_type"],
                None,
                env_params,
                static_env_params,
                ignore_mask_in_obs=config.get("permutation_invariant_mlp", False),
            )
        )
        return env

    env = make_env(static_env_params)

    sample_random_level = make_reset_fn_from_config(
        config, env_params, static_env_params, physics_engine=env.physics_engine
    )
    sample_random_levels = make_vmapped_filtered_level_sampler(
        sample_random_level, env_params, static_env_params, config, env=env
    )

    _, eval_static_env_params = generate_params_from_config(
        config["eval_env_size_true"]
        | {"frame_skip": config["frame_skip"], "dense_reward_scale": config["dense_reward_scale"]}
    )
    eval_env = make_env(eval_static_env_params)

    def make_render_fn(static_env_params):
        render_fn_inner = make_render_pixels(env_params, static_env_params)
        render_fn = lambda x: render_fn_inner(x).transpose(1, 0, 2)[::-1]
        return render_fn

    render_fn = make_render_fn(static_env_params)
    render_fn_eval = make_render_fn(eval_static_env_params)

    NUM_EVAL_DR_LEVELS = 100
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
    train_envs = 32  # arbitrary
    obs, _ = env.reset(rng, env_params, sample_random_level(rng))
    obs = jax.tree.map(
        lambda x: jnp.repeat(jnp.repeat(x[None, ...], train_envs, axis=0)[None, ...], 256, axis=0),
        obs,
    )
    init_x = (obs, jnp.zeros((256, train_envs)))
    init_hstate = ScannedRNN.initialize_carry(train_envs)
    network_params = network.init(_rng, init_hstate, init_x)

    lr_to_use = linear_schedule if config["anneal_lr"] else config["lr"]
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=lr_to_use, eps=1e-5),
        ),
    )
    if config["load_from_checkpoint"] != None:
        print("LOADING from", config["load_from_checkpoint"], "with only params =", config["load_only_params"])
        train_state = load_train_state_from_wandb_artifact_path(
            train_state, config["load_from_checkpoint"], load_only_params=config["load_only_params"]
        )

    rng, _rng = jax.random.split(rng)

    # INIT ENV
    rng, _rng, _rng2 = jax.random.split(rng, 3)
    rng_reset = jax.random.split(_rng, config["num_train_envs"])

    new_levels = sample_random_levels(_rng2, config["num_train_envs"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None, 0))(rng_reset, env_params, new_levels)

    start_state = env_state
    init_hstate = ScannedRNN.initialize_carry(config["num_train_envs"])

    def make_compute_learnability_batch_step(BATCH_ACTORS, instances_to_measure=None):
        should_sample_random_environments = instances_to_measure is None

        @jax.jit
        def _batch_step(train_state_to_use, rng):
            def _env_step(runner_state, unused):
                env_state, start_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = last_obs
                ac_in = (
                    jax.tree.map(lambda x: x[np.newaxis, :], obs_batch),
                    last_done[np.newaxis, :],
                )
                hstate, pi, value = train_state_to_use.apply_fn(train_state_to_use.params, hstate, ac_in)
                action = pi.sample(seed=_rng).squeeze()
                log_prob = pi.log_prob(action)
                env_act = action

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, BATCH_ACTORS)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None, 0))(
                    rng_step, env_state, env_act, env_params, start_state
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
                    goal_r = info["GoalR"]
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
            rng_reset = jax.random.split(_rng, BATCH_ACTORS)

            if should_sample_random_environments:
                env_instances = sample_random_levels(_rng2, BATCH_ACTORS)
            else:
                env_instances = instances_to_measure

            obsv, env_state = jax.vmap(env.reset, in_axes=(0, None, 0))(rng_reset, env_params, env_instances)

            init_hstate = ScannedRNN.initialize_carry(
                BATCH_ACTORS,
            )

            runner_state = (env_state, env_state, obsv, jnp.zeros((BATCH_ACTORS), dtype=bool), init_hstate, rng)
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["rollout_steps"])
            done_by_env = traj_batch.done.reshape((-1, BATCH_ACTORS))
            reward_by_env = traj_batch.reward.reshape((-1, BATCH_ACTORS))
            # info_by_actor = jax.tree.map(lambda x: x.swapaxes(2, 1).reshape((-1, BATCH_ACTORS)), traj_batch.info)
            o = _calc_outcomes_by_agent(config["rollout_steps"], traj_batch.done, traj_batch.reward, traj_batch.info)
            success_by_env = o["success_rate"].reshape((1, BATCH_ACTORS))
            learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=0)
            return train_state_to_use, (learnability_by_env, success_by_env.sum(axis=0), env_instances)

        return _batch_step

    @jax.jit
    def log_buffer_learnability(rng, train_state, instances):
        BATCH_SIZE = config["num_to_save"] * config["num_gpus"]
        BATCH_ACTORS = BATCH_SIZE
        batch_step = make_compute_learnability_batch_step(BATCH_ACTORS, instances_to_measure=instances)

        rngs = jax.random.split(rng, 1)
        _, (learnability, success_by_env, _) = jax.lax.scan(batch_step, train_state, rngs, 1)
        return learnability[0], success_by_env[0]

    num_eval_levels = len(config["eval_levels"])
    all_eval_levels = get_eval_levels(config["eval_levels"], eval_env.static_env_params)

    eval_group_indices = get_eval_level_groups(config["eval_levels"])

    def get_temp_metrics_from_learnability(learnability, success_rates, top_learn, top_success, make_full_nan=False):
        ans = {
            "learnability/learnability_selected_mean": top_learn.mean(),
            "learnability/learnability_selected_median": jnp.median(top_learn),
            "learnability/learnability_selected_min": top_learn.min(),
            "learnability/learnability_selected_max": top_learn.max(),
            "learnability/solve_rate_selected_mean": top_success.mean(),
            "learnability/solve_rate_selected_median": jnp.median(top_success),
            "learnability/solve_rate_selected_min": top_success.min(),
            "learnability/solve_rate_selected_max": top_success.max(),
            "learnability/learnability_sampled_mean": learnability.mean(),
            "learnability/learnability_sampled_median": jnp.median(learnability),
            "learnability/learnability_sampled_min": learnability.min(),
            "learnability/learnability_sampled_max": learnability.max(),
            "learnability/solve_rate_sampled_mean": success_rates.mean(),
            "learnability/solve_rate_sampled_median": jnp.median(success_rates),
            "learnability/solve_rate_sampled_min": success_rates.min(),
            "learnability/solve_rate_sampled_max": success_rates.max(),
        }
        if make_full_nan:
            for k in ans:
                if "_sampled_" in k:
                    ans[k] = jnp.nan
        return ans

    @jax.jit
    def get_learnability_set(rng, train_state):
        BATCH_ACTORS = config["batch_size"]

        batch_step = make_compute_learnability_batch_step(BATCH_ACTORS, instances_to_measure=None)

        if config["sampled_envs_ratio"] == 0.0:
            print("Not doing any rollouts because sampled_envs_ratio is 0.0")
            # Here we have zero envs, so we can literally just sample random ones because there is no point.
            top_instances = sample_random_levels(_rng, config["num_to_save"])
            top_success = top_learn = learnability = success_rates = jnp.zeros(config["num_to_save"])
        else:
            rngs = jax.random.split(rng, config["num_batches"])
            _, (learnability, success_rates, env_instances) = jax.lax.scan(
                batch_step, train_state, rngs, config["num_batches"]
            )

            flat_env_instances = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), env_instances)
            learnability = learnability.flatten() + success_rates.flatten() * 0.001
            success_rates = success_rates.flatten()
            top_1000 = jnp.argsort(learnability)[-config["num_to_save"] :]

            top_1000_instances = jax.tree.map(lambda x: x.at[top_1000].get(), flat_env_instances)
            top_learn, top_instances = learnability.at[top_1000].get(), top_1000_instances
            top_success = success_rates.at[top_1000].get()

            jax.debug.print(
                "Learnability the main func: {}, {} -- {}",
                learnability.mean(),
                success_rates.mean(),
                (success_rates * (1 - success_rates)).mean(),
            )
            jax.debug.print(
                "Full data {}|{}::{}|{}:: {} {}",
                learnability.shape,
                success_rates.shape,
                top_learn.shape,
                top_success.shape,
                learnability,
                success_rates,
            )

        if config["put_eval_levels_in_buffer"]:
            top_instances = jax.tree.map(
                lambda all, new: jnp.concatenate([all[:-num_eval_levels], new], axis=0),
                top_instances,
                all_eval_levels.env_state,
            )

        log = get_temp_metrics_from_learnability(learnability, success_rates, top_learn, top_success)

        return top_learn, top_instances, log

    def eval(rng: chex.PRNGKey, train_state: TrainState, keep_states=True):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["eval_levels"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        num_levels = len(config["eval_levels"])
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
    @jax.jit
    def train_step(carry, unused):
        rng, runner_state, instances = carry
        rng2, rng = jax.random.split(rng)
        # COLLECT TRAJECTORIES
        runner_state = (*runner_state[:-1], rng)
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
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None, 0))(
                rng_step, env_state, env_act, env_params, start_state
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
                total_loss, grads = jax.lax.pmean((total_loss, grads), axis_name="devices")
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
                    "timing/num_env_steps": metric["update_steps"] * config["num_train_envs"] * config["num_steps"],
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
        # jax.experimental.io_callback(callback, None, metric)

        # SAMPLE NEW ENVS
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        rng_reset = jax.random.split(_rng, config["num_envs_to_generate"])

        new_levels = sample_random_levels(_rng2, config["num_envs_to_generate"])
        obsv_gen, env_state_gen = jax.vmap(env.reset, in_axes=(0, None, 0))(rng_reset, env_params, new_levels)

        rng, _rng, _rng2 = jax.random.split(rng, 3)
        sampled_env_instances_idxs = jax.random.randint(_rng, (config["num_envs_from_sampled"],), 0, num_env_instances)
        sampled_env_instances = jax.tree.map(lambda x: x.at[sampled_env_instances_idxs].get(), instances)
        myrng = jax.random.split(_rng2, config["num_envs_from_sampled"])
        obsv_sampled, env_state_sampled = jax.vmap(env.reset, in_axes=(0, None, 0))(
            myrng, env_params, sampled_env_instances
        )

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
        return (rng2, runner_state, instances), metric

    def log_buffer(learnability, levels, epoch):
        num_samples = levels.polygon.position.shape[0]
        states = levels
        rows = 2
        fig, axes = plt.subplots(rows, int(num_samples / rows), figsize=(20, 10))
        axes = axes.flatten()
        all_imgs = jax.vmap(render_fn)(states)
        for i, ax in enumerate(axes):
            score = learnability[i]
            ax.imshow(all_imgs[i] / 255.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"learnability: {score:.3f}")
            ax.set_aspect("equal", "box")

        plt.tight_layout()
        fig.canvas.draw()
        im = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        return {"maps": wandb.Image(im)}

    def get_single_learnability_dict(x, name):
        return {
            f"{name}_mean": x.mean(),
            f"{name}_std": x.std(),
            f"{name}_min": x.min(),
            f"{name}_max": x.max(),
            f"{name}_median": jnp.median(x),
        }

    pmapped_get_set = jax.pmap(get_learnability_set, axis_name="devices")

    def train_and_eval_step(runner_state_instances, eval_rng):
        time_dic = {}
        time_start = time.time()
        runner_state, instances, carry = runner_state_instances

        learnability_rng, eval_singleton_rng, eval_sampled_rng, _rng, _rng3 = jax.random.split(eval_rng, 5)

        update_step = runner_state[-2]

        train_state_replicate = replicate(runner_state[0], jax.local_devices())

        def _new_buffer(learnability_rng):
            jax.debug.print("Getting New Buffer At {}", update_step)
            rngs = jax.random.split(learnability_rng, jax.local_device_count())
            learnabilty_scores, instances, test_metrics = pmapped_get_set(rngs, train_state_replicate)
            test_metrics = jax.tree.map(lambda x: x.mean(axis=0), test_metrics)
            learnabilty_scores, instances = jax.tree.map(
                lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), (learnabilty_scores, instances)
            )

            print("LEARNABILITY SCORE SIZE", learnabilty_scores.shape)

            return learnabilty_scores, instances, test_metrics

        should_get_new_buffer = jnp.array(update_step % config["buffer_update_frequency"] == 0, dtype=bool)

        if config["eval_freq"] == config["buffer_update_frequency"]:
            # This is much faster
            learnabilty_scores, instances, test_metrics = _new_buffer(learnability_rng)
        else:
            # this is actually a lot of wasted compute but is likely better
            learnabilty_scores_new, instances_new, test_metrics_new = _new_buffer(learnability_rng)

            def _do_new(_rng):
                return learnabilty_scores_new, instances_new, test_metrics_new

            def _do_old(_rng):
                count = config["num_to_save"] * config["num_gpus"]
                metrics = get_temp_metrics_from_learnability(
                    jnp.ones(count), jnp.ones(count), jnp.ones(count), jnp.ones(count)
                )
                metrics = {k: jnp.nan for k in metrics}
                return -jnp.ones(count), instances, metrics

            learnabilty_scores, instances, test_metrics = jax.lax.cond(
                should_get_new_buffer, _do_new, _do_old, learnability_rng
            )

        print("Timing:: Getting Buffer", t := time.time() - time_start)
        time_dic["timing/get_buffer"] = t
        time_curr = time.time()
        new_carry = []
        for i, old_c in enumerate(carry):
            freq = config["log_buffer_freq"][i]
            if freq == -1:
                cond = update_step == 0
            else:
                cond = update_step % freq == 0
            jax.debug.print("Condition at step {}: {} | {}", update_step, i, cond)
            new_carry.append(jax.lax.cond(cond, (lambda: instances), (lambda: old_c)))

        if config["log_learnability_before_after"]:
            learn_scores_before, success_score_before = log_buffer_learnability(
                _rng3, runner_state[0], instances  # intentionally not the same RNG to measure bias.
            )

        if len(config["log_buffer_freq"]) > 0:
            # now log learnability for each of these
            new_dic = {}
            for i, set_of_instances in enumerate(new_carry):
                freq = config["log_buffer_freq"][i]
                if int(freq) == int(1e11):
                    freq = 0
                _rng, _rng2 = jax.random.split(_rng)
                learn, solve = log_buffer_learnability(_rng2, runner_state[0], set_of_instances)
                new_dic.update(get_single_learnability_dict(learn, f"learnability_prev_{i}_{freq}"))
                new_dic.update(get_single_learnability_dict(solve, f"success_score_prev_{i}_{freq}"))
            test_metrics["learnability_log_v3/"] = new_dic

        print("Timing:: Logging Buffer", t := time.time() - time_curr)
        time_dic["timing/logging_buffer"] = t
        time_curr = time.time()

        # TRAIN
        def single_step(runner_state_instances):
            return jax.lax.scan(train_step, runner_state_instances, None, config["eval_freq"])

        pmapped_train_step = jax.pmap(single_step, axis_name="devices")

        runner_state_instances = (
            jax.random.split(runner_state[-1], jax.local_device_count()),
            replicate(runner_state, jax.local_devices()),
            replicate(instances, jax.local_devices()),
        )
        # runner_state_instances = (, runner_)
        runner_state_instances, _ = pmapped_train_step(runner_state_instances)
        # jax.lax.scan(pmapped_train_step, runner_state_instances, None, config["eval_freq"])
        runner_state_instances = (
            unreplicate(runner_state_instances[0]),
            unreplicate(runner_state_instances[1]),
            unreplicate(runner_state_instances[2]),
        )
        runner_state_instances = runner_state_instances[1:]

        print("Timing:: Training", t := time.time() - time_curr)
        time_dic["timing/training"] = t
        time_curr = time.time()

        if config["log_learnability_before_after"]:
            learn_scores_after, success_score_after = log_buffer_learnability(
                _rng3, runner_state_instances[0][0], instances  # same seed as the previous one
            )
        print("Timing:: Log Buffer", t := time.time() - time_curr)
        time_dic["timing/log_buffer2"] = t
        time_curr = time.time()
        # EVAL
        rng, rng_eval = jax.random.split(eval_singleton_rng)
        (states, cum_rewards, _, episode_lengths, eval_infos), (eval_dones, eval_rewards) = jax.vmap(eval, (0, None))(
            jax.random.split(rng_eval, config["eval_num_attempts"]), runner_state_instances[0][0]
        )
        all_eval_eplens = episode_lengths

        # Collect Metrics
        eval_returns = cum_rewards.mean(axis=0)  # (num_eval_levels,)
        mask = jnp.arange(env_params.max_timesteps)[None, ..., None] < episode_lengths[:, None, :]
        eval_solves = (eval_infos["returned_episode_solved"] * eval_dones * mask).sum(axis=1) / jnp.maximum(
            1, (eval_dones * mask).sum(axis=1)
        )
        eval_solves = eval_solves.mean(axis=0)
        # just grab the first run
        states, episode_lengths = jax.tree_util.tree_map(
            lambda x: x[0], (states, episode_lengths)
        )  # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        # And one attempt
        states = jax.tree_util.tree_map(lambda x: x[:, :], states)
        episode_lengths = episode_lengths[:]
        images = jax.vmap(jax.vmap(render_fn_eval))(states.env_state)  # (num_steps, num_eval_levels, ...)
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

        mask_dr = jnp.arange(env_params.max_timesteps)[None, ..., None] < dr_episode_lengths[:, None, :]
        my_eval_dones = dr_infos["returned_episode"]
        eval_dr_solves = (dr_infos["returned_episode_solved"] * my_eval_dones * mask_dr).sum(axis=1) / jnp.maximum(
            1, (my_eval_dones * mask_dr).sum(axis=1)
        )

        test_metrics["eval/mean_eval_return_sampled"] = eval_dr_returns
        test_metrics["eval/mean_eval_eplen_sampled"] = eval_dr_eplen
        test_metrics["eval/mean_eval_solve_sampled"] = eval_dr_solves

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
        jax.debug.print("Shapes here eval levels {} {}", eval_returns.shape, eval_returns.mean())

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
        images = jax.vmap(jax.vmap(render_fn))(states.env_state)  # (num_steps, num_eval_levels, ...)
        frames = images.transpose(
            0, 1, 4, 2, 3
        )  # WandB expects color channel before image dimensions when dealing with animations for some reason

        test_metrics["top_learnable_animation"] = (frames, episode_lengths, tl_cum_rewards)

        if config["log_learnability_before_after"]:

            def single(x, name):
                vals = {
                    f"{name}_mean": x.mean(),
                    f"{name}_std": x.std(),
                    f"{name}_min": x.min(),
                    f"{name}_max": x.max(),
                    f"{name}_median": jnp.median(x),
                    f"{name}_all": x,
                }
                jax.debug.print("X MEAN {}||{}: {}", x.mean(), x.std(), x.shape)
                return vals

            test_metrics["learnability_log_v2/"] = {
                **single(learn_scores_before, "learnability_before"),
                **single(learn_scores_after, "learnability_after"),
                # **single(learn_scores_before_same_seed, "learnability_before_same_seed"),
                **single(success_score_before, "success_score_before"),
                **single(success_score_after, "success_score_after"),
                # **single(success_score_before_same_seed, "success_score_before_same_seed"),
            }
        print("Timing:: Eval", t := time.time() - time_curr)
        time_dic["timing/eval"] = t
        time_curr = time.time()
        print("Timing:: Iteration", t := time_curr - time_start)
        time_dic["timing/total_iteration"] = t
        test_metrics.update(time_dic)
        return (runner_state, instances, new_carry), (learnabilty_scores.at[-20:].get(), top_instances), test_metrics

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

        env_steps = (
            int(stats["update_count"])
            * int(config["num_train_envs"])
            * int(config["num_steps"])
            * int(config["num_gpus"])
        )
        env_steps_delta = int(config["eval_freq"] * config["num_train_envs"] * config["num_steps"] * config["num_gpus"])
        time_now = time.time()
        log_dict = {
            "timing/num_updates": stats["update_count"],
            "timing/num_env_steps": env_steps,
            "timing/sps": env_steps_delta / stats["time_delta"],
            "timing/sps_agg": env_steps / (time_now - time_start),
        }

        def _aggregate_per_size(values, name):
            to_return = {}
            for group_name, indices in eval_group_indices.items():
                to_return[f"{name}_{group_name}"] = values[indices].mean()
            return to_return

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
        wandb.log(stats)

    checkpoint_steps = config["checkpoint_save_freq"]
    assert config["num_updates"] % config["eval_freq"] == 0, "num_updates must be divisible by eval_freq"

    instances = sample_random_levels(rng, config["num_to_save"] * config["num_gpus"])
    carry = [
        sample_random_levels(rng, config["num_to_save"] * config["num_gpus"])
        for _ in range(len(config["log_buffer_freq"]))
    ]

    for eval_step in range(int(config["num_updates"] * config["num_gpus"] // config["eval_freq"])):
        start_time = time.time()
        rng, eval_rng = jax.random.split(rng)
        runner_state_instances, top_levels_for_logging, metrics = train_and_eval_step(
            (runner_state, instances, carry), eval_rng
        )
        runner_state, instances, carry = runner_state_instances
        curr_time = time.time()
        metrics.update(log_buffer(*top_levels_for_logging, metrics["update_count"]))
        metrics["time_delta"] = curr_time - start_time
        metrics["steps_per_section"] = (
            config["eval_freq"] * config["num_steps"] * config["num_train_envs"] * config["num_gpus"]
        ) / metrics["time_delta"]
        log_eval(metrics)
        if ((eval_step + 1) * config["eval_freq"]) % checkpoint_steps == 0:
            if config["save_path"] is not None:
                steps = (
                    int(metrics["update_count"])
                    * int(config["num_train_envs"])
                    * int(config["num_steps"])
                    * int(config["num_gpus"])
                )
                save_model(runner_state[0], steps, config)

        if config["save_learnability_buffer_pickle"]:
            steps = metrics["update_count"] * config["num_train_envs"] * config["num_steps"] * config["num_gpus"]
            run_name = config["run_name"] + "-" + str(config["random_hash"])
            filepath_to_save = f"artifacts/{run_name}/"
            os.makedirs(filepath_to_save, exist_ok=True)
            with open(f"{filepath_to_save}/learnability_buffer_{str(steps).zfill(10)}.pkl", "wb") as f:
                pickle.dump(instances, f)

    if config["save_policy"]:
        save_model(runner_state[0], config["total_timesteps"], config, is_final=True, save_to_wandb=config["use_wandb"])


if __name__ == "__main__":
    main()
