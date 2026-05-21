"""Based on PureJaxRL Implementation of PPO"""

import functools
import math
import os
import pickle
import time
from functools import partial
from typing import Any, NamedTuple

import hydra
import jax
import jax.distributed

try:
    jax.distributed.initialize(local_device_ids="")
    print("[INFO] Running across multiple nodes")
except Exception as e:
    print("[INFO] Running in single-node mode")

import jax.experimental
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.experimental import shard_map
from jax.sharding import PartitionSpec
from omegaconf import OmegaConf
from PIL import Image

from kinetix.environment import make_reset_fn_from_config, make_vmapped_filtered_level_sampler
from kinetix.models import make_network_from_config
from kinetix.models.actor_critic import GeneralActorCriticRNN
from kinetix.render import make_render_pixels
from kinetix.util import (
    EvalSpec,
    create_eval_metrics_dict_for_logging,
    generate_params_from_config,
    get_eval_level_groups,
    init_wandb,
    load_evaluation_levels,
    make_eval_fn,
    make_video_fn,
    normalise_config,
    parallel_rms_update,
    save_model,
)
from kinetix.util.config import get_video_frequency
from kinetix.util.learning import RunningMeanStandard, rms_init, rms_normalise
from kinetix.util.train_utils import get_logger, make_env, weight_norm
from kinetix.util.saving import load_train_state_from_wandb_artifact_path

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


logger = get_logger()


class RunnerState(NamedTuple):
    train_state: Any
    env_state: Any
    start_state: Any
    last_obs: Any
    last_done: Any
    extra: Any
    hstate: Any
    update_steps: int
    rng: Any


def _to_python_scalar(path, x):
    try:
        return x.item()
    except Exception:
        return x


def compute_learnability(successes, total_episodes, do_correction=False):
    success_p = successes / jnp.maximum(1, total_episodes)
    learn = success_p * (1 - success_p)
    correction = total_episodes / (total_episodes + 1)
    assert successes.shape == total_episodes.shape
    assert correction.shape == total_episodes.shape
    if do_correction:
        learn = learn * correction
    return learn


@hydra.main(version_base=None, config_path="../configs", config_name="sfl")
def main(config):

    process_id = jax.process_index()
    num_processes = jax.process_count()
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=["devices"])
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    partitioned_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices"))

    time_start = _last_time = time.time()

    def _log_time(msg):
        nonlocal _last_time
        curr = time.time()
        if process_id == 0:
            logger.info(f"STARTUP:: {msg} took {curr - _last_time:.2f}s")
        _last_time = curr

    logger.info(f"Process {process_id} of {num_processes} started %s %s", jax.devices(), jax.local_devices())
    config = normalise_config(
        OmegaConf.to_container(config), "SFL" if config["ued"]["sampled_envs_ratio"] > 0 else "SFL-DR"
    )

    if config["learnability_mode"] == "timesteps":
        config["rollout_episodes"] = 1
    env_params, static_env_params = generate_params_from_config(config)
    _log_time("generate_params_from_config")

    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)
    config["num_gpus"] = jax.device_count()
    logger.info("Running SFL on {} GPUs".format(config["num_gpus"]))

    if process_id == 0:
        init_wandb(config, "SFL")
    else:
        os.environ["WANDB_MODE"] = "disabled"

    config["num_to_save"] = config["num_to_save"] // config["num_gpus"]
    config["num_train_envs"] = config["num_train_envs"] // config["num_gpus"]
    config["num_batches"] = config["num_batches"] // config["num_gpus"]

    rng = jax.random.PRNGKey(config["seed"])

    config["num_envs_from_sampled"] = int(config["num_train_envs"] * config["sampled_envs_ratio"])
    config["num_envs_to_generate"] = int(config["num_train_envs"] * (1 - config["sampled_envs_ratio"]))
    config["total_timesteps"] = (
        config["num_updates"] * config["num_steps"] * config["num_train_envs"] * config["num_gpus"]
    )
    config["minibatch_size"] = config["num_train_envs"] * config["num_steps"] // config["num_minibatches"]

    assert (config["num_envs_from_sampled"] + config["num_envs_to_generate"]) == config["num_train_envs"]

    def _manual_replicate(value):
        value = jax.tree.map(lambda x: jnp.expand_dims(x, 0), value)
        value = jax.device_put(value, replicated_sharding)
        return value

    env = make_env(config, static_env_params, env_params)

    sample_random_level = make_reset_fn_from_config(
        config, env_params, static_env_params, physics_engine=env.physics_engine
    )

    sample_random_level = shard_map.shard_map(
        sample_random_level,
        mesh=mesh,
        in_specs=PartitionSpec(),
        out_specs=PartitionSpec(),
    )
    sample_random_levels = make_vmapped_filtered_level_sampler(
        sample_random_level, env_params, static_env_params, config, env=env
    )
    rng, _rng = jax.random.split(rng)
    _sample_levels = sample_random_levels
    if config["train_on_buffer"]:
        buffer_of_train_levels = _sample_levels(_rng, config["train_buffer_size"])

        def sample_random_levels(rng, num_samples):
            def _single_sample(rng):
                idx = jax.random.randint(rng, (), 0, config["train_buffer_size"])
                return jax.tree.map(lambda x: x[idx], buffer_of_train_levels)

            return jax.vmap(_single_sample)(jax.random.split(rng, num_samples))

    num_eval_levels = len(config["eval_levels"])
    all_eval_levels, eval_static_env_params = load_evaluation_levels(config["eval_levels"])
    eval_group_indices = get_eval_level_groups(config["eval_levels"])

    eval_static_env_params = eval_static_env_params.replace(downscale=static_env_params.downscale)
    eval_env = make_env(config, eval_static_env_params, env_params)
    _log_time("env and eval env setup")

    def make_render_fn(static_env_params):
        render_fn_inner = make_render_pixels(env_params, static_env_params)
        render_fn = lambda x: render_fn_inner(x).transpose(1, 0, 2)[::-1]
        return render_fn

    render_fn = make_render_fn(static_env_params)
    render_fn_eval = make_render_fn(eval_static_env_params)
    network = make_network_from_config(env, env_params, config)

    NUM_EVAL_DR_LEVELS = 512
    key_to_sample_dr_eval_set = jax.random.PRNGKey(100)
    DR_EVAL_LEVELS = sample_random_levels(key_to_sample_dr_eval_set, NUM_EVAL_DR_LEVELS)

    def linear_schedule(count):
        count = count // (config["num_minibatches"] * config["update_epochs"])
        frac = 1.0 - count / config["num_updates"]
        return config["lr"] * frac

    # ── Network & optimizer init ──────────────────────────────────────────────
    rng, _rng = jax.random.split(rng)
    train_envs = 32  # arbitrary
    obs, _ = env.reset(rng, env_params, sample_random_level(rng))
    obs = jax.tree.map(
        lambda x: jnp.repeat(jnp.repeat(x[None, ...], train_envs, axis=0)[None, ...], 256, axis=0),
        obs,
    )
    init_x = (obs, jnp.zeros((256, train_envs)))
    init_hstate = GeneralActorCriticRNN.initialize_carry(train_envs)
    network_params = {"params": network.init(_rng, init_hstate, init_x)["params"]}

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(network_params))
    logger.info(f"Number of parameters: {param_count/1e6:.2f}M")
    _log_time("network init")

    lr_to_use = linear_schedule if config["anneal_lr"] else config["lr"]
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=lr_to_use, eps=1e-5),
        ),
    )
    if config["load_from_checkpoint"] is not None:
        logger.info("Loading checkpoint from %s", config["load_from_checkpoint"])
        train_state = load_train_state_from_wandb_artifact_path(
            train_state,
            config["load_from_checkpoint"],
            load_only_params=config["load_only_params"],
        )

    rng, _rng = jax.random.split(rng)

    # ── Env init ──────────────────────────────────────────────────────────────
    rng, _rng, _rng2 = jax.random.split(rng, 3)
    rng_reset = jax.random.split(_rng, config["num_train_envs"])

    new_levels = sample_random_levels(_rng2, config["num_train_envs"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None, 0))(rng_reset, env_params, new_levels)

    start_state = env_state
    init_hstate = GeneralActorCriticRNN.initialize_carry(config["num_train_envs"])

    extra = {
        "learnability": (
            jnp.zeros((config["num_to_save"] * config["num_gpus"])),
            jnp.zeros((config["num_to_save"] * config["num_gpus"], 2), dtype=jnp.float32),
            jnp.zeros((config["num_envs_from_sampled"]), dtype=jnp.int32),
        ),
        "rms": rms_init(jax.tree.map(lambda x: x[0, 0], obsv)),
    }

    # ── Runner state init ─────────────────────────────────────────────────────
    rng, _rng = jax.random.split(rng)
    runner_state = RunnerState(
        train_state=train_state,
        env_state=env_state,
        start_state=start_state,
        last_obs=obsv,
        last_done=jnp.zeros((config["num_train_envs"]), dtype=bool),
        extra=extra,
        hstate=init_hstate,
        update_steps=0,
        rng=_rng,
    )
    instances = sample_random_levels(rng, config["num_to_save"] * config["num_gpus"])
    runner_state, instances = jax.experimental.multihost_utils.broadcast_one_to_all((runner_state, instances))

    # ── Observation normalisation ─────────────────────────────────────────────
    def _normalise_obs(rms: RunningMeanStandard, obs: jnp.ndarray) -> jnp.ndarray:
        if config["rms_norm"]:
            return rms_normalise(rms, obs, flatten=True)
        return obs

    # ── Learnability computation ───────────────────────────────────────────────
    def make_compute_learnability_batch_step(
        BATCH_ACTORS, learnability_mode: str, num_episodes: int, instances_to_measure=None
    ):
        should_sample_random_environments = instances_to_measure is None

        @jax.jit
        def _batch_step(carry, rng):
            train_state_to_use, extra = carry

            def _env_step(runner_state, unused):
                env_state, start_state, last_obs, last_done, hstate, rng = runner_state

                rng, _rng = jax.random.split(rng)
                obs_batch = last_obs
                obs_to_use = _normalise_obs(extra["rms"], obs_batch)

                ac_in = (
                    jax.tree.map(lambda x: x[np.newaxis, :], obs_to_use),
                    last_done[np.newaxis, :],
                )
                hstate, pi, value = train_state_to_use.apply_fn(train_state_to_use.params, hstate, ac_in)
                action = pi.sample(seed=_rng).squeeze()
                log_prob = pi.log_prob(action)
                env_act = action

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, BATCH_ACTORS)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None, 0))(
                    rng_step, env_state, env_act, env_params, start_state
                )
                done_batch = done

                transition = Transition(
                    done,
                    done,
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

                if learnability_mode == "timesteps":
                    done_idxs = jnp.argwhere(dones, size=50, fill_value=max_steps).squeeze()
                    mask_done = jnp.where(done_idxs == max_steps, 0, 1)
                    indices_to_use = jnp.concatenate([jnp.array([-1]), done_idxs[:-1]])
                else:
                    done_idxs = jnp.argwhere(dones, size=1, fill_value=max_steps).squeeze(axis=0)
                    mask_done = jnp.where(done_idxs == max_steps, 0, 1)
                    indices_to_use = jnp.array([-1])
                ep_return, success, length = __ep_outcomes(indices_to_use, done_idxs)

                return {
                    "ep_return": ep_return.mean(where=mask_done),
                    "num_episodes": mask_done.sum(),
                    "success_rate": success.mean(where=mask_done),
                    "ep_len": length.mean(where=mask_done),
                    "done_sums": dones.sum(),
                    "should_be_zero": ((info["GoalR"] * (1 - dones)).sum()),
                    "num_successes": success.sum(),
                    "total_episodes": mask_done.sum(),
                }

            rng, _rng, _rng2 = jax.random.split(rng, 3)
            rng_reset = jax.random.split(_rng, BATCH_ACTORS)

            if should_sample_random_environments:
                env_instances = sample_random_levels(_rng2, BATCH_ACTORS)
            else:
                env_instances = instances_to_measure

            def _single(rng, unused):
                rng, _rng, __rng = jax.random.split(rng, 3)
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None, 0))(
                    jax.random.split(_rng, BATCH_ACTORS), env_params, env_instances
                )
                init_hstate = GeneralActorCriticRNN.initialize_carry(BATCH_ACTORS)
                runner_state = (env_state, env_state, obsv, jnp.zeros((BATCH_ACTORS), dtype=bool), init_hstate, __rng)
                runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["rollout_steps"])
                o = _calc_outcomes_by_agent(
                    config["rollout_steps"], traj_batch.done, traj_batch.reward, traj_batch.info
                )
                success_by_env = o["success_rate"].reshape(BATCH_ACTORS)
                return rng, (
                    success_by_env,
                    o["num_successes"].reshape(BATCH_ACTORS),
                    o["total_episodes"].reshape(BATCH_ACTORS),
                )

            rng, (all_successes, total_successes, total_episodes) = jax.lax.scan(_single, rng, None, num_episodes)
            success_by_env = all_successes.mean(axis=0)
            num_episodes_by_env = total_episodes.sum(axis=0)
            total_successes_by_env = total_successes.sum(axis=0)

            learnability_by_env = compute_learnability(
                total_successes_by_env, num_episodes_by_env, config["learnability_correction"]
            )
            return (train_state_to_use, extra), (
                learnability_by_env,
                success_by_env,
                total_successes_by_env,
                num_episodes_by_env,
                env_instances,
            )

        return _batch_step

    def get_learnability_metrics(learnability, success_rates, top_learn, top_success, fill_with_nans=False):
        def _s(name, x):
            return {
                f"learnability/collection/{name}_mean": x.mean(),
                f"learnability/collection/{name}_median": jnp.median(x),
                f"learnability/collection/{name}_min": x.min(),
                f"learnability/collection/{name}_max": x.max(),
            }

        ans = (
            _s("learnability_selected", top_learn)
            | _s("solve_rate_selected", top_success)
            | _s("learnability_sampled", learnability)
            | _s("solve_rate_sampled", success_rates)
        )
        if fill_with_nans:
            ans = {k: (jnp.nan if "_sampled_" in k else v) for k, v in ans.items()}
        return ans

    def get_learnability_set(rng, train_state, extra):
        # shard_map shards the device axis into a local axis, so inputs arrive with a leading size-1 dim
        rng, train_state, extra = jax.tree.map(lambda x: x.squeeze(0), (rng, train_state, extra))

        BATCH_ACTORS = config["batch_size"]
        batch_step = make_compute_learnability_batch_step(
            BATCH_ACTORS,
            learnability_mode=config["learnability_mode"],
            num_episodes=config["rollout_episodes"],
            instances_to_measure=None,
        )

        if config["sampled_envs_ratio"] == 0.0:
            rng, _rng = jax.random.split(rng)
            top_instances = sample_random_levels(_rng, config["num_to_save"])
            top_success = top_learn = learnability = success_rates = jnp.zeros(config["num_to_save"])
            top_num_successes = top_num_episodes = jnp.zeros(config["num_to_save"], dtype=jnp.int32)
        else:
            rngs = jax.random.split(rng, config["num_batches"])
            _, (learnability, success_rates, num_successes, num_episodes, env_instances) = jax.lax.scan(
                batch_step, (train_state, extra), rngs, config["num_batches"]
            )

            flat_env_instances = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), env_instances)
            learnability = learnability.flatten() + success_rates.flatten() * 0.001
            success_rates = success_rates.flatten()
            num_successes = num_successes.flatten()
            num_episodes = num_episodes.flatten()
            top_learnability_indices = jnp.argsort(learnability)[-config["num_to_save"] :]

            top_instances = jax.tree.map(lambda x: x.at[top_learnability_indices].get(), flat_env_instances)
            top_learn = learnability.at[top_learnability_indices].get()
            top_success = success_rates.at[top_learnability_indices].get()

            top_num_successes = num_successes.at[top_learnability_indices].get()
            top_num_episodes = num_episodes.at[top_learnability_indices].get()

        if config["put_eval_levels_in_buffer"]:
            top_instances = jax.tree.map(
                lambda all, new: jnp.concatenate([all[:-num_eval_levels], new], axis=0),
                top_instances,
                all_eval_levels.env_state,
            )

        log = get_learnability_metrics(learnability, success_rates, top_learn, top_success)

        # unsqueeze outputs so shard_map can stack results across devices
        return jax.tree.map(jnp.atleast_1d, (top_learn, top_instances, log, top_num_successes, top_num_episodes))

    # ── Eval specs ────────────────────────────────────────────────────────────
    _TOP_N = 5
    hand_designed_eval_spec = EvalSpec(
        levels_to_eval_on=all_eval_levels,
        number_of_levels=len(config["eval_levels"]),
        level_names=config["eval_levels"],
        plot_videos=True,
        num_envs_to_video=-1,
    )
    sampled_eval_spec = EvalSpec(
        levels_to_eval_on=DR_EVAL_LEVELS,
        number_of_levels=NUM_EVAL_DR_LEVELS,
        level_names=[f"sampled/{str(i).zfill(3)}" for i in range(NUM_EVAL_DR_LEVELS)],
        plot_videos=False,
    )
    eval_specs_static = {"hand_designed": hand_designed_eval_spec, "sampled": sampled_eval_spec}
    if config["train_on_buffer"]:
        buffer_eval_spec = EvalSpec(
            levels_to_eval_on=buffer_of_train_levels,
            number_of_levels=config["train_buffer_size"],
            level_names=[f"buffer/{str(i).zfill(3)}" for i in range(config["train_buffer_size"])],
            plot_videos=False,
        )
        eval_specs_static["train_buffer"] = buffer_eval_spec
    top_learnable_spec_for_log = EvalSpec(
        levels_to_eval_on=None,
        number_of_levels=_TOP_N,
        level_names=[f"top_learnable/{i}" for i in range(_TOP_N)],
        plot_videos=True,
        num_envs_to_video=-1,
    )
    video_fn_eval = make_video_fn(env_params, eval_static_env_params, render_fn_eval)
    video_fn_train = make_video_fn(env_params, static_env_params, render_fn)

    def log_buffer(epoch, learnability, levels, num_success, num_episodes):
        num_samples = levels.polygon.position.shape[0]
        states = levels
        rows = int(math.sqrt(num_samples))

        fig, axes = plt.subplots(rows, int(num_samples / rows), figsize=(20, 20))
        axes = axes.flatten()
        all_imgs = jax.vmap(render_fn)(states)
        for i, ax in enumerate(axes):
            score = learnability[i]
            ax.imshow(all_imgs[i] / 255.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"L: {score:.3f} | #S: {num_success[i]} | #E: {num_episodes[i]}")
            ax.set_aspect("equal", "box")

        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        im = Image.fromarray(buf[..., :3])
        try:
            ans = {"maps": wandb.Image(im)}
        except Exception as e:
            print("Failed to load maps", e)
            ans = {}
        return ans

    last_log_time = time.perf_counter()

    # ── Train step ────────────────────────────────────────────────────────────
    def train_step(carry, unused):
        rng, runner_state, instances = carry
        rng2, rng = jax.random.split(rng)

        runner_state = runner_state._replace(rng=rng)
        num_env_instances = instances.polygon.position.shape[0]

        def _env_step(runner_state, unused):
            train_state, env_state, start_state, last_obs, last_done, extra, hstate, update_steps, rng = runner_state

            rng, _rng = jax.random.split(rng)
            obs_batch = last_obs
            obs_to_use = _normalise_obs(extra["rms"], obs_batch)
            ac_in = (
                jax.tree.map(lambda x: x[np.newaxis, :], obs_to_use),
                last_done[np.newaxis, :],
            )
            hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
            action = pi.sample(seed=_rng).squeeze()
            log_prob = pi.log_prob(action)
            env_act = action

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
            runner_state = RunnerState(
                train_state=train_state,
                env_state=env_state,
                start_state=start_state,
                last_obs=obsv,
                last_done=done_batch,
                extra=extra,
                hstate=hstate,
                update_steps=update_steps,
                rng=rng,
            )
            return runner_state, transition

        initial_hstate = runner_state.hstate
        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["num_steps"])

        # CALCULATE ADVANTAGE
        train_state, env_state, start_state, last_obs, last_done, extra, hstate, update_steps, rng = runner_state
        obs_to_use = _normalise_obs(extra["rms"], last_obs)
        _, _, last_val = network.apply(
            train_state.params,
            hstate,
            (
                jax.tree.map(lambda x: x[np.newaxis, :], obs_to_use),
                last_done[np.newaxis, :],
            ),
        )
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
                (jax.lax.pvary(jnp.zeros_like(last_val), ("devices",)), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        if config["rms_norm"]:
            extra["rms"] = parallel_rms_update(extra["rms"], traj_batch.obs)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn_masked(params, init_hstate, traj_batch, gae, targets):
                    obs_to_use = jax.vmap(_normalise_obs, (None, 0))(extra["rms"], traj_batch.obs)

                    _, pi, value = network.apply(
                        params,
                        jax.tree.map(lambda x: x.transpose(), init_hstate),
                        (obs_to_use, traj_batch.done),
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
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"]) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    approx_kl = jax.lax.stop_gradient(((ratio - 1) - logratio).mean())
                    clipfrac = jax.lax.stop_gradient((jnp.abs(ratio - 1) > config["clip_eps"]).mean())

                    total_loss = loss_actor + critic_loss - config["ent_coef"] * entropy
                    return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clipfrac, {})

                grad_fn = jax.value_and_grad(_loss_fn_masked, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                total_loss, grads = jax.lax.pmean((total_loss, grads), axis_name="devices")
                train_state = train_state.apply_gradients(grads=grads)

                total_loss[-1][-1]["model/gradient_norm"] = weight_norm(grads)
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

            shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)

            minibatches = jax.tree.map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(x, [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:])),
                    1,
                    0,
                ),
                shuffled_batch,
            )

            train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            return update_state, total_loss

        init_hstate = jax.tree.map(lambda x: x[None, :].squeeze().transpose(), initial_hstate)
        update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["update_epochs"])
        train_state, rng = update_state[0], update_state[-1]
        metric = jax.tree.map(
            lambda x: x.reshape((config["num_steps"], config["num_train_envs"])),
            traj_batch.info,
        )

        def _log_to_wandb(loss_info, metric):
            def callback(metric):
                if process_id != 0:
                    return
                if metric["device_id"] != 0:
                    return
                nonlocal last_log_time
                delta_time = time.time() - last_log_time
                last_log_time = time.time()
                dones = metric["dones"]
                env_steps_delta = int(config["num_train_envs"]) * int(config["num_steps"]) * int(config["num_gpus"])
                env_steps = int(metric["update_steps"]) * env_steps_delta
                wandb.log(
                    {
                        "episode_return": (metric["returned_episode_returns"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "episode_solved": (metric["returned_episode_solved"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "episode_length": (metric["returned_episode_lengths"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "timing/num_env_steps": env_steps,
                        "timing/num_updates": int(metric["update_steps"]),
                        "model/weight_norm": metric["model/weight_norm"],
                        "timing/callback_sps": env_steps_delta / delta_time,
                        "timing/callback_sps_per_gpu": env_steps_delta / (delta_time * int(config["num_gpus"])),
                        **metric["loss_info"],
                        **metric["model_info"],
                    }
                )

            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            model_metrics = loss_info[1][-1]

            metric["loss_info"] = {
                "loss/total_loss": loss_info[0],
                "loss/value_loss": loss_info[1][0],
                "loss/policy_loss": loss_info[1][1],
                "loss/entropy_loss": loss_info[1][2],
            }
            metric["model_info"] = model_metrics
            metric["dones"] = traj_batch.done
            metric["update_steps"] = update_steps
            metric["model/weight_norm"] = weight_norm(train_state.params)
            metric["device_id"] = jax.lax.axis_index("devices")
            jax.experimental.io_callback(callback, None, metric)

        _log_to_wandb(loss_info, metric)

        # Update learnability metrics for the sampled levels
        def _update_learnability_for_sampled_levels(extra):
            reached_goal = traj_batch.info["GoalR"].sum(axis=0)[config["num_envs_to_generate"] :]
            total_dones = traj_batch.global_done.sum(axis=0)[config["num_envs_to_generate"] :]

            learn_scores, success_total, indices_to_use = extra
            is_valid_idxs = jnp.where(indices_to_use == -1, 0.0, 1.0)
            success_total_temp = jnp.zeros_like(success_total)

            success_total_temp = success_total_temp.at[indices_to_use, 0].add(reached_goal * is_valid_idxs)
            success_total_temp = success_total_temp.at[indices_to_use, 1].add(total_dones * is_valid_idxs)

            success_total_temp = jax.lax.psum(success_total_temp, axis_name="devices")
            success_total = success_total_temp + success_total

            temp_learn_scores = compute_learnability(
                success_total[indices_to_use, 0], success_total[indices_to_use, 1], config["learnability_correction"]
            )

            learn_scores = learn_scores.at[indices_to_use].set(
                jnp.where(is_valid_idxs, temp_learn_scores, learn_scores[indices_to_use])
            )

            return (learn_scores, success_total, indices_to_use)

        extra["learnability"] = _update_learnability_for_sampled_levels(extra["learnability"])

        # SAMPLE NEW ENVS
        def _sample_new_envs(rng):
            rng, _rng, _rng2 = jax.random.split(rng, 3)
            rng_reset = jax.random.split(_rng, config["num_envs_to_generate"])

            new_levels = sample_random_levels(_rng2, config["num_envs_to_generate"])
            obsv_gen, env_state_gen = jax.vmap(env.reset, in_axes=(0, None, 0))(rng_reset, env_params, new_levels)

            rng, _rng, _rng2 = jax.random.split(rng, 3)

            sampled_env_instances_idxs = jax.random.randint(
                _rng, (config["num_envs_from_sampled"],), 0, num_env_instances
            )
            sampled_env_instances = jax.tree.map(lambda x: x.at[sampled_env_instances_idxs].get(), instances)
            myrng = jax.random.split(_rng2, config["num_envs_from_sampled"])
            obsv_sampled, env_state_sampled = jax.vmap(env.reset, in_axes=(0, None, 0))(
                myrng, env_params, sampled_env_instances
            )

            obsv = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), obsv_gen, obsv_sampled)
            env_state = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), env_state_gen, env_state_sampled)

            return env_state, env_state, obsv, sampled_env_instances_idxs

        rng, _rng = jax.random.split(rng)
        env_state, start_state, obsv, sampled_env_instances_idxs = _sample_new_envs(_rng)
        extra["learnability"] = extra["learnability"][:2] + (sampled_env_instances_idxs,)

        update_steps = update_steps + 1
        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            start_state=start_state,
            last_obs=obsv,
            last_done=jnp.zeros((config["num_train_envs"]), dtype=bool),
            extra=extra,
            hstate=GeneralActorCriticRNN.initialize_carry(config["num_train_envs"]),
            update_steps=update_steps,
            rng=rng,
        )
        return (rng2, runner_state, instances), metric

    def _eval_step(runner_state_instances, rng):
        runner_state, _instances = runner_state_instances
        update_count_in_eval = runner_state.update_steps
        train_state = runner_state.train_state
        extra = runner_state.extra
        rng, rng_eval_hd, rng_eval_env = jax.random.split(rng, 3)

        obs_pp_fn = functools.partial(_normalise_obs, extra["rms"])

        # make_eval_fn must be rebuilt each step (obs_pp_fn closes over extra["rms"])
        eval_fn_hd = make_eval_fn(
            eval_env,
            env_params,
            config["eval_num_attempts"],
            fixed_video_rng=jax.random.PRNGKey(102),
            observation_preprocessing_fn=obs_pp_fn,
        )
        eval_fn_env = make_eval_fn(
            env,
            env_params,
            config["eval_num_attempts"],
            observation_preprocessing_fn=obs_pp_fn,
        )

        hd_metrics = eval_fn_hd(rng_eval_hd, train_state, {"hand_designed": hand_designed_eval_spec})
        env_eval_specs = {"sampled": sampled_eval_spec}
        if config["train_on_buffer"]:
            env_eval_specs["train_buffer"] = buffer_eval_spec
        env_metrics = eval_fn_env(rng_eval_env, train_state, env_eval_specs)

        eval_metrics = hd_metrics | env_metrics

        vf = get_video_frequency(config, update_count_in_eval)
        should_log_videos = jnp.logical_and(update_count_in_eval % vf == 0, config["should_log_videos"])
        videos = video_fn_eval(should_log_videos, hd_metrics)

        return {
            "update_count": update_count_in_eval,
            "eval": {
                "episode_metrics": {k: v.episode_metrics for k, v in eval_metrics.items()},
                "videos": videos,
                "should_log_videos": should_log_videos,
            },
        }

    # ── shard_map wrappers ────────────────────────────────────────────────────
    def single_device_train_step(runner_state_instances):
        # shard_map shards the device axis into a local axis, so squeeze the leading 1 before scanning
        runner_state_instances = jax.tree.map(lambda x: x.squeeze(0), runner_state_instances)
        return jax.lax.scan(train_step, runner_state_instances, None, config["eval_freq"])

    in_specs = ((PartitionSpec("devices"), PartitionSpec(), PartitionSpec()),)
    shard_mapped_train_step = jax.jit(
        shard_map.shard_map(
            single_device_train_step,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=PartitionSpec(),
            check_rep=False,
        )
    )

    shard_mapped_get_learnability_set = jax.jit(
        shard_map.shard_map(
            get_learnability_set,
            mesh=mesh,
            in_specs=(PartitionSpec("devices"), PartitionSpec(), PartitionSpec()),
            out_specs=PartitionSpec("devices"),
            check_rep=False,
        )
    )

    sharded_eval_step = jax.jit(
        jax.shard_map(
            _eval_step,
            mesh=mesh,
            in_specs=PartitionSpec(),
            out_specs=PartitionSpec(),
        )
    )

    # ── Main train-and-eval loop ──────────────────────────────────────────────
    def train_and_eval_step(runner_state_instances, eval_rng):
        time_dic = {}
        time_train_start = time.time()
        runner_state, instances = runner_state_instances

        learnability_rng, eval_singleton_rng, eval_tl_rng = jax.random.split(eval_rng, 3)

        update_step = runner_state.update_steps

        train_state_replicate = _manual_replicate(runner_state.train_state)
        extra_replicate = _manual_replicate(runner_state.extra)

        def _update_buffer(instances, learnability_rng):
            def _new_buffer(learnability_rng):
                rngs = jax.device_put(jax.random.split(learnability_rng, jax.device_count()), partitioned_sharding)
                results = shard_mapped_get_learnability_set(rngs, train_state_replicate, extra_replicate)
                learnability_scores, instances, test_metrics, num_successes, num_episodes = results
                test_metrics = jax.tree.map(lambda x: x.mean(axis=0), test_metrics)
                assert learnability_scores.ndim == 1
                assert learnability_scores.shape == (
                    config["num_to_save"] * config["num_gpus"],
                ), f"{learnability_scores.shape} {config['num_to_save']} {config['num_gpus']}"
                return learnability_scores, instances, test_metrics, num_successes, num_episodes

            should_get_new_buffer = jnp.array(update_step % config["buffer_update_frequency"] == 0, dtype=bool)

            learnability_extra = runner_state.extra["learnability"]
            if config["eval_freq"] == config["buffer_update_frequency"]:
                learnability_scores, instances, test_metrics, num_successes, num_episodes = _new_buffer(
                    learnability_rng
                )
            else:
                learnability_scores_new, instances_new, test_metrics_new, num_successes, num_episodes = _new_buffer(
                    learnability_rng
                )

                def _do_new():
                    return (
                        learnability_scores_new,
                        instances_new,
                        test_metrics_new,
                        num_successes * 1.0,
                        num_episodes * 1.0,
                    )

                def _do_old():
                    count = config["num_to_save"] * config["num_gpus"]
                    metrics = get_learnability_metrics(
                        jnp.ones(count), jnp.ones(count), jnp.ones(count), jnp.ones(count)
                    )
                    metrics = {k: jnp.nan for k in metrics}
                    return (
                        learnability_extra[0],
                        instances,
                        metrics,
                        learnability_extra[1][:, 0],
                        learnability_extra[1][:, 1],
                    )

                learnability_scores, instances, test_metrics, num_successes, num_episodes = jax.lax.cond(
                    should_get_new_buffer, _do_new, _do_old
                )
            return learnability_scores, instances, test_metrics, num_successes, num_episodes

        learnability_scores, instances, test_metrics, num_successes, num_episodes = _update_buffer(
            instances, learnability_rng
        )
        extra = runner_state.extra
        extra["learnability"] = (
            learnability_scores,
            jnp.stack([num_successes, num_episodes], axis=1) * 1.0,
            jnp.zeros((config["num_envs_from_sampled"]), dtype=jnp.int32) - 1,
        )

        runner_state = runner_state._replace(extra=extra)
        time_dic["timing/get_buffer"] = t = time.time() - time_train_start
        logger.info(f"Timing:: Getting Buffer {t:.2f}s")
        time_eval_start = time.time()

        # TRAIN
        runner_state_instances = (
            jax.device_put(jax.random.split(runner_state.rng, jax.device_count()), partitioned_sharding),
            _manual_replicate(runner_state),
            _manual_replicate(instances),
        )

        runner_state_instances, _ = shard_mapped_train_step(runner_state_instances)
        runner_state_instances = runner_state_instances[1:]

        time_dic["timing/training"] = t = time.time() - time_eval_start
        logger.info(f"Timing:: Training {t:.2f}s")
        time_eval_start = time.time()

        # EVAL
        test_metrics.update(sharded_eval_step(runner_state_instances, eval_singleton_rng))
        runner_state, _ = runner_state_instances

        test_metrics["update_count"] = runner_state.update_steps

        top_instances = jax.tree.map(lambda x: x.at[-_TOP_N:].get(), instances)
        tl_spec = EvalSpec(
            levels_to_eval_on=top_instances,
            number_of_levels=_TOP_N,
            level_names=top_learnable_spec_for_log.level_names,
            plot_videos=True,
            num_envs_to_video=-1,
        )
        tl_obs_pp_fn = functools.partial(_normalise_obs, runner_state.extra["rms"])
        # make_eval_fn rebuilt here too (closes over rms)
        tl_eval_fn = make_eval_fn(
            env, env_params, config["eval_num_attempts"], observation_preprocessing_fn=tl_obs_pp_fn
        )
        tl_eval_metrics = tl_eval_fn(eval_tl_rng, runner_state.train_state, {"top_learnable": tl_spec})
        should_log_videos_tl = test_metrics["eval"]["should_log_videos"]
        tl_videos = video_fn_train(should_log_videos_tl, tl_eval_metrics)
        test_metrics["tl_eval"] = {
            "episode_metrics": {k: v.episode_metrics for k, v in tl_eval_metrics.items()},
            "videos": tl_videos,
            "should_log_videos": should_log_videos_tl,
        }
        num_maps_to_save = 25
        test_metrics.update(
            log_buffer(
                test_metrics["update_count"],
                *jax.tree.map(
                    lambda x: x.at[-num_maps_to_save:].get()[::-1],
                    (learnability_scores, instances, num_successes, num_episodes),
                ),
            )
        )

        # Track how learnability estimates change over the training interval
        new_learnability_extra = runner_state.extra["learnability"]
        old_succ, old_total = num_successes, num_episodes
        new_succ, new_total = new_learnability_extra[1][:, 0], new_learnability_extra[1][:, 1]

        learnability_scores_new = compute_learnability(new_succ, new_total, config["learnability_correction"])
        learnability_scores_old = compute_learnability(old_succ, old_total, config["learnability_correction"])
        diff = learnability_scores_new - learnability_scores_old
        test_metrics["learnability/update_over_time/old_learnability"] = learnability_scores_old.mean()
        test_metrics["learnability/update_over_time/new_learnability"] = learnability_scores_new.mean()
        test_metrics["learnability/update_over_time/diff_learnability"] = diff.mean()
        test_metrics["learnability/update_over_time/diff_episodes"] = (new_total - old_total).mean()
        test_metrics["learnability/update_over_time/new_episodes"] = new_total.mean()
        test_metrics["learnability/update_over_time/old_episodes"] = old_total.mean()

        time_dic["timing/eval"] = t = time.time() - time_eval_start
        logger.info(f"Timing:: Eval {t:.2f}s")
        time_dic["timing/total_iteration"] = time.time() - time_train_start
        test_metrics.update(time_dic)

        return (runner_state, instances), test_metrics

    def log_eval(stats):
        eval_data = stats["eval"]

        env_steps = (
            int(stats["update_count"])
            * int(config["num_train_envs"])
            * int(config["num_steps"])
            * int(config["num_gpus"])
        )
        env_steps_delta = int(config["eval_freq"] * config["num_train_envs"] * config["num_steps"] * config["num_gpus"])
        time_now = time.time()

        def _aggregate_per_size(values, name):
            return {
                f"{name}_{group_name}": values[indices].mean() for group_name, indices in eval_group_indices.items()
            }

        hd_metrics = eval_data["episode_metrics"]["hand_designed"]
        log_dict = {
            "timing/num_updates": stats["update_count"],
            "timing/num_env_steps": env_steps,
            **(
                {}
                if "time_delta" not in stats
                else {
                    "timing/sps": (sps := env_steps_delta / stats["time_delta"]),
                    "timing/sps_agg": env_steps / (time_now - time_start),
                    "timing/sps_per_gpu": sps / config["num_gpus"],
                    "timing/remaining_time_hours": (config["total_timesteps"] - env_steps) / sps / 3600,
                }
            ),
            **_aggregate_per_size(hd_metrics.episode_returns, "eval_info/aggregate_return"),
            **_aggregate_per_size(hd_metrics.episode_solve_rates, "eval_info/aggregate_solve_rate"),
            **create_eval_metrics_dict_for_logging(
                eval_specs_static,
                eval_data["episode_metrics"],
                eval_data["videos"] if eval_data["should_log_videos"] else None,
            ),
        }

        if "tl_eval" in stats:
            tl_data = stats["tl_eval"]
            log_dict.update(
                create_eval_metrics_dict_for_logging(
                    {"top_learnable": top_learnable_spec_for_log},
                    tl_data["episode_metrics"],
                    tl_data["videos"] if tl_data["should_log_videos"] else None,
                )
            )

        if process_id == 0:
            wandb.log({k: v for k, v in stats.items() if k not in ("eval", "tl_eval")} | log_dict)

    checkpoint_steps = config["checkpoint_save_freq"]
    assert (
        config["num_updates"] % config["eval_freq"] == 0
    ), f"num_updates ({config['num_updates']}) must be divisible by eval_freq ({config['eval_freq']})"

    # ── Main training loop ────────────────────────────────────────────────────
    rng, _rng = jax.random.split(rng)
    if not config["skip_initial_eval"]:
        logger.info("Starting eval at beginning of training")
        st = time.perf_counter()
        metrics = sharded_eval_step((runner_state, instances), _rng)
        log_eval(metrics | {"update_count": 0})
        logger.info("Finished initial eval in {:.2f}s".format(time.perf_counter() - st))

    for eval_step in range(int(config["num_updates"] // config["eval_freq"])):
        start_time = time.time()
        rng, eval_rng = jax.random.split(rng)
        runner_state_instances, metrics = train_and_eval_step((runner_state, instances), eval_rng)
        runner_state, instances = runner_state_instances
        curr_time = time.time()
        metrics["time_delta"] = curr_time - start_time

        metrics = jax.tree.map_with_path(_to_python_scalar, metrics)
        log_eval(metrics)
        if ((eval_step + 1) * config["eval_freq"]) % checkpoint_steps == 0:
            if config["save_path"] is not None:
                steps = (
                    int(metrics["update_count"])
                    * int(config["num_train_envs"])
                    * int(config["num_steps"])
                    * int(config["num_gpus"])
                )
                save_model(runner_state.train_state, steps, config, extra={"rms": runner_state.extra["rms"]})

        if config["save_learnability_buffer_pickle"]:
            steps = metrics["update_count"] * config["num_train_envs"] * config["num_steps"] * config["num_gpus"]
            run_name = config["run_name"] + "-" + str(config["random_hash"])
            filepath_to_save = f"artifacts/{run_name}/"
            os.makedirs(filepath_to_save, exist_ok=True)
            with open(f"{filepath_to_save}/learnability_buffer_{str(steps).zfill(10)}.pkl", "wb") as f:
                pickle.dump(instances, f)

    if config["save_policy"]:
        save_model(
            runner_state.train_state,
            config["total_timesteps"],
            config,
            is_final=True,
            save_to_wandb=config["use_wandb"],
            extra={"rms": runner_state.extra["rms"]},
        )


if __name__ == "__main__":
    main()
