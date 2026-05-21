import functools
import math
import os
import time
from typing import Any, NamedTuple

import hydra
import jax

try:
    jax.distributed.initialize(local_device_ids="")  # local_device_ids="" to let jax pick devices
    print("[INFO] Running across multiple nodes")
except Exception as e:
    print("[INFO] Running in single-node mode")

import jax.numpy as jnp
from jax.sharding import PartitionSpec

import numpy as np
import optax
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import wandb
from kinetix.data import get_valid_action_mask
from kinetix.environment import EnvParams, KinetixEnv, make_reset_fn_from_config
from kinetix.models import GeneralActorCriticRNN, make_network_from_config
from kinetix.render import make_render_pixels
from kinetix.util import (
    EpisodeMetrics,
    EvalSpec,
    RunningMeanStandard,
    create_eval_metrics_dict_for_logging,
    generate_params_from_config,
    get_video_frequency,
    init_wandb,
    load_evaluation_levels,
    load_train_state_from_wandb_artifact_path,
    make_eval_fn,
    make_fake_video,
    make_video_fn,
    normalise_config,
    parallel_rms_update,
    rms_init,
    rms_normalise,
    save_model,
)
from kinetix.util.train_utils import (
    compute_gns_metrics,
    get_logger,
    get_randomly_sampled_eval_levels,
    make_env,
    weight_norm,
)

logger = get_logger()

os.environ["WANDB_DISABLE_SERVICE"] = "True"


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Any
    info: jnp.ndarray
    valid_action_mask: jnp.ndarray


class RunnerState(NamedTuple):
    train_state: Any
    env_state: Any
    last_obs: Any
    last_done: jnp.ndarray
    extra: dict
    hstate: Any
    rng: jnp.ndarray
    update_step: jnp.ndarray


def get_train_state_from_config(config, rng: jax.Array, env: KinetixEnv, env_params: EnvParams):
    dummy_batch_dim = 1
    network = make_network_from_config(env, env_params, config)
    rng, _rng = jax.random.split(rng)
    obsv, env_state = jax.vmap(env.reset, (0, None))(jax.random.split(_rng, dummy_batch_dim), env_params)
    dones = jnp.zeros((dummy_batch_dim), dtype=jnp.bool_)
    rng, _rng = jax.random.split(rng)
    init_hstate = GeneralActorCriticRNN.initialize_carry(dummy_batch_dim)
    init_x = jax.tree.map(lambda x: x[None, ...], (obsv, dones))
    network_params = {"params": network.init(_rng, init_hstate, init_x)["params"]}

    def linear_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / config["num_updates"]
        return config["lr"] * frac

    def linear_warmup_cosine_decay_schedule(count):
        frac = (count // (config["num_minibatches"] * config["update_epochs"])) / config[
            "num_updates"
        ]  # between 0 and 1
        delta = config["peak_lr"] - config["initial_lr"]
        frac_diff_max = 1.0 - config["warmup_frac"]
        frac_cosine = (frac - config["warmup_frac"]) / frac_diff_max

        return jax.lax.select(
            frac < config["warmup_frac"],
            config["initial_lr"] + delta * frac / config["warmup_frac"],
            config["peak_lr"] * jnp.maximum(0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * ((frac_cosine) % 1.0)))),
        )

    if config["anneal_lr"]:
        lr_to_use = linear_schedule
    elif config["warmup_lr"]:
        lr_to_use = linear_warmup_cosine_decay_schedule
    else:
        lr_to_use = config["lr"]
    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(lr_to_use, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    return train_state


def make_train(config, ppo_env_params, static_env_params, return_update_step_and_init_state: bool = False):
    NUM_GPUS = jax.device_count()
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=["devices"])

    replicated_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())
    partitioned_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec("devices"))

    config["num_updates"] = config["total_timesteps"] // config["num_steps"] // config["num_train_envs"]
    config["num_gpus"] = NUM_GPUS
    config["num_train_envs"] = config["num_train_envs"] // NUM_GPUS

    reset_fn = make_reset_fn_from_config(config, ppo_env_params, static_env_params)

    eval_levels, eval_static_env_params = load_evaluation_levels(config["eval_levels"])
    ppo_env = make_env(config, static_env_params, ppo_env_params, reset_fn)
    eval_env = make_env(config, eval_static_env_params, ppo_env_params, None)

    NUM_EVAL_DR_LEVELS = 512
    DR_EVAL_LEVELS = get_randomly_sampled_eval_levels(
        config, ppo_env_params, static_env_params, ppo_env, NUM_EVAL_DR_LEVELS
    )

    all_eval_specs = {
        "hand_designed": EvalSpec(
            levels_to_eval_on=eval_levels,
            number_of_levels=len(config["eval_levels"]),
            level_names=config["eval_levels"],
            plot_videos=True,
            num_envs_to_video=-1,
        ),
        "sampled": EvalSpec(
            levels_to_eval_on=DR_EVAL_LEVELS,
            number_of_levels=NUM_EVAL_DR_LEVELS,
            level_names=[f"DR/{str(i).zfill(3)}" for i in range(NUM_EVAL_DR_LEVELS)],
            plot_videos=True,
            num_envs_to_video=10,
        ),
    }

    time_start = time.time()

    def train(rng):
        def _maybe_normalise(
            rms: RunningMeanStandard, obs: jnp.ndarray, should_update=True
        ) -> tuple[RunningMeanStandard, jnp.ndarray]:
            if config["rms_norm"]:
                new_obs = rms_normalise(rms, obs, flatten="auto")
                if should_update:
                    rms = parallel_rms_update(rms, obs)
            else:
                new_obs = obs

            return rms, new_obs

        last_time = time.time()
        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        train_state = get_train_state_from_config(config, _rng, ppo_env, ppo_env_params)

        new_extra = None
        if config["load_from_checkpoint"] is not None:
            logger.info(
                "Loading checkpoint from %s (load_only_params=%s)",
                config["load_from_checkpoint"],
                config["load_only_params"],
            )
            train_state, new_extra = load_train_state_from_wandb_artifact_path(
                train_state,
                config["load_from_checkpoint"],
                load_only_params=config["load_only_params"],
                return_extra=True,
                specific_dir="/tmp",
            )
            assert new_extra is not None
        # INIT ENV
        rng, _rng = jax.random.split(rng)

        def _init(rng):
            rng = rng.squeeze(0)
            obsv, env_state = jax.vmap(ppo_env.reset, (0, None))(
                jax.random.split(rng, config["num_train_envs"]), ppo_env_params
            )
            init_hstate = GeneralActorCriticRNN.initialize_carry(config["num_train_envs"])
            init_dones = jnp.zeros((config["num_train_envs"]), dtype=bool)

            return obsv, env_state, init_hstate, init_dones

        _init_shard_mapped = jax.jit(
            jax.shard_map(
                _init,
                mesh=mesh,
                in_specs=(PartitionSpec("devices"),),
                out_specs=PartitionSpec("devices"),
                check_vma=False,
            )
        )
        rngs_per_device = jax.device_put(jax.random.split(_rng, NUM_GPUS), partitioned_sharding)
        obsv, env_state, init_hstate, init_dones = _init_shard_mapped(rngs_per_device)

        render_static_env_params = eval_env.static_env_params.replace(downscale=1, screen_dim=(125, 125))
        render_env_params = ppo_env_params.replace(pixels_per_unit=25)
        pixel_renderer = jax.jit(make_render_pixels(render_env_params, render_static_env_params))
        pixel_render_fn = lambda x: pixel_renderer(x) / 255.0
        video_fn = make_video_fn(render_env_params, render_static_env_params, pixel_render_fn)

        initial_extra = new_extra if new_extra is not None else {"rms": rms_init(jax.tree.map(lambda x: x[0], obsv))}

        def _vmapped_eval_step(runner_state, rng):
            eval_fn = make_eval_fn(
                eval_env,
                ppo_env_params,
                config["eval_num_attempts"],
                observation_preprocessing_fn=lambda obs: _maybe_normalise(runner_state.extra["rms"], obs)[1],
                fixed_video_rng=jax.random.PRNGKey(102),
            )

            return eval_fn(rng, runner_state.train_state, all_eval_specs)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # squeeze the RNG (undoes expand_dims added at the end of the previous step for sharding)
            runner_state = runner_state._replace(rng=runner_state.rng.squeeze(0))

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    extra,
                    hstate,
                    rng,
                    update_step,
                ) = runner_state

                extra["rms"], obs_to_use = _maybe_normalise(extra["rms"], last_obs, should_update=True)
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                ac_in = (
                    jax.tree.map(lambda x: x[np.newaxis, :], obs_to_use),
                    last_done[np.newaxis, :],
                )

                hstate, pi, value = train_state.apply_fn(train_state.params, hstate, ac_in)
                obs_to_save = last_obs

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)

                valid_action_mask = get_valid_action_mask(env_state, static_env_params, action)

                obsv, env_state, reward, done, info = jax.vmap(ppo_env.step, in_axes=(0, 0, 0, None))(
                    jax.random.split(_rng, config["num_train_envs"]),
                    env_state,
                    action,
                    ppo_env_params,
                )

                transition = Transition(
                    last_done, action, value, reward, log_prob, obs_to_save, info, valid_action_mask=valid_action_mask
                )
                runner_state = RunnerState(train_state, env_state, obsv, done, extra, hstate, rng, update_step)
                return runner_state, transition

            initial_hstate = runner_state.hstate
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["num_steps"])

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                extra,
                hstate,
                rng,
                update_step,
            ) = runner_state
            _, obs_to_use = _maybe_normalise(extra["rms"], last_obs, should_update=False)
            ac_in = (
                jax.tree.map(lambda x: x[np.newaxis, :], obs_to_use),
                last_done[np.newaxis, :],
            )
            _, _, last_val = train_state.apply_fn(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["gamma"] * next_value * (1 - next_done) - value
                    gae = delta + config["gamma"] * config["gae_lambda"] * (1 - next_done) * gae
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, obs_to_use = jax.vmap(
                            functools.partial(_maybe_normalise, should_update=False),
                            (None, 0),
                        )(extra["rms"], traj_batch.obs)
                        _, pi, value = train_state.apply_fn(params, init_hstate[0], (obs_to_use, traj_batch.done))

                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
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

                        entropy_disentangled = pi.entropy_disentangled()
                        action_mask = traj_batch.valid_action_mask
                        entropy_disentangled_mean = (
                            (entropy_disentangled * action_mask).sum(axis=-1) / jnp.maximum(1, action_mask.sum(axis=-1))
                        ).mean()

                        total_loss = loss_actor + config["vf_coef"] * value_loss - config["ent_coef"] * entropy

                        return total_loss, {
                            "loss/value": value_loss,
                            "loss/entropy": entropy,
                            "loss/actor": loss_actor,
                            "loss/total": total_loss,
                            "metrics/entropy_disentangled": entropy_disentangled_mean,
                        }

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                    total_loss = jax.lax.pmean(total_loss, axis_name="devices")
                    g_squared, s, grad_norm, grads = compute_gns_metrics(grads, advantages.shape[0])
                    total_loss = total_loss[0], total_loss[1] | {
                        "metrics/g_squared": jnp.mean(g_squared),
                        "metrics/s": jnp.mean(s),
                        "metrics/grad_norm": jnp.mean(grad_norm),
                    }

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
                permutation = jax.random.permutation(_rng, config["num_train_envs"])
                batch = (init_hstate, traj_batch, advantages, targets)

                if config["full_minibatch_shuffle"]:
                    assert not config["recurrent_model"], "Full minibatch shuffle only works with non-recurrent models"

                    # Properly shuffle across ALL dimensions (steps and envs)
                    num_elements = config["num_steps"] * config["num_train_envs"]
                    full_permutation = jax.random.permutation(_rng, num_elements)

                    def reshuffle(x):
                        # x has shape (num_steps, num_train_envs, ...)
                        orig_shape = x.shape
                        x = x.reshape((num_elements,) + orig_shape[2:])
                        x = jnp.take(x, full_permutation, axis=0)
                        x = x.reshape(orig_shape)
                        return x

                    # We don't shuffle init_hstate because it's per-env, but it's irrelevant for non-recurrent models anyway.
                    # We shuffle traj_batch, advantages, and targets.
                    shuffled_traj_batch = jax.tree.map(reshuffle, traj_batch)
                    shuffled_advantages = reshuffle(advantages)
                    shuffled_targets = reshuffle(targets)

                    shuffled_batch = (init_hstate, shuffled_traj_batch, shuffled_advantages, shuffled_targets)
                else:
                    shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)

                minibatches = jax.tree.map(
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
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = jax.tree.map(lambda x: x[None, :], initial_hstate)
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
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum() / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            metrics_to_log = jax.tree.map(lambda x: x.mean(), loss_info[1])
            metrics_to_log["metrics/gns"] = metrics_to_log["metrics/s"] / metrics_to_log["metrics/g_squared"]
            rng = update_state[-1]

            if config["use_wandb"]:
                param_metrics = {"metrics/param_norm": weight_norm(train_state.params)}

                def _real_eval(rng, update_step):
                    rng, _rng = jax.random.split(rng)

                    # eval
                    eval_metrics = _vmapped_eval_step(runner_state, _rng)

                    # maybe make video
                    vid_frequency = get_video_frequency(config, update_step)
                    should_log_videos = update_step % vid_frequency == 0
                    videos = video_fn(should_log_videos, eval_metrics)

                    # return just episode metrics
                    ep_metrics = {k: v.episode_metrics for k, v in eval_metrics.items()}
                    return (ep_metrics, True), (videos, should_log_videos)

                def _fake_eval(rng, update_step):
                    fake_videos = {
                        k: make_fake_video(
                            render_env_params,
                            render_static_env_params,
                            v.num_envs_to_video if v.num_envs_to_video > 0 else v.number_of_levels,
                        )
                        for k, v in all_eval_specs.items()
                        if v.plot_videos
                    }
                    ep_metrics = {k: EpisodeMetrics.create_empty(v.number_of_levels) for k, v in all_eval_specs.items()}
                    return (ep_metrics, False), (fake_videos, False)

                rng, _rng = jax.random.split(rng)
                should_eval = jnp.logical_and(config["eval_freq"] > 0, update_step % config["eval_freq"] == 0)
                all_eval_metrics, all_video_metrics = jax.lax.cond(
                    should_eval,
                    _real_eval,
                    _fake_eval,
                    _rng,
                    update_step,
                )

                def callback(
                    raw_info,
                    update_step,
                    all_eval_metrics,
                    should_log_evals,
                    all_video_metrics,
                    should_log_videos,
                    metrics_to_log,
                ):
                    nonlocal last_time
                    time_now = time.time()
                    delta_time = time_now - last_time
                    last_time = time_now
                    dones = raw_info["returned_episode"]
                    to_log = {
                        "episode_return": (raw_info["returned_episode_returns"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "episode_solved": (raw_info["returned_episode_solved"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "episode_length": (raw_info["returned_episode_lengths"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "num_completed_episodes": dones.sum(),
                        **metrics_to_log,
                    }
                    to_log["timing/num_updates"] = update_step
                    to_log["timing/num_model_forward_passes"] = int(update_step) * (
                        int(config["num_steps"]) + config["num_minibatches"] * config["update_epochs"]
                    )
                    to_log["timing/num_model_backward_passes"] = int(update_step) * (
                        config["num_minibatches"] * config["update_epochs"]
                    )
                    to_log["timing/num_env_steps"] = (
                        int(update_step) * int(config["num_steps"]) * int(config["num_train_envs"]) * int(NUM_GPUS)
                    )
                    to_log["timing/sps"] = (
                        int(config["num_steps"]) * int(config["num_train_envs"]) * int(NUM_GPUS)
                    ) / delta_time
                    to_log["timing/sps_agg"] = (to_log["timing/num_env_steps"]) / (time_now - time_start)

                    if metrics_to_log["device_id"] != 0:
                        return
                    to_log |= create_eval_metrics_dict_for_logging(
                        all_eval_specs,
                        all_eval_metrics if should_log_evals else None,
                        all_video_metrics if should_log_videos else None,
                    )
                    wandb.log(to_log)

                metrics_to_log = jax.lax.pmean(metrics_to_log, axis_name="devices")
                metrics_to_log["device_id"] = jax.lax.axis_index("devices")
                metrics_to_log |= param_metrics
                jax.debug.callback(
                    callback, traj_batch.info, update_step, *all_eval_metrics, *all_video_metrics, metrics_to_log
                )

            runner_state = RunnerState(
                train_state,
                env_state,
                last_obs,
                last_done,
                extra,
                hstate,
                jnp.expand_dims(rng, axis=0),
                update_step + 1,
            )
            return runner_state, metric

        runner_state_spec = RunnerState(
            train_state=PartitionSpec(),  # replicated
            env_state=PartitionSpec("devices"),
            last_obs=PartitionSpec("devices"),
            last_done=PartitionSpec("devices"),
            extra=PartitionSpec(),  # rms norm: replicated
            hstate=PartitionSpec("devices"),
            rng=PartitionSpec("devices"),
            update_step=PartitionSpec(),  # replicated
        )

        _update_step_shard_mapped = jax.jit(
            jax.shard_map(
                _update_step,
                mesh=mesh,
                in_specs=(
                    runner_state_spec,
                    PartitionSpec(),
                ),
                out_specs=(runner_state_spec, PartitionSpec()),
                check_vma=False,
            )
        )

        rng, _rng = jax.random.split(rng)
        # put on appropriate devices
        sharded_rng = jax.random.split(_rng, NUM_GPUS)
        replicated_train_state = jax.device_put(train_state, replicated_sharding)
        replicated_extra = jax.device_put(initial_extra, replicated_sharding)
        replicated_update_step = jax.device_put(jnp.array(0), replicated_sharding)
        # and run
        runner_state = RunnerState(
            train_state=replicated_train_state,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            extra=replicated_extra,
            hstate=init_hstate,
            rng=sharded_rng,
            update_step=replicated_update_step,
        )

        if return_update_step_and_init_state:
            return _update_step_shard_mapped, runner_state

        how_many_checkpoints_do_we_save = math.ceil(config["num_updates"] / config["checkpoint_save_freq"])
        if jax.process_index() == 0:
            logger.info(
                f"Saving {how_many_checkpoints_do_we_save} checkpoints, frequency={config['checkpoint_save_freq']}, total updates={config['num_updates']}"
            )

        def _single_checkpoint_step(runner_state, checkpoint_no):
            runner_state, metric = jax.lax.scan(
                _update_step_shard_mapped, runner_state, None, config["checkpoint_save_freq"]
            )

            def _callback(checkpoint_no, train_state, extra):
                if config["save_policy"] and jax.process_index() == 0:
                    update_number = int(checkpoint_no) * config["checkpoint_save_freq"]
                    timesteps_here = (
                        (int(update_number) + int(config["checkpoint_save_freq"]))
                        * int(config["num_steps"])
                        * int(config["num_train_envs"])
                        * int(NUM_GPUS)
                    )
                    logger.info(
                        f"Saving checkpoint at update = {update_number} | env steps = {timesteps_here/1e9:.2f}B"
                    )
                    save_model(
                        train_state,
                        timesteps_here,
                        config,
                        save_to_wandb=config["use_wandb"],
                        extra=extra,
                    )

            jax.debug.callback(_callback, checkpoint_no, runner_state.train_state, runner_state.extra)
            return runner_state, metric

        runner_state, metric = jax.lax.scan(
            _single_checkpoint_step, runner_state, jnp.arange(how_many_checkpoints_do_we_save)
        )

        return {"runner_state": runner_state, "metric": metric}

    return train


@hydra.main(version_base=None, config_path="../configs", config_name="ppo")
def main(config):
    process_id = jax.process_index()

    name = "PPO"
    if jax.process_count() > 1:
        name += "-" + f"{jax.device_count()}-GPUs"
    config = normalise_config(OmegaConf.to_container(config), name)
    env_params, static_env_params = generate_params_from_config(config)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    if config["use_wandb"]:
        if process_id == 0:
            init_wandb(config, name, settings=wandb.Settings(quiet=True))
        else:
            os.environ["WANDB_MODE"] = "disabled"

    rng = jax.random.PRNGKey(config["seed"])
    rng, _rng = jax.random.split(rng)
    t = time.time()
    train = jax.jit(make_train(config, env_params, static_env_params)).lower(_rng).compile()
    if jax.process_index() == 0:
        logger.info(f"Took {time.time() - t:.2f}s to compile")
    train(_rng)


if __name__ == "__main__":
    main()
