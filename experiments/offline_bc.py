# region imports
import gc
import multiprocessing
import os
import platform
import sys

import jax

from kinetix.environment import static_env_params_from_size

# Try to initialize jax.distributed (no-op / safe to fail in single-process runs)
try:
    jax.distributed.initialize(local_device_ids="")  # local_device_ids="" to let jax pick devices
    print("[INFO] Running across multiple nodes")
except Exception as e:
    print("[INFO] Running in single-node mode")

from absl import flags

# If there are multiple workers
is_worker = "grain_worker" in sys.argv or os.environ.get("GRAIN_WORKER_ID") is not None
if multiprocessing.parent_process() is not None or is_worker:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    setattr(flags.FLAGS, "jax_allow_unused_gpus", True)


os.environ["WANDB_DISABLE_SERVICE"] = "True"
import time

import hydra
import jax.experimental
import jax.numpy as jnp
import optax
import wandb
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.sharding import PartitionSpec
from omegaconf import OmegaConf

from kinetix.data import (
    ActionEnvStateMask,
    ShuffledDatasetManager,
    RunnerState,
    RunnerStateAndData,
    TrainAndValData,
    TrajectoryDatasetManager,
)
from kinetix.environment import make_reset_fn_from_config
from kinetix.models import GeneralActorCriticRNN, make_network_from_config
from kinetix.render import make_render_pixels
from kinetix.util import (
    EvalSpec,
    RunningMeanStandard,
    create_eval_metrics_dict_for_logging,
    expand_env_state,
    generate_params_from_config,
    get_eval_level_groups,
    get_video_frequency,
    init_wandb,
    load_evaluation_levels,
    load_train_state_from_wandb_artifact_path,
    make_eval_fn,
    make_video_fn,
    maybe_normalise,
    normalise_config,
    parallel_rms_update,
    rms_init_from_batch,
    rms_normalise,
    save_model,
)
from kinetix.util.train_utils import (
    EMA,
    compute_gns_metrics,
    get_logger,
    get_randomly_sampled_eval_levels,
    make_env,
    weight_norm,
)

# endregion


class TrainStateWithRMSNorm(TrainState):
    rms_norm: RunningMeanStandard


logger = get_logger()


def _to_python_scalar(path, x):
    try:
        return x.item()
    except Exception:
        return x


def _dim_mean(x, mask):
    """Weighted mean over action dims (axis=-1); x and mask are [N, D]."""
    return (x * mask).sum(-1) / jnp.maximum(mask.sum(-1), 1)


def _example_mean(x, mask):
    """Weighted mean over examples (axis=0); x is [N] or [N, D], mask matches."""
    return (x * mask).sum(0) / jnp.maximum(mask.sum(0), 1)


def make_run_bc(bc_config, bc_network, bc_render_fn):
    vmapped_render_fn = jax.vmap(jax.vmap(jax.jit(bc_render_fn)))

    def _loss_fn(params, xs_mb, ys_mb, expert_return_mask_mb, action_mask_mb, rng, traj_done_mb=None):
        # xs_mb: (n_traj, T, *obs), ys_mb: (n_traj, T, num_actions)
        n_traj, T = ys_mb.shape[:2]
        N = n_traj * T
        hstate = GeneralActorCriticRNN.initialize_carry(n_traj)
        xs_mb_t = jax.tree.map(lambda x: x.swapaxes(0, 1), xs_mb)
        _rnn_dones = traj_done_mb.swapaxes(0, 1) if traj_done_mb is not None else jnp.zeros((T, n_traj), dtype=bool)
        extra_kwargs = (
            {}
            if rng is None or bc_config["dropout_prob"] <= 0.0
            else {"deterministic": False, "rngs": {"dropout": rng}}
        )
        _, pi, _ = bc_network.apply(params, hstate, (xs_mb_t, _rnn_dones), **extra_kwargs)

        entropy_disentangled = pi.entropy_disentangled().reshape(N, -1)
        lp_per_dim = pi.log_prob_disentangled(ys_mb.swapaxes(0, 1)).reshape(N, -1)
        ys_mb = ys_mb.reshape(N, -1)
        action_mask_mb = action_mask_mb.reshape(N, -1)
        expert_return_mask_mb = expert_return_mask_mb.reshape(N)

        entropy_mean = entropy_disentangled.sum(axis=-1).mean() / action_mask_mb.shape[-1]
        entropy_disentangled_mean = _dim_mean(entropy_disentangled, action_mask_mb).mean()

        log_prob_data_disentangled = _example_mean(_dim_mean(lp_per_dim, action_mask_mb), expert_return_mask_mb)
        log_prob_data_normal = _example_mean(lp_per_dim.sum(-1), expert_return_mask_mb)

        erm = expert_return_mask_mb[:, None]
        per_dim_unmasked = _example_mean(lp_per_dim, erm)
        per_dim_masked = _example_mean(lp_per_dim, action_mask_mb * erm)

        if bc_config["disentangled_bc_loss"]:
            log_prob_data_to_use = log_prob_data_disentangled
        else:
            log_prob_data_to_use = log_prob_data_normal
        total_loss = -log_prob_data_to_use

        per_dim_metrics = {}
        for _i in range(lp_per_dim.shape[-1]):
            per_dim_metrics[f"metrics/per_dim_unmasked_nll_{_i}"] = -per_dim_unmasked[_i]
            per_dim_metrics[f"metrics/per_dim_masked_nll_{_i}"] = -per_dim_masked[_i]

        loss_dict = {
            "loss/total": total_loss,
            "loss/bc": -log_prob_data_to_use,
            "metrics/entropy": entropy_mean * action_mask_mb.shape[-1],
            "metrics/entropy_normalised": entropy_mean,
            "metrics/entropy_disentangled": entropy_disentangled_mean,
            "metrics/disentangled_nll_data": -log_prob_data_disentangled,
            "metrics/normal_nll_data": -log_prob_data_normal,
            "metrics/motors_masked_nll": -per_dim_masked[:-2].mean(),
            "metrics/motors_unmasked_nll": -per_dim_unmasked[:-2].mean(),
            "metrics/thrusters_masked_nll": -per_dim_masked[-2:].mean(),
            "metrics/thrusters_unmasked_nll": -per_dim_unmasked[-2:].mean(),
            **per_dim_metrics,
        }
        return total_loss, loss_dict

    @jax.jit
    def step(runner_state: RunnerState, data: TrainAndValData) -> tuple[RunnerState, dict]:
        if jax.process_index() == 0:
            logger.info("JIT::step")

        train_data, val_data = data.train_data, data.val_data
        _use_traj = bc_config["use_trajectory_data"]
        if not _use_traj:
            train_data = jax.tree.map(lambda x: x[:, None], train_data)

        ys_train = train_data.action
        done_train = (
            train_data.done
            if (_use_traj and train_data.done is not None)
            else jnp.zeros(ys_train.shape[:2], dtype=bool)
        )
        xs_train = train_data.env_state
        expert_return_mask = train_data.mask
        action_mask = train_data.action_mask

        train_state = runner_state.train_state
        rng = runner_state.rng
        update_step = runner_state.update_step

        xs_mb = vmapped_render_fn(xs_train)
        xs_mb = maybe_normalise(bc_config["rms_norm"], train_state.rms_norm, xs_mb)

        rng, _rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (total_loss, loss_dict), grads = grad_fn(
            train_state.params, xs_mb, ys_train, expert_return_mask, action_mask, _rng, done_train
        )

        new_rms_norm = (
            parallel_rms_update(train_state.rms_norm, xs_mb) if bc_config["rms_norm"] else train_state.rms_norm
        )
        total_loss = jax.lax.pmean(total_loss, axis_name="devices")
        g_squared, s, grad_norm, grads = compute_gns_metrics(grads, ys_train.shape[0] * ys_train.shape[1])
        train_state = train_state.apply_gradients(grads=grads).replace(rms_norm=new_rms_norm)

        val_losses = {}
        for k, v in val_data.items():
            v_traj = v if _use_traj else jax.tree.map(lambda x: x[:, None], v)
            _v_done = v_traj.done if v_traj.done is not None else jnp.zeros(v_traj.action.shape[:2], dtype=bool)
            _v_xs = maybe_normalise(bc_config["rms_norm"], train_state.rms_norm, vmapped_render_fn(v_traj.env_state))
            val_loss, val_loss_dict = _loss_fn(
                train_state.params, _v_xs, v_traj.action, v_traj.mask, v_traj.action_mask, None, _v_done
            )
            val_losses[f"bc/{k}_loss"] = jnp.mean(val_loss)
            val_losses |= {
                f"loss/{k}_{name.replace('loss/', '')}": jnp.mean(val)
                for name, val in val_loss_dict.items()
                if name.startswith("loss/")
            }
        val_losses = jax.lax.pmean(val_losses, axis_name="devices")

        new_update_step = update_step + 1
        metrics = {
            "bc/training_loss": total_loss,
            "bc/update_step": new_update_step,
            "bc/mean_mask": expert_return_mask.mean(),
            **val_losses,
            **loss_dict,
            "opt_state/lr": train_state.opt_state.hyperparams["learning_rate"],
            "metrics/g_squared": jnp.mean(g_squared),
            "metrics/s": jnp.mean(s),
            "metrics/grad_norm": jnp.mean(grad_norm),
            "timing/num_update_steps": new_update_step,
            "timing/num_env_steps": (new_update_step * 1.0) * bc_config["minibatch_size"],
            "timing/num_unique_transitions_processed": (new_update_step * 1.0) * bc_config["minibatch_size"],
            "timing/num_batches_processed": new_update_step,
        }
        metrics = jax.lax.pmean(metrics, axis_name="devices")

        return RunnerState(train_state=train_state, update_step=new_update_step, rng=rng), metrics

    return step


@hydra.main(version_base=None, config_path="../configs", config_name="offline_bc")
def main(config):
    overall_start_time = time.time()

    # ── Process & device setup ────────────────────────────────────────────────
    process_id = jax.process_index()

    _last_time = overall_start_time

    def _log_time(msg):
        nonlocal _last_time
        curr = time.time()
        if process_id == 0:
            logger.info(f"STARTUP:: {msg} took {curr - _last_time:.2f}s")
        _last_time = curr

    num_processes = jax.process_count()
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=["devices"])

    # This replicates the data across all devices
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # splits a single array across all devices
    partitioned_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices"))

    logger.info(
        f"Process {process_id} of {num_processes} started Total / Local Devices = [%d] / [%d] [%s]",
        len(jax.devices()),
        len(jax.local_devices()),
        platform.node(),
    )

    # ── Config normalization ──────────────────────────────────────────────────
    config = OmegaConf.to_container(config)

    config = normalise_config(config, "SFL-BC-OFFLINE")
    config["num_updates"] = config["total_timesteps"] // config["minibatch_size"]
    NUM_GRADIENT_STEPS = config["num_updates"]

    assert not config["recurrent_model"] or config.get(
        "use_trajectory_data", False
    ), "recurrent_model=True requires use_trajectory_data=True (recurrent training needs sequential data)"

    # ── Environment setup ─────────────────────────────────────────────────────
    env_params, static_env_params = generate_params_from_config(config)

    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    config["num_gpus"] = jax.device_count()

    if process_id == 0:
        init_wandb(config, "OFFLINE-BC", settings=wandb.Settings(quiet=True))
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # Do this after wandb init so we have consistent params for multi and single gpu runs
    config["per_process_minibatch_size"] = config["minibatch_size"] // num_processes

    rng = jax.random.PRNGKey(config["seed"])

    env = make_env(config, static_env_params, env_params)
    _log_time("make_env and setup")

    sample_random_level = make_reset_fn_from_config(
        config, env_params, static_env_params, physics_engine=env.physics_engine
    )

    all_eval_levels, eval_static_env_params = load_evaluation_levels(config["eval_levels"])
    _log_time("load_evaluation_levels")

    if config["eval_on_sml_sampled"]:
        assert config["env_size_name"] == "l", "When eval_on_sml_sampled is true, the env size must be 'l'."

    eval_group_indices = get_eval_level_groups(config["eval_levels"])
    eval_static_env_params = eval_static_env_params.replace(downscale=static_env_params.downscale)
    eval_env = make_env(config, eval_static_env_params, env_params)

    render_static_env_params = eval_env.static_env_params.replace(downscale=1, screen_dim=(125, 125))
    render_env_params = env_params.replace(pixels_per_unit=25)

    pixel_renderer = jax.jit(make_render_pixels(render_env_params, render_static_env_params))
    render_fn_eval = lambda x: pixel_renderer(x) / 255.0
    network = make_network_from_config(env, env_params, config)
    _log_time("eval and network setup")

    # ── Network & optimizer init ──────────────────────────────────────────────
    rng, _rng = jax.random.split(rng)
    train_envs, timesteps = 32, 16  # arbitrary
    obs, _ = env.reset(rng, env_params, sample_random_level(rng))
    _log_time("env.reset (init network)")
    obs = jax.tree.map(
        lambda x: jnp.repeat(jnp.repeat(x[None, ...], train_envs, axis=0)[None, ...], timesteps, axis=0),
        obs,
    )

    init_x = (obs, jnp.zeros((timesteps, train_envs)))
    init_hstate = GeneralActorCriticRNN.initialize_carry(train_envs)
    network_params = {"params": jax.jit(network.init)(_rng, init_hstate, init_x)["params"]}
    _log_time("network.init (actual call)")

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(network_params))
    if process_id == 0:
        logger.info(f"Number of parameters: {param_count/1e6:.2f}M")

    if config["anneal_lr"]:

        def linear_schedule(count):
            frac = 1.0 - count / NUM_GRADIENT_STEPS
            return config["lr"] * frac

        assert not config["warmup_lr"], "Choose only one of anneal_lr and warmup_lr"
        lr_to_use = linear_schedule
    elif config["warmup_lr"]:

        def warmup_schedule(count):
            frac_through_training = count / NUM_GRADIENT_STEPS
            warmup_frac = config["warmup_frac"]
            is_warmup_phase = frac_through_training < warmup_frac

            lr = config["lr"] * (
                is_warmup_phase * (frac_through_training / warmup_frac)
                + (1 - is_warmup_phase) * (1.0 - (frac_through_training - warmup_frac) / (1.0 - warmup_frac))
            )

            return lr

        lr_to_use = warmup_schedule
    else:
        lr_to_use = lambda count: config["lr"]
    train_state = TrainStateWithRMSNorm.create(
        apply_fn=network.apply,
        params=network_params,
        tx=optax.inject_hyperparams(
            lambda learning_rate: optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adamw(learning_rate=learning_rate, eps=1e-5, weight_decay=config["weight_decay"]),
            )
        )(learning_rate=lr_to_use),
        rms_norm=rms_init_from_batch(obs, config["observation_type"]),
    )
    _log_time("TrainState create")

    # ── Checkpoint loading ────────────────────────────────────────────────────
    if config["load_from_checkpoint"] is not None:
        logger.info(f"Loading checkpoint from {config['load_from_checkpoint']}")

        path = f"{config['checkpoint_download_dir']}/{config['hash']}/{process_id}"
        os.makedirs(path, exist_ok=True)

        train_state, loaded_extra = load_train_state_from_wandb_artifact_path(
            train_state,
            config["load_from_checkpoint"],
            load_only_params=config["load_only_params"],
            specific_dir=path,
            return_extra=True,
        )
        if config["rms_norm"]:
            train_state = train_state.replace(rms_norm=loaded_extra["rms"])
        _log_time("Loading checkpoint")

    # ── Dataset loading ───────────────────────────────────────────────────────
    st = time.time()
    _val_batch_size = config["validation_batch_size"]
    _dataset_common = dict(
        dataset_dir=config["dataset_dir"],
        static_env_params=static_env_params,
        seed=config["seed"],
        maximum_number_of_shards=config["maximum_number_of_shards"],
        n_val_shards=1,
        shard_index=process_id,
        total_shard_count=num_processes,
        should_expand_static_env_params=config["should_expand_static_env_params"],
    )
    if config.get("use_trajectory_data", False):
        _TRAJ_T = 256
        dataset_manager = TrajectoryDatasetManager(
            batch_size=config["minibatch_size"] // num_processes // _TRAJ_T,
            val_batch_size=_val_batch_size // _TRAJ_T,
            **_dataset_common,
        )
    else:
        dataset_manager = ShuffledDatasetManager(
            batch_size=config["minibatch_size"] // num_processes,
            val_batch_size=_val_batch_size,
            dataset_proportions=config["dataset_proportions"],
            **_dataset_common,
        )

    assert (
        dataset_manager.validation_batch is not None
    ), "Validation batch not loaded — n_val_shards=1 requires at least 2 shards in the dataset"
    validation_datasets = {"validation": dataset_manager.validation_batch}

    if process_id == 0:
        logger.info(
            f"Full Dataset Size:: {dataset_manager.length * config['minibatch_size'] / 1e6:.2f}M transitions. Global / Per Process Minibatch Size = {config['minibatch_size'] // 1024}k / {config['per_process_minibatch_size'] // 1024}k"
        )

    validation_datasets = jax.device_put(validation_datasets, partitioned_sharding)

    single_batch = dataset_manager.load_next_batch()

    _use_traj = config["use_trajectory_data"]
    _total_timesteps = single_batch.action.shape[0] * (single_batch.action.shape[1] if _use_traj else 1)
    assert _total_timesteps >= config["per_process_minibatch_size"], "Dataset smaller than minibatch size"

    et = time.time()

    if process_id == 0:
        n = _total_timesteps
        logger.info(
            f"Loaded a dataset with {n / 1e3:.1f}k transitions in {et - st:.2f}s: {n / (et - st) / 1e3:.1f}k transitions/s"
        )
        n_val = validation_datasets["validation"].action.shape[0] * (
            validation_datasets["validation"].action.shape[1] if _use_traj else 1
        )
        logger.info(f"Validation dataset has {n_val} transitions")

    # ── BC step function ──────────────────────────────────────────────────────
    run_single_bc_step = make_run_bc(
        config,
        network,
        bc_render_fn=env.observation_type.get_obs,
    )

    # ── Eval specs ────────────────────────────────────────────────────────────
    NUM_EVAL_DR_LEVELS = 512
    DR_EVAL_LEVELS = get_randomly_sampled_eval_levels(config, env_params, static_env_params, env, NUM_EVAL_DR_LEVELS)

    all_eval_specs = {
        "hand_designed": EvalSpec(
            levels_to_eval_on=all_eval_levels,
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

    if config["eval_on_sml_sampled"]:
        # on `l` env size; add sampled levels for s and m sizes expanded to l static params
        for size in ["s", "m"]:
            size_static_env_params, subsize_config = static_env_params_from_size(size, return_dict=True)
            sampled_levels = get_randomly_sampled_eval_levels(
                OmegaConf.to_container(OmegaConf.merge(config, subsize_config), resolve=True),
                env_params,
                size_static_env_params,
                make_env(config, size_static_env_params, env_params),
                NUM_EVAL_DR_LEVELS,
            )
            vmapped_expand = jax.vmap(lambda l: expand_env_state(l, static_env_params=eval_static_env_params))
            all_eval_specs[f"sampled_{size}"] = EvalSpec(
                levels_to_eval_on=vmapped_expand(sampled_levels),
                number_of_levels=NUM_EVAL_DR_LEVELS,
                level_names=[f"sml/{size}/{str(i).zfill(3)}" for i in range(NUM_EVAL_DR_LEVELS)],
                plot_videos=True,
                num_envs_to_video=10,
            )
    video_fn = make_video_fn(render_env_params, render_static_env_params, render_fn_eval)

    # ── Runner state init ─────────────────────────────────────────────────────
    rng, _rng = jax.random.split(rng)
    runner_state = RunnerState(train_state=train_state, update_step=0, rng=_rng)
    runner_state = jax.experimental.multihost_utils.broadcast_one_to_all(runner_state)

    # ── Training function definitions ─────────────────────────────────────────
    def _eval_step(runner_state, rng):
        train_state, update_count_in_eval, _ = runner_state
        test_metrics = {}
        _, rng_eval = jax.random.split(rng)
        observation_preprocessing_fn = None
        if config["rms_norm"]:
            observation_preprocessing_fn = lambda obs: rms_normalise(train_state.rms_norm, obs, flatten="auto")

        # make_eval_fn must be called here because observation_preprocessing_fn
        # closes over train_state.rms_norm which changes each eval step.
        eval_fn = make_eval_fn(
            eval_env,
            env_params,
            config["eval_num_attempts"],
            fixed_video_rng=jax.random.PRNGKey(102),
            observation_preprocessing_fn=observation_preprocessing_fn,
        )

        eval_metrics = eval_fn(rng_eval, train_state, all_eval_specs)
        if config["eval_on_sml_sampled"]:
            eval_metrics["sampled_l"] = eval_metrics["sampled"]

        # videos
        vf = get_video_frequency(config, update_count_in_eval)
        should_log_videos = update_count_in_eval % vf == 0
        videos = video_fn(should_log_videos, eval_metrics)

        test_metrics["eval"] = {
            "episode_metrics": {k: v.episode_metrics for k, v in eval_metrics.items()},
            "videos": videos,
            "should_log_videos": should_log_videos,
        }

        return test_metrics

    def _train_step(runner_state_and_data):
        # shard_map shards the device axis into a local axis, so squeeze rng from (1,) -> scalar and unsqueeze after
        rsd = runner_state_and_data
        rsd = rsd._replace(runner_state=rsd.runner_state._replace(rng=rsd.runner_state.rng.squeeze(0)))
        new_runner_state, metrics = run_single_bc_step(rsd.runner_state, rsd.batch_of_data)
        rsd = rsd._replace(runner_state=new_runner_state._replace(rng=new_runner_state.rng[None, ...]))
        return rsd, metrics

    train_partition_specs = RunnerStateAndData(
        RunnerState(
            train_state=PartitionSpec(),  # replicate
            update_step=PartitionSpec(),  # replicate
            rng=PartitionSpec("devices"),  # shard
        ),
        batch_of_data=PartitionSpec("devices"),
    )
    shard_mapped_train_step = jax.jit(
        jax.shard_map(
            _train_step,
            mesh=mesh,
            in_specs=(train_partition_specs,),
            out_specs=(
                train_partition_specs,
                PartitionSpec(),
            ),
            check_vma=False,
        )
    )

    # ── Sharded eval step ─────────────────────────────────────────────────────
    sharded_eval_step = jax.jit(
        jax.shard_map(
            _eval_step,
            mesh=mesh,
            in_specs=PartitionSpec(),
            out_specs=PartitionSpec(),
        )
    )

    def train_and_eval_step(
        runner_state: RunnerState,
        validation_datasets: dict[str, ActionEnvStateMask],
        eval_rng: jax.Array,
        eval_step: int,
        ema_gns_metrics: dict[str, EMA],
        compile_step_fn: bool,
    ):
        nonlocal last_log_time, shard_mapped_train_step
        test_metrics = {}
        time_start = time_train_start = time.time()

        rng, _rng = jax.random.split(runner_state.rng)
        runner_state = jax.device_put(runner_state, replicated_sharding)._replace(
            rng=jax.device_put(jax.random.split(_rng, jax.device_count()), partitioned_sharding)
        )

        total_times = {
            "disk": 0,
            "transfer": 0,
            "train": 0,
        }
        for inner_step in range(config["eval_freq"]):
            st = time.time()
            st_data = time.time()
            subset_of_data_one_device = dataset_manager.load_next_batch()
            et_data = time.time()

            total_times["disk"] += et_data - st_data

            if config["verbose_logging"]:
                logger.info("\tData selection took {:.2f}s".format(et_data - st_data))

            st_data = time.time()
            subset_of_data = jax.block_until_ready(
                jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(partitioned_sharding, x), subset_of_data_one_device
                )
            )
            et_data = time.time()
            total_times["transfer"] += et_data - st_data
            if config["verbose_logging"]:
                logger.info("\tData transfer took {:.2f}s".format(et_data - st_data))

            st_train = time.time()
            train_step_args = RunnerStateAndData(
                runner_state=runner_state,
                batch_of_data=TrainAndValData(train_data=subset_of_data, val_data=validation_datasets),
            )
            if inner_step == 0 and compile_step_fn:
                st_compile = time.time()
                shard_mapped_train_step = shard_mapped_train_step.lower(train_step_args).compile()
                logger.info(f"Took {time.time() - st_compile:.2f}s to compile `train_step`")

            runner_state_and_data, metrics = shard_mapped_train_step(train_step_args)
            runner_state = jax.block_until_ready(runner_state_and_data.runner_state)
            if config["verbose_logging"]:
                logger.info("\tTrain step took {:.2f}s".format(time.time() - st_train))

            et = time.time()
            total_times["train"] += et - st_train
            throughput = config["minibatch_size"] / (et - st)
            metrics["timing/throughput"] = throughput
            if process_id == 0:
                for k, current_ema in ema_gns_metrics.items():
                    ema_gns_metrics[k] = current_ema.update_ema(metrics[k])

                wandb.log(
                    {
                        **metrics,
                        **{k + "_ema": v.value for k, v in ema_gns_metrics.items()},
                        "metrics/gns_ema": ema_gns_metrics["metrics/s"].value
                        / ema_gns_metrics["metrics/g_squared"].value,
                        "metrics/param_norm": weight_norm(runner_state.train_state.params),
                    }
                )
            if config["verbose_logging"]:
                logger.info("Throughput {:.2f}K samples/s in {:.2f}s".format(throughput / 1e3, et - st))

        runner_state = runner_state._replace(rng=runner_state.rng[0])  # unshard rng

        _now = time.time()
        _mb = config["minibatch_size"]
        _freq = config["eval_freq"]
        test_metrics["train_metrics"] = {
            "update_count": runner_state.update_step,
            "timing/throughput": throughput,
            "timing/overall_throughput": _mb * _freq / (_now - last_log_time),
            "timing/overall_throughput_since_start": _mb * int(runner_state.update_step) / (_now - overall_start_time),
            **{f"timing/avg_time_{k}_batch": v / _freq for k, v in total_times.items()},
            **{f"timing/avg_time_{k}_element": v / _freq / _mb for k, v in total_times.items()},
        }

        last_log_time = _now
        t = _now - time_train_start
        if process_id == 0:
            logger.info(
                f"Timing:: Training {t:.2f}s -- [{eval_step} | {runner_state.update_step}] -- {_mb * _freq / t:.2f} samples/s"
            )
        gc.collect()

        time_eval_start = time.time()
        test_metrics.update(jax.block_until_ready(sharded_eval_step(runner_state, eval_rng)))

        if config["verbose_logging"]:
            logger.info(f"Timing:: Eval {time.time() - time_eval_start:.2f}s")
        test_metrics["train_metrics"]["timing/aggregate_throughput"] = _mb * _freq / (time.time() - time_start)

        return runner_state, test_metrics, ema_gns_metrics

    def log_eval(stats):
        eval_data = stats["eval"]

        def _aggregate_per_size(values, name):
            return {
                f"{name}_{group_name}": values[indices].mean() for group_name, indices in eval_group_indices.items()
            }

        log_dict = {
            **_aggregate_per_size(
                eval_data["episode_metrics"]["hand_designed"].episode_returns, "eval_info/aggregate_return"
            ),
            **_aggregate_per_size(
                eval_data["episode_metrics"]["hand_designed"].episode_solve_rates, "eval_info/aggregate_solve_rate"
            ),
            **create_eval_metrics_dict_for_logging(
                all_eval_specs | ({"sampled_l": all_eval_specs["sampled"]} if config["eval_on_sml_sampled"] else {}),
                eval_data["episode_metrics"],
                eval_data["videos"] if eval_data["should_log_videos"] else None,
            ),
        }
        if process_id == 0:
            wandb.log({k: v for k, v in stats.items() if k != "eval"} | log_dict)

    rng, _rng = jax.random.split(rng)
    # eval at the start
    if not config["skip_initial_eval"]:
        if process_id == 0:
            logger.info("Starting eval at beginning of training")
        st = time.time()
        metrics = sharded_eval_step(runner_state, _rng)
        log_eval(metrics | {"update_count": 0})
        if process_id == 0:
            logger.info("Finished initial eval in {:.2f}s".format(time.time() - st))
    last_log_time = time.time()

    # ── Main training loop ────────────────────────────────────────────────────
    ema_gns_metrics = {"metrics/s": EMA(0, 0, 0.99), "metrics/g_squared": EMA(0, 0, 0.99)}
    for eval_step in range(int(config["num_updates"] // config["eval_freq"])):
        start_time = time.time()
        rng, eval_rng = jax.random.split(rng)
        runner_state, metrics, ema_gns_metrics = train_and_eval_step(
            runner_state,
            validation_datasets,
            eval_rng,
            eval_step,
            ema_gns_metrics,
            compile_step_fn=config["compile_train_step"] and eval_step == 0,
        )
        curr_time = time.time()
        metrics["time_delta"] = curr_time - start_time
        metrics = jax.tree.map_with_path(_to_python_scalar, metrics)

        log_eval(metrics)
        if ((eval_step + 1) * config["eval_freq"]) % config["checkpoint_save_freq"] == 0:
            if config["save_path"] is not None:
                steps = int(metrics["train_metrics"]["update_count"]) * int(config["minibatch_size"])
                save_model(
                    runner_state.train_state,
                    steps,
                    config,
                    is_final=False,
                    save_to_wandb=config["use_wandb"],
                    extra={"rms": runner_state.train_state.rms_norm},
                )

    if config["save_policy"] and config["save_path"] is not None and process_id == 0:
        save_model(
            runner_state.train_state,
            config["total_timesteps"],
            config,
            is_final=True,
            save_to_wandb=config["use_wandb"],
            extra={"rms": runner_state.train_state.rms_norm},
        )


if __name__ == "__main__":
    main()
