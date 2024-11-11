from functools import partial
import time
from enum import IntEnum
from typing import Tuple

import chex
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState

import wandb
from kinetix.environment.ued.distributions import (
    create_random_starting_distribution,
)
from kinetix.environment.ued.ued import (
    make_mutate_env,
    make_reset_train_function_with_mutations,
    make_vmapped_filtered_level_sampler,
)
from kinetix.environment.ued.ued import (
    make_mutate_env,
    make_reset_train_function_with_list_of_levels,
    make_reset_train_function_with_mutations,
)
from kinetix.util.config import (
    generate_ued_params_from_config,
    get_video_frequency,
    init_wandb,
    normalise_config,
    save_data_to_local_file,
    generate_params_from_config,
    get_eval_level_groups,
)
from jaxued.environments.underspecified_env import EnvState
from jaxued.level_sampler import LevelSampler
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from flax.serialization import to_state_dict

import sys

sys.path.append("experiments")
from kinetix.environment.env import make_kinetix_env_from_name
from kinetix.environment.env_state import StaticEnvParams
from kinetix.environment.wrappers import (
    UnderspecifiedToGymnaxWrapper,
    LogWrapper,
    DenseRewardWrapper,
    AutoReplayWrapper,
)
from kinetix.models import make_network_from_config
from kinetix.render.renderer_pixels import make_render_pixels
from kinetix.models.actor_critic import ScannedRNN
from kinetix.util.learning import (
    general_eval,
    get_eval_levels,
    no_op_and_random_rollout,
    sample_trajectories_and_learn,
)
from kinetix.util.saving import (
    load_train_state_from_wandb_artifact_path,
    save_model_to_wandb,
)


class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1
    MUTATE = 2


def get_level_complexity_metrics(all_levels: EnvState, static_env_params: StaticEnvParams):
    def get_for_single_level(level):
        return {
            "complexity/num_shapes": level.polygon.active[static_env_params.num_static_fixated_polys :].sum()
            + level.circle.active.sum(),
            "complexity/num_joints": level.joint.active.sum(),
            "complexity/num_thrusters": level.thruster.active.sum(),
            "complexity/num_rjoints": (level.joint.active * jnp.logical_not(level.joint.is_fixed_joint)).sum(),
            "complexity/num_fjoints": (level.joint.active * (level.joint.is_fixed_joint)).sum(),
            "complexity/has_ball": ((level.polygon_shape_roles == 1) * level.polygon.active).sum()
            + ((level.circle_shape_roles == 1) * level.circle.active).sum(),
            "complexity/has_goal": ((level.polygon_shape_roles == 2) * level.polygon.active).sum()
            + ((level.circle_shape_roles == 2) * level.circle.active).sum(),
        }

    return jax.tree.map(lambda x: x.mean(), jax.vmap(get_for_single_level)(all_levels))


def get_ued_score_metrics(all_ued_scores):
    (mc, pvl, learn) = all_ued_scores
    scores = {}
    for score, name in zip([mc, pvl, learn], ["MaxMC", "PVL", "Learnability"]):
        scores[f"ued_scores/{name}/Mean"] = score.mean()
        scores[f"ued_scores_additional/{name}/Max"] = score.max()
        scores[f"ued_scores_additional/{name}/Min"] = score.min()

    return scores


class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int

    dr_last_level_batch_scores: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch_scores: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch_scores: chex.ArrayTree = struct.field(pytree_node=True)

    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

    dr_last_rollout_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_rollout_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_rollout_batch: chex.ArrayTree = struct.field(pytree_node=True)


# region PPO helper functions

# endregion


def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler) -> dict:
    """To prevent the entire (large) train_state to be copied to the CPU when doing logging, this function returns all of the important information in a dictionary format.

        Anything in the `log` key will be logged to wandb.

    Args:
        train_state (TrainState):
        level_sampler (LevelSampler):

    Returns:
        dict:
    """
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)
    return {
        "log": {
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
            "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
        },
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        },
    }


def compute_learnability(config, done, reward, info, num_envs):
    num_agents = 1
    BATCH_ACTORS = num_envs * num_agents

    rollout_length = config["num_steps"] * config["outer_rollout_steps"]

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
            collision = 0
            timeo = 0
            l = end_idx - start_idx
            return r, success, collision, timeo, l

        done_idxs = jnp.argwhere(dones, size=50, fill_value=max_steps).squeeze()
        mask_done = jnp.where(done_idxs == max_steps, 0, 1)
        ep_return, success, collision, timeo, length = __ep_outcomes(
            jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs
        )

        return {
            "ep_return": ep_return.mean(where=mask_done),
            "num_episodes": mask_done.sum(),
            "num_success": success.sum(where=mask_done),
            "success_rate": success.mean(where=mask_done),
            "collision_rate": collision.mean(where=mask_done),
            "timeout_rate": timeo.mean(where=mask_done),
            "ep_len": length.mean(where=mask_done),
        }

    done_by_env = done.reshape((-1, num_agents, num_envs))
    reward_by_env = reward.reshape((-1, num_agents, num_envs))
    o = _calc_outcomes_by_agent(rollout_length, done, reward, info)
    success_by_env = o["success_rate"].reshape((num_agents, num_envs))
    learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=0)

    return (
        learnability_by_env,
        o["num_episodes"].reshape(num_agents, num_envs).sum(axis=0),
        o["num_success"].reshape(num_agents, num_envs).T,
    )  # so agents is at the end.


def compute_score(
    config: dict, dones: chex.Array, values: chex.Array, max_returns: chex.Array, reward, info, advantages: chex.Array
) -> chex.Array:
    # Computes the score for each level
    if config["score_function"] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config["score_function"] == "pvl":
        return positive_value_loss(dones, advantages)
    elif config["score_function"] == "learnability":
        learnability, num_episodes, num_success = compute_learnability(
            config, dones, reward, info, config["num_train_envs"]
        )
        return learnability
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")


def compute_all_scores(
    config: dict,
    dones: chex.Array,
    values: chex.Array,
    max_returns: chex.Array,
    reward,
    info,
    advantages: chex.Array,
    return_success_rate=False,
):
    mc = max_mc(dones, values, max_returns)
    pvl = positive_value_loss(dones, advantages)
    learnability, num_episodes, num_success = compute_learnability(
        config, dones, reward, info, config["num_train_envs"]
    )
    if config["score_function"] == "MaxMC":
        main_score = mc
    elif config["score_function"] == "pvl":
        main_score = pvl
    elif config["score_function"] == "learnability":
        main_score = learnability
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")
    if return_success_rate:
        success_rate = num_success.squeeze(1) / jnp.maximum(num_episodes, 1)
        return main_score, (mc, pvl, learnability, success_rate)
    return main_score, (mc, pvl, learnability)


@hydra.main(version_base=None, config_path="../configs", config_name="plr")
def main(config=None):
    my_name = "PLR"
    config = OmegaConf.to_container(config)
    if config["ued"]["replay_prob"] == 0.0:
        my_name = "DR"
    elif config["ued"]["use_accel"]:
        my_name = "ACCEL"

    time_start = time.time()
    config = normalise_config(config, my_name)
    env_params, static_env_params = generate_params_from_config(config)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    run = init_wandb(config, my_name)
    config = wandb.config
    time_prev = time.time()

    def log_eval(stats, train_state_info):
        nonlocal time_prev
        print(f"Logging update: {stats['update_count']}")
        total_loss = jnp.mean(stats["losses"][0])
        if jnp.isnan(total_loss):
            print("NaN loss, skipping logging")
            raise ValueError("NaN loss")

        # generic stats
        env_steps = int(
            int(stats["update_count"]) * config["num_train_envs"] * config["num_steps"] * config["outer_rollout_steps"]
        )
        env_steps_delta = (
            config["eval_freq"] * config["num_train_envs"] * config["num_steps"] * config["outer_rollout_steps"]
        )
        time_now = time.time()
        log_dict = {
            "timing/num_updates": stats["update_count"],
            "timing/num_env_steps": env_steps,
            "timing/sps": env_steps_delta / (time_now - time_prev),
            "timing/sps_agg": env_steps / (time_now - time_start),
            "loss/total_loss": jnp.mean(stats["losses"][0]),
            "loss/value_loss": jnp.mean(stats["losses"][1][0]),
            "loss/policy_loss": jnp.mean(stats["losses"][1][1]),
            "loss/entropy_loss": jnp.mean(stats["losses"][1][2]),
        }
        time_prev = time_now

        # evaluation performance

        returns = stats["eval_returns"]
        log_dict.update({"eval/mean_eval_return": returns.mean()})
        log_dict.update({"eval/mean_eval_learnability": stats["eval_learn"].mean()})
        log_dict.update({"eval/mean_eval_solve_rate": stats["eval_solves"].mean()})
        log_dict.update({"eval/mean_eval_eplen": stats["eval_ep_lengths"].mean()})
        for i in range(config["num_eval_levels"]):
            log_dict[f"eval_avg_return/{config['eval_levels'][i]}"] = returns[i]
            log_dict[f"eval_avg_learnability/{config['eval_levels'][i]}"] = stats["eval_learn"][i]
            log_dict[f"eval_avg_solve_rate/{config['eval_levels'][i]}"] = stats["eval_solves"][i]
            log_dict[f"eval_avg_episode_length/{config['eval_levels'][i]}"] = stats["eval_ep_lengths"][i]
            log_dict[f"eval_get_max_eplen/{config['eval_levels'][i]}"] = stats["eval_get_max_eplen"][i]
            log_dict[f"episode_return_bigger_than_negative/{config['eval_levels'][i]}"] = stats[
                "episode_return_bigger_than_negative"
            ][i]

        def _aggregate_per_size(values, name):
            to_return = {}
            for group_name, indices in eval_group_indices.items():
                to_return[f"{name}_{group_name}"] = values[indices].mean()
            return to_return

        log_dict.update(_aggregate_per_size(returns, "eval_aggregate/return"))
        log_dict.update(_aggregate_per_size(stats["eval_solves"], "eval_aggregate/solve_rate"))

        if config["EVAL_ON_SAMPLED"]:
            log_dict.update({"eval/mean_eval_return_sampled": stats["eval_dr_returns"].mean()})
            log_dict.update({"eval/mean_eval_solve_rate_sampled": stats["eval_dr_solve_rates"].mean()})
            log_dict.update({"eval/mean_eval_eplen_sampled": stats["eval_dr_eplen"].mean()})

        # level sampler
        log_dict.update(train_state_info["log"])

        # images
        log_dict.update(
            {
                "images/highest_scoring_level": wandb.Image(
                    np.array(stats["highest_scoring_level"]), caption="Highest scoring level"
                )
            }
        )
        log_dict.update(
            {
                "images/highest_weighted_level": wandb.Image(
                    np.array(stats["highest_weighted_level"]), caption="Highest weighted level"
                )
            }
        )

        for s in ["dr", "replay", "mutation"]:
            if train_state_info["info"][f"num_{s}_updates"] > 0:
                log_dict.update(
                    {
                        f"images/{s}_levels": [
                            wandb.Image(np.array(image), caption=f"{score}")
                            for image, score in zip(stats[f"{s}_levels"], stats[f"{s}_scores"])
                        ]
                    }
                )
                if stats["log_videos"]:
                    # animations
                    rollout_ep = stats[f"{s}_ep_len"]
                    arr = np.array(stats[f"{s}_rollout"][:rollout_ep])
                    log_dict.update(
                        {
                            f"media/{s}_eval": wandb.Video(
                                arr.astype(np.uint8), fps=15, caption=f"{s.capitalize()} (len {rollout_ep})"
                            )
                        }
                    )
                #  * 255

        # DR, Replay and Mutate Returns
        dr_inds = (stats["update_state"] == UpdateState.DR).nonzero()[0]
        rep_inds = (stats["update_state"] == UpdateState.REPLAY).nonzero()[0]
        mut_inds = (stats["update_state"] == UpdateState.MUTATE).nonzero()[0]

        for name, inds in [
            ("DR", dr_inds),
            ("REPLAY", rep_inds),
            ("MUTATION", mut_inds),
        ]:
            if len(inds) > 0:
                log_dict.update(
                    {
                        f"{name}/episode_return": stats["episode_return"][inds].mean(),
                        f"{name}/mean_eplen": stats["returned_episode_lengths"][inds].mean(),
                        f"{name}/mean_success": stats["returned_episode_solved"][inds].mean(),
                        f"{name}/noop_return": stats["noop_returns"][inds].mean(),
                        f"{name}/noop_eplen": stats["noop_eplen"][inds].mean(),
                        f"{name}/noop_success": stats["noop_success"][inds].mean(),
                        f"{name}/random_return": stats["random_returns"][inds].mean(),
                        f"{name}/random_eplen": stats["random_eplen"][inds].mean(),
                        f"{name}/random_success": stats["random_success"][inds].mean(),
                    }
                )
                for k in stats:
                    if "complexity/" in k:
                        k2 = "complexity/" + name + "_" + k.replace("complexity/", "")
                        log_dict.update({k2: stats[k][inds].mean()})
                    if "ued_scores/" in k:
                        k2 = "ued_scores/" + name + "_" + k.replace("ued_scores/", "")
                        log_dict.update({k2: stats[k][inds].mean()})

        # Eval rollout animations
        if stats["log_videos"]:
            for i in range((config["num_eval_levels"])):
                frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
                frames = np.array(frames[:episode_length])
                log_dict.update(
                    {
                        f"media/eval_video_{config['eval_levels'][i]}": wandb.Video(
                            frames.astype(np.uint8), fps=15, caption=f"Len ({episode_length})"
                        )
                    }
                )

        wandb.log(log_dict)

    def get_all_metrics(
        rng,
        losses,
        info,
        init_env_state,
        init_obs,
        dones,
        grads,
        all_ued_scores,
        new_levels,
    ):
        noop_returns, noop_len, noop_success, random_returns, random_lens, random_success = no_op_and_random_rollout(
            env,
            env_params,
            rng,
            init_obs,
            init_env_state,
            config["num_train_envs"],
            config["num_steps"] * config["outer_rollout_steps"],
        )
        metrics = (
            {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "returned_episode_lengths": (info["returned_episode_lengths"] * dones).sum()
                / jnp.maximum(1, dones.sum()),
                "max_episode_length": info["returned_episode_lengths"].max(),
                "levels_played": init_env_state.env_state.env_state,
                "episode_return": (info["returned_episode_returns"] * dones).sum() / jnp.maximum(1, dones.sum()),
                "episode_return_v2": (info["returned_episode_returns"] * info["returned_episode"]).sum()
                / jnp.maximum(1, info["returned_episode"].sum()),
                "grad_norms": grads.mean(),
                "noop_returns": noop_returns,
                "noop_eplen": noop_len,
                "noop_success": noop_success,
                "random_returns": random_returns,
                "random_eplen": random_lens,
                "random_success": random_success,
                "returned_episode_solved": (info["returned_episode_solved"] * dones).sum()
                / jnp.maximum(1, dones.sum()),
            }
            | get_level_complexity_metrics(new_levels, static_env_params)
            | get_ued_score_metrics(all_ued_scores)
        )
        return metrics

    # Setup the environment.
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
            config, config["train_levels_list"], static_env_params, make_pcg_state=False, is_loading_train_levels=True
        )
    elif config["train_level_mode"] == "random":
        sample_random_level = make_reset_train_function_with_mutations(
            env.physics_engine, env_params, static_env_params, config, make_pcg_state=False
        )
    else:
        raise ValueError(f"Unknown train_level_mode: {config['train_level_mode']}")

    if config["use_accel"] and config["accel_start_from_empty"]:

        def make_sample_random_level():
            def inner(rng):
                def _inner_accel(rng):
                    return create_random_starting_distribution(
                        rng, env_params, static_env_params, ued_params, config["env_size_name"], controllable=True
                    )

                def _inner_accel_not_controllable(rng):
                    return create_random_starting_distribution(
                        rng, env_params, static_env_params, ued_params, config["env_size_name"], controllable=False
                    )

                rng, _rng = jax.random.split(rng)
                return _inner_accel(_rng)

            return inner

        sample_random_level = make_sample_random_level()

    sample_random_levels = make_vmapped_filtered_level_sampler(
        sample_random_level, env_params, static_env_params, config, make_pcg_state=False, env=env
    )

    def generate_world():
        raise NotImplementedError
        pass

    def generate_eval_world(rng, env_params, static_env_params, level_idx):
        # jax.random.split(jax.random.PRNGKey(101), num_levels), env_params, static_env_params, jnp.arange(num_levels)

        raise NotImplementedError

    _, eval_static_env_params = generate_params_from_config(
        config["eval_env_size_true"] | {"frame_skip": config["frame_skip"]}
    )
    eval_env = make_env(eval_static_env_params)
    ued_params = generate_ued_params_from_config(config)

    mutate_world = make_mutate_env(static_env_params, env_params, ued_params)

    def make_render_fn(static_env_params):
        render_fn_inner = make_render_pixels(env_params, static_env_params)
        render_fn = lambda x: render_fn_inner(x).transpose(1, 0, 2)[::-1]
        return render_fn

    render_fn = make_render_fn(static_env_params)
    render_fn_eval = make_render_fn(eval_static_env_params)
    if config["EVAL_ON_SAMPLED"]:
        NUM_EVAL_DR_LEVELS = 200
        key_to_sample_dr_eval_set = jax.random.PRNGKey(100)
        DR_EVAL_LEVELS = sample_random_levels(key_to_sample_dr_eval_set, NUM_EVAL_DR_LEVELS)

    # And the level sampler
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config["topk_k"]},
        duplicate_check=config["buffer_duplicate_check"],
    )

    @jax.jit
    def create_train_state(rng) -> TrainState:
        # Creates the train state
        def linear_schedule(count):
            frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / (
                config["num_updates"] * config["outer_rollout_steps"]
            )
            return config["lr"] * frac

        rng, _rng = jax.random.split(rng)
        init_state = jax.tree.map(lambda x: x[0], sample_random_levels(_rng, 1))

        rng, _rng = jax.random.split(rng)
        obs, _ = env.reset_to_level(_rng, init_state, env_params)
        ns = config["num_steps"] * config["outer_rollout_steps"]
        obs = jax.tree.map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], ns, axis=0),
            obs,
        )
        init_x = (obs, jnp.zeros((ns, config["num_train_envs"]), dtype=jnp.bool_))
        network = make_network_from_config(env, env_params, config)
        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, ScannedRNN.initialize_carry(config["num_train_envs"]), init_x)

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

        pholder_level = jax.tree.map(lambda x: x[0], sample_random_levels(jax.random.PRNGKey(0), 1))
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        pholder_level_batch = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level
        )
        pholder_rollout_batch = (
            jax.tree.map(
                lambda x: jnp.repeat(
                    jnp.expand_dims(x, 0), repeats=config["num_steps"] * config["outer_rollout_steps"], axis=0
                ),
                init_state,
            ),
            init_x[1][:, 0],
        )

        pholder_level_batch_scores = jnp.zeros((config["num_train_envs"],), dtype=jnp.float32)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch_scores=pholder_level_batch_scores,
            replay_last_level_batch_scores=pholder_level_batch_scores,
            mutation_last_level_batch_scores=pholder_level_batch_scores,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
            dr_last_rollout_batch=pholder_rollout_batch,
            replay_last_rollout_batch=pholder_rollout_batch,
            mutation_last_rollout_batch=pholder_rollout_batch,
        )

        if config["load_from_checkpoint"] != None:
            print("LOADING from", config["load_from_checkpoint"], "with only params =", config["load_only_params"])
            train_state = load_train_state_from_wandb_artifact_path(
                train_state,
                config["load_from_checkpoint"],
                load_only_params=config["load_only_params"],
                legacy=config["load_legacy_checkpoint"],
            )
        return train_state

    all_eval_levels = get_eval_levels(config["eval_levels"], eval_env.static_env_params)
    eval_group_indices = get_eval_level_groups(config["eval_levels"])

    @jax.jit
    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """
        This is the main training loop. It basically calls either `on_new_levels`, `on_replay_levels`, or `on_mutate_levels` at every step.
        """

        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
            Samples new (randomly-generated) levels and evaluates the policy on these. It also then adds the levels to the level buffer if they have high-enough scores.
            The agent is updated on these trajectories iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler

            # Reset
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = sample_random_levels(rng_levels, config["num_train_envs"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params
            )
            init_hstate = ScannedRNN.initialize_carry(config["num_train_envs"])
            # Rollout
            (
                (rng, train_state, new_hstate, last_obs, last_env_state),
                (
                    obs,
                    actions,
                    rewards,
                    dones,
                    log_probs,
                    values,
                    info,
                    advantages,
                    targets,
                    losses,
                    grads,
                    rollout_states,
                ),
            ) = sample_trajectories_and_learn(
                env,
                env_params,
                config,
                rng,
                train_state,
                init_hstate,
                init_obs,
                init_env_state,
                update_grad=config["exploratory_grad_updates"],
                return_states=True,
            )
            max_returns = compute_max_returns(dones, rewards)
            scores, all_ued_scores = compute_all_scores(config, dones, values, max_returns, rewards, info, advantages)
            sampler, _ = level_sampler.insert_batch(sampler, new_levels, scores, {"max_return": max_returns})
            rng, _rng = jax.random.split(rng)
            metrics = {
                "update_state": UpdateState.DR,
            } | get_all_metrics(_rng, losses, info, init_env_state, init_obs, dones, grads, all_ued_scores, new_levels)

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
                dr_last_level_batch_scores=scores,
                dr_last_rollout_batch=jax.tree.map(
                    lambda x: x[:, 0], (rollout_states.env_state.env_state.env_state, dones)
                ),
            )
            return (rng, train_state), metrics

        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
            This samples levels from the level buffer, and updates the policy on them.
            """
            sampler = train_state.sampler

            # Collect trajectories on replay levels
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(
                sampler, rng_levels, config["num_train_envs"]
            )
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params
            )
            init_hstate = ScannedRNN.initialize_carry(config["num_train_envs"])
            (
                (rng, train_state, new_hstate, last_obs, last_env_state),
                (
                    obs,
                    actions,
                    rewards,
                    dones,
                    log_probs,
                    values,
                    info,
                    advantages,
                    targets,
                    losses,
                    grads,
                    rollout_states,
                ),
            ) = sample_trajectories_and_learn(
                env,
                env_params,
                config,
                rng,
                train_state,
                init_hstate,
                init_obs,
                init_env_state,
                update_grad=True,
                return_states=True,
            )

            max_returns = jnp.maximum(
                level_sampler.get_levels_extra(sampler, level_inds)["max_return"], compute_max_returns(dones, rewards)
            )
            scores, all_ued_scores = compute_all_scores(config, dones, values, max_returns, rewards, info, advantages)
            sampler = level_sampler.update_batch(sampler, level_inds, scores, {"max_return": max_returns})

            rng, _rng = jax.random.split(rng)
            metrics = {
                "update_state": UpdateState.REPLAY,
            } | get_all_metrics(_rng, losses, info, init_env_state, init_obs, dones, grads, all_ued_scores, levels)
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
                replay_last_level_batch_scores=scores,
                replay_last_rollout_batch=jax.tree.map(
                    lambda x: x[:, 0], (rollout_states.env_state.env_state.env_state, dones)
                ),
            )
            return (rng, train_state), metrics

        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
            This mutates the previous batch of replay levels and potentially adds them to the level buffer.
            This also updates the policy iff `config["exploratory_grad_updates"]` is True.
            """

            sampler = train_state.sampler
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)

            # mutate
            parent_levels = train_state.replay_last_level_batch
            child_levels = jax.vmap(mutate_world, (0, 0, None))(
                jax.random.split(rng_mutate, config["num_train_envs"]), parent_levels, config["num_edits"]
            )
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, config["num_train_envs"]), child_levels, env_params
            )

            init_hstate = ScannedRNN.initialize_carry(config["num_train_envs"])
            # rollout
            (
                (rng, train_state, new_hstate, last_obs, last_env_state),
                (
                    obs,
                    actions,
                    rewards,
                    dones,
                    log_probs,
                    values,
                    info,
                    advantages,
                    targets,
                    losses,
                    grads,
                    rollout_states,
                ),
            ) = sample_trajectories_and_learn(
                env,
                env_params,
                config,
                rng,
                train_state,
                init_hstate,
                init_obs,
                init_env_state,
                update_grad=config["exploratory_grad_updates"],
                return_states=True,
            )

            max_returns = compute_max_returns(dones, rewards)
            scores, all_ued_scores = compute_all_scores(config, dones, values, max_returns, rewards, info, advantages)
            sampler, _ = level_sampler.insert_batch(sampler, child_levels, scores, {"max_return": max_returns})

            rng, _rng = jax.random.split(rng)
            metrics = {"update_state": UpdateState.MUTATE,} | get_all_metrics(
                _rng, losses, info, init_env_state, init_obs, dones, grads, all_ued_scores, child_levels
            )

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=child_levels,
                mutation_last_level_batch_scores=scores,
                mutation_last_rollout_batch=jax.tree.map(
                    lambda x: x[:, 0], (rollout_states.env_state.env_state.env_state, dones)
                ),
            )
            return (rng, train_state), metrics

        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)

        # The train step makes a decision on which branch to take, either on_new, on_replay or on_mutate.
        # on_mutate is only called if the replay branch has been taken before (as it uses `train_state.update_state`).
        branches = [
            on_new_levels,
            on_replay_levels,
        ]
        if config["use_accel"]:
            s = train_state.update_state
            branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s
            branches.append(on_mutate_levels)
        else:
            branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)

        return jax.lax.switch(branch, branches, rng, train_state)

    @partial(jax.jit, static_argnums=(2,))
    def eval(rng: chex.PRNGKey, train_state: TrainState, keep_states=True):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["eval_levels"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        num_levels = config["num_eval_levels"]
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

    @partial(jax.jit, static_argnums=(2,))
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

    @jax.jit
    def train_and_eval_step(runner_state, _):
        """
        This function runs the train_step for a certain number of iterations, and then evaluates the policy.
        It returns the updated train state, and a dictionary of metrics.
        """
        # Train
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])

        # Eval
        metrics["update_count"] = (
            train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        )

        vid_frequency = get_video_frequency(config, metrics["update_count"])
        should_log_videos = metrics["update_count"] % vid_frequency == 0

        def _compute_eval_learnability(dones, rewards, infos):
            @jax.vmap
            def _single(d, r, i):
                learn, num_eps, num_succ = compute_learnability(config, d, r, i, config["num_eval_levels"])

                return num_eps, num_succ.squeeze(-1)

            num_eps, num_succ = _single(dones, rewards, infos)
            num_eps, num_succ = num_eps.sum(axis=0), num_succ.sum(axis=0)
            success_rate = num_succ / jnp.maximum(1, num_eps)

            return success_rate * (1 - success_rate)

        @jax.jit
        def _get_eval(rng):
            metrics = {}
            rng, rng_eval = jax.random.split(rng)
            (states, cum_rewards, done_idx, episode_lengths, eval_infos), (eval_dones, eval_rewards) = jax.vmap(
                eval, (0, None)
            )(jax.random.split(rng_eval, config["eval_num_attempts"]), train_state)

            # learnability here of the holdout set:
            eval_learn = _compute_eval_learnability(eval_dones, eval_rewards, eval_infos)
            # Collect Metrics
            eval_returns = cum_rewards.mean(axis=0)  # (num_eval_levels,)
            eval_solves = (eval_infos["returned_episode_solved"] * eval_dones).sum(axis=1) / jnp.maximum(
                1, eval_dones.sum(axis=1)
            )
            eval_solves = eval_solves.mean(axis=0)
            metrics["eval_returns"] = eval_returns
            metrics["eval_ep_lengths"] = episode_lengths.mean(axis=0)
            metrics["eval_learn"] = eval_learn
            metrics["eval_solves"] = eval_solves

            metrics["eval_get_max_eplen"] = (episode_lengths == env_params.max_timesteps).mean(axis=0)
            metrics["episode_return_bigger_than_negative"] = (cum_rewards > -0.4).mean(axis=0)

            if config["EVAL_ON_SAMPLED"]:
                states_dr, cum_rewards_dr, done_idx_dr, episode_lengths_dr, infos_dr = jax.vmap(
                    eval_on_dr_levels, (0, None)
                )(jax.random.split(rng_eval, config["eval_num_attempts"]), train_state)

                eval_dr_returns = cum_rewards_dr.mean(axis=0).mean()
                eval_dr_eplen = episode_lengths_dr.mean(axis=0).mean()

                my_eval_dones = infos_dr["returned_episode"]
                eval_dr_solves = (infos_dr["returned_episode_solved"] * my_eval_dones).sum(axis=1) / jnp.maximum(
                    1, my_eval_dones.sum(axis=1)
                )

                metrics["eval_dr_returns"] = eval_dr_returns
                metrics["eval_dr_eplen"] = eval_dr_eplen
                metrics["eval_dr_solve_rates"] = eval_dr_solves
            return metrics, states, episode_lengths, cum_rewards

        @jax.jit
        def _get_videos(rng, states, episode_lengths, cum_rewards):
            metrics = {"log_videos": True}

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

            @jax.jit
            def _get_video(rollout_batch):
                states = rollout_batch[0]
                images = jax.vmap(render_fn)(states)  # dimensions are (steps, x, y, 3)
                return (
                    # jax.tree.map(lambda x: x[:].transpose(0, 2, 1, 3)[:, ::-1], images).transpose(0, 3, 1, 2),
                    images.transpose(0, 3, 1, 2),
                    # images.transpose(0, 1, 4, 2, 3),
                    rollout_batch[1][:].argmax(),
                )

            # rollouts
            metrics["dr_rollout"], metrics["dr_ep_len"] = _get_video(train_state.dr_last_rollout_batch)
            metrics["replay_rollout"], metrics["replay_ep_len"] = _get_video(train_state.replay_last_rollout_batch)
            metrics["mutation_rollout"], metrics["mutation_ep_len"] = _get_video(
                train_state.mutation_last_rollout_batch
            )

            metrics["eval_animation"] = (frames, episode_lengths)

            metrics["eval_returns_video"] = cum_rewards[0]
            metrics["eval_len_video"] = episode_lengths

            # Eval on sampled

            return metrics

        @jax.jit
        def _get_dummy_videos(rng, states, episode_lengths, cum_rewards):
            n_eval = config["num_eval_levels"]
            nsteps = env_params.max_timesteps
            nsteps2 = config["outer_rollout_steps"] * config["num_steps"]
            img_size = (
                env.static_env_params.screen_dim[0] // env.static_env_params.downscale,
                env.static_env_params.screen_dim[1] // env.static_env_params.downscale,
            )
            return {
                "log_videos": False,
                "dr_rollout": jnp.zeros((nsteps2, 3, *img_size), jnp.float32),
                "dr_ep_len": jnp.zeros((), jnp.int32),
                "replay_rollout": jnp.zeros((nsteps2, 3, *img_size), jnp.float32),
                "replay_ep_len": jnp.zeros((), jnp.int32),
                "mutation_rollout": jnp.zeros((nsteps2, 3, *img_size), jnp.float32),
                "mutation_ep_len": jnp.zeros((), jnp.int32),
                # "eval_returns": jnp.zeros((n_eval,), jnp.float32),
                # "eval_solves": jnp.zeros((n_eval,), jnp.float32),
                # "eval_learn": jnp.zeros((n_eval,), jnp.float32),
                # "eval_ep_lengths": jnp.zeros((n_eval,), jnp.int32),
                "eval_animation": (
                    jnp.zeros((nsteps, n_eval, 3, *img_size), jnp.float32),
                    jnp.zeros((n_eval,), jnp.int32),
                ),
                "eval_returns_video": jnp.zeros((n_eval,), jnp.float32),
                "eval_len_video": jnp.zeros((n_eval,), jnp.int32),
            }

        rng, rng_eval, rng_vid = jax.random.split(rng, 3)

        metrics_eval, states, episode_lengths, cum_rewards = _get_eval(rng_eval)
        metrics = {
            **metrics,
            **metrics_eval,
            **jax.lax.cond(
                should_log_videos, _get_videos, _get_dummy_videos, rng_vid, states, episode_lengths, cum_rewards
            ),
        }
        max_num_images = 8

        top_regret_ones = max_num_images // 2
        bot_regret_ones = max_num_images - top_regret_ones

        @jax.jit
        def get_values(level_batch, scores):
            args = jnp.argsort(scores)  # low scores are at the start, high scores are at the end

            low_scores = args[:bot_regret_ones]
            high_scores = args[-top_regret_ones:]

            low_levels = jax.tree.map(lambda x: x[low_scores], level_batch)
            high_levels = jax.tree.map(lambda x: x[high_scores], level_batch)

            low_scores = scores[low_scores]
            high_scores = scores[high_scores]
            # now concatenate:
            return jax.vmap(render_fn)(
                jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), low_levels, high_levels)
            ), jnp.concatenate([low_scores, high_scores], axis=0)

        metrics["dr_levels"], metrics["dr_scores"] = get_values(
            train_state.dr_last_level_batch, train_state.dr_last_level_batch_scores
        )
        metrics["replay_levels"], metrics["replay_scores"] = get_values(
            train_state.replay_last_level_batch, train_state.replay_last_level_batch_scores
        )
        metrics["mutation_levels"], metrics["mutation_scores"] = get_values(
            train_state.mutation_last_level_batch, train_state.mutation_last_level_batch_scores
        )

        def _t(i):
            return jax.lax.select(i == 0, config["num_steps"], i)

        metrics["dr_ep_len"] = _t(train_state.dr_last_rollout_batch[1][:].argmax())
        metrics["replay_ep_len"] = _t(train_state.replay_last_rollout_batch[1][:].argmax())
        metrics["mutation_ep_len"] = _t(train_state.mutation_last_rollout_batch[1][:].argmax())

        highest_scoring_level = level_sampler.get_levels(train_state.sampler, train_state.sampler["scores"].argmax())
        highest_weighted_level = level_sampler.get_levels(
            train_state.sampler, level_sampler.level_weights(train_state.sampler).argmax()
        )

        metrics["highest_scoring_level"] = render_fn(highest_scoring_level)
        metrics["highest_weighted_level"] = render_fn(highest_weighted_level)

        # log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        jax.debug.callback(log_eval, metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        return (rng, train_state), {"update_count": metrics["update_count"]}

    def log_checkpoint(update_count, train_state):
        if config["save_path"] is not None and config["checkpoint_save_freq"] > 1:
            steps = (
                int(update_count)
                * int(config["num_train_envs"])
                * int(config["num_steps"])
                * int(config["outer_rollout_steps"])
            )
            # save_params_to_wandb(train_state.params, steps, config)
            save_model_to_wandb(train_state, steps, config)

    def train_eval_and_checkpoint_step(runner_state, _):
        runner_state, metrics = jax.lax.scan(
            train_and_eval_step, runner_state, xs=jnp.arange(config["checkpoint_save_freq"] // config["eval_freq"])
        )
        jax.debug.callback(log_checkpoint, metrics["update_count"][-1], runner_state[1])
        return runner_state, metrics

    # Set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    runner_state, metrics = jax.lax.scan(
        train_eval_and_checkpoint_step,
        runner_state,
        xs=jnp.arange((config["num_updates"]) // (config["checkpoint_save_freq"])),
    )

    if config["save_path"] is not None:
        # save_params_to_wandb(runner_state[1].params, config["total_timesteps"], config)
        save_model_to_wandb(runner_state[1], config["total_timesteps"], config, is_final=True)

    return runner_state[1]


if __name__ == "__main__":
    main()
