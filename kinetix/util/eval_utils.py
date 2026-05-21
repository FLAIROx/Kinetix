from typing import Callable, NamedTuple
import jax
import numpy as np
import wandb
from kinetix.environment.env import KinetixEnv
from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.environment.spaces import PixelObservations
from kinetix.util.learning import general_eval
import jax.numpy as jnp
from flax.training.train_state import TrainState


class EvalSpec(NamedTuple):
    levels_to_eval_on: EnvState  # batched env state, one entry per level
    number_of_levels: int
    level_names: list[str]

    plot_videos: bool
    num_envs_to_video: int = -1  # how many levels to render; -1 renders all


class EpisodeMetrics(NamedTuple):
    # shapes are (num_levels)
    episode_lengths: jax.Array
    episode_returns: jax.Array
    episode_solve_rates: jax.Array
    episode_solve_rates_best_of_k: jax.Array = -1.0

    @staticmethod
    def create_empty(num_levels, length_dtype=jnp.float32):
        return EpisodeMetrics(
            episode_lengths=jnp.zeros(num_levels, dtype=length_dtype),
            episode_returns=jnp.zeros(num_levels),
            episode_solve_rates=jnp.zeros(num_levels),
            episode_solve_rates_best_of_k=jnp.array(-1.0),
        )


class RenderMetrics(NamedTuple):
    # Shape is (num_timesteps, num_levels)
    states_to_plot: EnvState

    # Shapes are (num_levels)
    episode_metrics: EpisodeMetrics

    # The index of this level in the original list
    original_level_list_index: jax.Array


class EvalMetrics(NamedTuple):
    # Shapes are all (num_levels)
    episode_metrics: EpisodeMetrics

    render_metrics: RenderMetrics | None


class KinetixVideo(NamedTuple):
    is_fake: bool
    episode_metrics: EpisodeMetrics

    # Shapes are (num_timesteps, num_levels, w, h, 3)
    frames: jax.Array

    # The index of this level in the original list
    original_level_list_index: jax.Array


def make_eval_fn(
    eval_env: KinetixEnv,
    env_params: EnvParams,
    num_eval_attempts: int,
    observation_preprocessing_fn=None,
    fixed_video_rng=None,
    temperature=1.0,
) -> Callable[[jax.Array, TrainState, dict[str, EvalSpec]], dict[str, EvalMetrics]]:
    """Build a jit-able evaluation function over a named set of EvalSpecs.

    Args:
        eval_env: The environment to evaluate in.
        env_params: Environment parameters (used for max_timesteps).
        num_eval_attempts: Number of independent rollouts per level; solve rates are averaged.
        observation_preprocessing_fn: Optional fn applied to observations before the policy
            (e.g. RMS normalisation). Must close over the *current* train_state.rms_norm —
            rebuild this function each eval call, do not cache it.
        fixed_video_rng: If set, use this RNG for video level selection (so we video the same envs each time).
        temperature: Softmax temperature for action sampling during eval.

    Returns:
        eval(rng, train_state, eval_specs) -> dict mapping eval name to EvalMetrics.
    """

    def _single_eval_step(rng: jax.Array, train_state: TrainState, eval_spec: EvalSpec):
        return general_eval(
            rng,
            eval_env,
            env_params,
            train_state,
            eval_spec.levels_to_eval_on,
            env_params.max_timesteps,
            eval_spec.number_of_levels,
            keep_states=eval_spec.plot_videos,
            return_trajectories=False,
            observation_preprocessing_fn=observation_preprocessing_fn,
            temperature=temperature,
        )

    def eval(rng: jax.Array, train_state: TrainState, eval_specs: dict[str, EvalSpec]):
        values_to_return = {}
        for eval_name, eval_spec in eval_specs.items():
            rng, _rng, _video_rng = jax.random.split(rng, 3)
            (eval_states, episode_returns, done_idxs, episode_lengths, eval_infos) = jax.vmap(
                _single_eval_step, (0, None, None)
            )(jax.random.split(_rng, num_eval_attempts), train_state, eval_spec)
            episode_dones = eval_infos["returned_episode"]
            first_episode_mask = jnp.arange(env_params.max_timesteps)[None, ..., None] < episode_lengths[:, None, :]
            episode_solve_rates = (eval_infos["returned_episode_solved"] * episode_dones * first_episode_mask).sum(
                axis=1
            ) / jnp.maximum(1, (episode_dones * first_episode_mask).sum(axis=1))

            render_metrics = None
            if eval_spec.plot_videos:
                # select the first attempt
                if fixed_video_rng is not None:
                    _video_rng = fixed_video_rng
                if eval_spec.num_envs_to_video > 0:
                    # keep only a fixed number
                    indexes_to_keep = jax.random.permutation(_video_rng, eval_spec.number_of_levels)[
                        : eval_spec.num_envs_to_video
                    ]
                else:
                    indexes_to_keep = jnp.arange(eval_spec.number_of_levels)

                video_lengths, video_returns, video_solve_rates = jax.tree.map(
                    lambda x: x[0][indexes_to_keep], (done_idxs, episode_returns, episode_solve_rates)
                )

                # Shape of (timesteps, num_envs)
                video_states = jax.tree.map(lambda x: x[0, :][:, indexes_to_keep], eval_states)

                render_metrics = RenderMetrics(
                    states_to_plot=video_states.env_state,  # unwrap
                    episode_metrics=EpisodeMetrics(
                        episode_lengths=video_lengths,
                        episode_returns=video_returns,
                        episode_solve_rates=video_solve_rates,
                    ),
                    original_level_list_index=indexes_to_keep,
                )

            values_to_return[eval_name] = EvalMetrics(
                episode_metrics=EpisodeMetrics(
                    episode_returns=episode_returns.mean(axis=0),
                    episode_lengths=episode_lengths.mean(axis=0),
                    episode_solve_rates=episode_solve_rates.mean(axis=0),
                    episode_solve_rates_best_of_k=(episode_solve_rates > 0).any(axis=0).mean(),
                ),
                render_metrics=render_metrics,
            )
        return values_to_return

    return eval


def make_fake_video(env_params: EnvParams, static_env_params: StaticEnvParams, num_levels: int) -> KinetixVideo:
    return KinetixVideo(
        is_fake=True,
        episode_metrics=EpisodeMetrics.create_empty(num_levels, length_dtype=jnp.int32),
        frames=jnp.zeros(
            (
                env_params.max_timesteps,
                num_levels,
                *PixelObservations(env_params, static_env_params).observation_space(env_params).shape,
            )
        ),
        original_level_list_index=jnp.zeros(num_levels, dtype=jnp.int32),
    )


def make_video_fn(
    env_params: EnvParams, static_env_params: StaticEnvParams, render_fn: Callable[[EnvState], jax.Array]
) -> Callable[[bool, dict[str, EvalMetrics]], dict[str, KinetixVideo]]:
    def maybe_create_videos(should_log_videos: bool, eval_metrics: dict[str, EvalMetrics]) -> dict[str, KinetixVideo]:
        def _fake_video(render_metrics: RenderMetrics) -> KinetixVideo:
            return make_fake_video(env_params, static_env_params, render_metrics.original_level_list_index.shape[0])

        def _real_video(render_metrics: RenderMetrics) -> KinetixVideo:
            frames = jax.vmap(jax.vmap(render_fn))(render_metrics.states_to_plot)
            return KinetixVideo(
                is_fake=False,
                episode_metrics=render_metrics.episode_metrics,
                frames=frames,
                original_level_list_index=render_metrics.original_level_list_index,
            )

        videos = {}

        for eval_name, metric in eval_metrics.items():
            if metric.render_metrics is not None:

                videos[eval_name] = jax.lax.cond(should_log_videos, _real_video, _fake_video, metric.render_metrics)
        return videos

    return maybe_create_videos


def create_eval_metrics_dict_for_logging(
    eval_specs: dict[str, EvalSpec],
    eval_metrics: dict[str, EpisodeMetrics] | None,
    videos: dict[str, KinetixVideo] | None,
):
    dict_of_all_logs = {}

    # Normal eval metrics
    if eval_metrics is not None:
        for eval_name, metric in eval_metrics.items():
            dict_of_all_logs[f"eval/{eval_name}_episode_return"] = metric.episode_returns.mean()
            dict_of_all_logs[f"eval/{eval_name}_episode_length"] = metric.episode_lengths.mean()
            dict_of_all_logs[f"eval/{eval_name}_episode_solve_rate"] = metric.episode_solve_rates.mean()

            if metric.episode_solve_rates_best_of_k != -1.0:
                dict_of_all_logs[f"eval/{eval_name}_episode_best_of_k_solve_rate"] = jnp.mean(
                    metric.episode_solve_rates_best_of_k
                )

            # per level scores
            assert metric.episode_returns.shape[0] == len(
                eval_specs[eval_name].level_names
            ), f"Mismatched Shape. Expected {len(eval_specs[eval_name].level_names)}, got {metric.episode_returns.shape[0]}"
            if eval_name == "hand_designed":
                for i, level_name in enumerate(eval_specs[eval_name].level_names):
                    dict_of_all_logs[f"eval_avg/{eval_name}_episode_return_{level_name}"] = metric.episode_returns[i]
                    dict_of_all_logs[
                        f"eval_avg/{eval_name}_episode_solve_rate_{level_name}"
                    ] = metric.episode_solve_rates[i]
                    dict_of_all_logs[f"eval_avg/{eval_name}_episode_length_{level_name}"] = metric.episode_lengths[i]

    # And, optionally, videos
    if videos is not None:
        for eval_name, video in videos.items():
            for i, index_into_level_names_list in enumerate(video.original_level_list_index):
                length = video.episode_metrics.episode_lengths[i].astype(jnp.int32)
                frames = video.frames[:length, i]

                eval_env_name = eval_specs[eval_name].level_names[index_into_level_names_list]

                caption = f"R = {video.episode_metrics.episode_returns[i]:.2f} | L = {video.episode_metrics.episode_lengths[i]} | S = {video.episode_metrics.episode_solve_rates[i] > 0}"

                np_vid = np.asarray(frames).transpose(0, 3, 2, 1)[:, :, ::-1, :]
                dict_of_all_logs[f"media/eval_video_{eval_env_name}"] = wandb.Video(
                    (np_vid * 255).astype(np.uint8), fps=15, format="gif", caption=caption
                )

                dict_of_all_logs[
                    f"eval_video/{eval_name}_episode_return_{eval_env_name}"
                ] = video.episode_metrics.episode_returns[i]
                dict_of_all_logs[
                    f"eval_video/{eval_name}_episode_length_{eval_env_name}"
                ] = video.episode_metrics.episode_lengths[i]
                dict_of_all_logs[
                    f"eval_video/{eval_name}_episode_solve_rate_{eval_env_name}"
                ] = video.episode_metrics.episode_solve_rates[i]

    return dict_of_all_logs
