import os

import hydra
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from kinetix.environment import LogWrapper
from kinetix.environment.env import make_kinetix_env
from kinetix.environment.ued.ued import make_reset_fn_from_config
from kinetix.models import ScannedRNN, make_network_from_config
from kinetix.render.renderer_pixels import make_render_pixels
from kinetix.util import (
    generate_params_from_config,
    get_eval_levels,
    load_train_state_from_wandb_artifact_path,
    normalise_config,
)
import mediapy as media

os.environ["WANDB_DISABLE_SERVICE"] = "True"


@hydra.main(version_base=None, config_path="../configs", config_name="ppo")
def main(config):
    config = normalise_config(OmegaConf.to_container(config), "PPO")
    env_params, static_env_params = generate_params_from_config(config)

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

    eval_levels = get_eval_levels(config["eval_levels"], env.static_env_params)
    # to keep the batch dimension
    NUM_ENVS_IN_PARALLEL = 1
    level_to_evaluate_on = jax.tree.map(lambda x: x[:NUM_ENVS_IN_PARALLEL], eval_levels)
    rng, _rng = jax.random.split(jax.random.PRNGKey(config["seed"]))

    network = make_network_from_config(env, env_params, config)
    obsv, env_state = jax.vmap(env.reset, (0, None, 0))(
        jax.random.split(_rng, NUM_ENVS_IN_PARALLEL), env_params, level_to_evaluate_on
    )
    dones = jnp.zeros((NUM_ENVS_IN_PARALLEL), dtype=jnp.bool_)
    rng, _rng = jax.random.split(rng)
    init_hstate = ScannedRNN.initialize_carry(NUM_ENVS_IN_PARALLEL)
    init_x = jax.tree.map(lambda x: x[None, ...], (obsv, dones))
    network_params = network.init(_rng, init_hstate, init_x)

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(config["lr"], eps=1e-5),
        ),
    )
    assert config["load_from_checkpoint"] != None, "When doing inference, we must have a checkpoint to load from"
    train_state = load_train_state_from_wandb_artifact_path(
        train_state, config["load_from_checkpoint"], load_only_params=config["load_only_params"]
    )
    # INIT ENV
    render_static_env_params = env.static_env_params.replace(downscale=4)
    pixel_renderer = jax.jit(make_render_pixels(env_params, render_static_env_params))
    pixel_render_fn = lambda x: pixel_renderer(x) / 255.0

    def _eval_step(carry, _):
        train_state, rng, obs, env_state, init_env_state, done, hstate = carry
        rng, _rng_step, _rng_action = jax.random.split(rng, 3)
        x = jax.tree.map(lambda x: x[None, ...], (obs, done))  # add the dummy time dimension for the RNN
        hstate, pi, _ = train_state.apply_fn(train_state.params, hstate, x)
        action = pi.sample(seed=_rng_action).squeeze(0)
        action = env.action_type.noop_action()[None]

        next_obs, next_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None, 0))(
            jax.random.split(_rng_step, NUM_ENVS_IN_PARALLEL),
            env_state,
            action,
            env_params,
            init_env_state,  # use this to reset to the init env state, keyword arguments are not easily vmappable
        )

        return (train_state, rng, next_obs, next_state, init_env_state, done, hstate), (obs, reward, done, env_state)

    rng, _rng = jax.random.split(rng)
    obs, env_state = jax.vmap(env.reset, (0, None, 0))(
        jax.random.split(_rng, NUM_ENVS_IN_PARALLEL), env_params, level_to_evaluate_on
    )
    init_hstate = ScannedRNN.initialize_carry(NUM_ENVS_IN_PARALLEL)

    init_carry = (
        train_state,
        rng,
        obs,
        env_state,
        env_state,
        jnp.zeros((NUM_ENVS_IN_PARALLEL), dtype=jnp.bool_),
        init_hstate,
    )

    _, (all_obs, all_rewards, all_dones, all_env_states) = jax.lax.scan(
        _eval_step, init_carry, length=env_params.max_timesteps
    )

    # get just the first level
    (all_obs, all_rewards, all_dones, all_env_states) = jax.tree.map(
        lambda x: x[:, 0], (all_obs, all_rewards, all_dones, all_env_states)
    )
    idx = all_dones.argmax() + 1
    idx = jax.lax.select(all_dones.sum() == 0, env_params.max_timesteps, idx)
    mask = jnp.arange(env_params.max_timesteps) < idx
    imgs = jax.vmap(pixel_render_fn)(all_env_states.env_state)
    print("Reward", (all_rewards * mask).sum(), "episode length", idx)

    os.makedirs("artifacts", exist_ok=True)
    with media.VideoWriter("artifacts/rollout.mp4", shape=imgs.shape[1:3], fps=30, crf=18) as video:
        for i in range(idx):
            video.add_image(imgs[i].transpose(1, 0, 2)[::-1])


if __name__ == "__main__":
    main()
