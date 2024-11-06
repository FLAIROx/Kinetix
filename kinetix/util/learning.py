from functools import partial
import json
import os
import re
import time
from enum import IntEnum
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState

import wandb
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.level_sampler import LevelSampler
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss

from kinetix.environment.env import PixelObservations, make_kinetix_env_from_name
from kinetix.environment.env_state import StaticEnvParams
from kinetix.environment.utils import permute_pcg_state
from kinetix.environment.wrappers import (
    UnderspecifiedToGymnaxWrapper,
    LogWrapper,
    DenseRewardWrapper,
    AutoReplayWrapper,
)
from kinetix.models import make_network_from_config
from kinetix.pcg.pcg import env_state_to_pcg_state
from kinetix.render.renderer_pixels import make_render_pixels
from kinetix.models.actor_critic import ScannedRNN
from kinetix.util.saving import (
    expand_pcg_state,
    get_pcg_state_from_json,
    load_pcg_state_pickle,
    load_world_state_pickle,
    stack_list_of_pytrees,
    import_env_state_from_json,
    load_from_json_file,
)
from flax.training.train_state import TrainState

BASE_DIR = "worlds"

DEFAULT_EVAL_LEVELS = [
    "easy.cartpole",
    "easy.flappy_bird",
    "easy.unicycle",
    "easy.car_left",
    "easy.car_right",
    "easy.pinball",
    "easy.swing_up",
    "easy.thruster",
]


def get_eval_levels(eval_levels, static_env_params):
    should_permute = [".permute" in l for l in eval_levels]
    eval_levels = [re.sub(r"\.permute\d+", "", l) for l in eval_levels]
    ls = [get_pcg_state_from_json(os.path.join(BASE_DIR, l + ("" if l.endswith(".json") else ".json"))) for l in eval_levels]
    ls = [expand_pcg_state(l, static_env_params) for l in ls]
    new_ls = []
    rng = jax.random.PRNGKey(0)
    for sp, l in zip(should_permute, ls):
        rng, _rng = jax.random.split(rng)
        if sp:
            l = permute_pcg_state(_rng, l, static_env_params)
        new_ls.append(l)
    return stack_list_of_pytrees(new_ls)


def evaluate_rnn(  # from jaxued
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
    keep_states=True,
    return_trajectories=False,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
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
            return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward, done, info)
        else:
            return (rng, hstate, obs, next_state, done, next_mask, episode_length), (None, reward, done, info)

    (_, _, _, _, _, _, episode_lengths), (states, rewards, dones, infos) = jax.lax.scan(
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
    done_idx = jnp.argmax(dones, axis=0)

    to_return = (states, rewards, done_idx, episode_lengths, infos)
    if return_trajectories:
        return to_return, (dones, rewards)
    return to_return


def general_eval(
    rng: chex.PRNGKey,
    eval_env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    levels: EnvState,
    num_eval_steps: int,
    num_levels: int,
    keep_states=True,
    return_trajectories=False,
):
    """
    This evaluates the current policy on the set of evaluation levels
    It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
    """
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
        jax.random.split(rng_reset, num_levels), levels, env_params
    )
    init_hstate = ScannedRNN.initialize_carry(num_levels)
    (states, rewards, done_idx, episode_lengths, infos), (dones, reward) = evaluate_rnn(
        rng,
        eval_env,
        env_params,
        train_state,
        init_hstate,
        init_obs,
        init_env_state,
        num_eval_steps,
        keep_states=keep_states,
        return_trajectories=True,
    )
    mask = jnp.arange(num_eval_steps)[..., None] < episode_lengths
    cum_rewards = (rewards * mask).sum(axis=0)
    to_return = (
        states,
        cum_rewards,
        done_idx,
        episode_lengths,
        infos,
    )  # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)

    if return_trajectories:
        return to_return, (dones, reward)
    return to_return


def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float):
        lambd (float):
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """

    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
    return_states: bool = False,
) -> Tuple[
    Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array],
    Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict],
]:
    """This samples trajectories from the environment using the agent specified by the `train_state`.

    Args:

        rng (chex.PRNGKey): Singleton
        env (UnderspecifiedEnv):
        env_params (EnvParams):
        train_state (TrainState): Singleton
        init_hstate (chex.ArrayTree): This is the init RNN hidden state, has to have shape (NUM_ENVS, ...)
        init_obs (Observation): The initial observation, shape (NUM_ENVS, ...)
        init_env_state (EnvState): The initial env state (NUM_ENVS, ...)
        num_envs (int): The number of envs that are vmapped over.
        max_episode_length (int): The maximum episode length, i.e., the number of steps to do the rollouts for.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]: (rng, train_state, hstate, last_obs, last_env_state, last_value), traj, where traj is (obs, action, reward, done, log_prob, value, info). The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...) (except `rng` and and `train_state` which are singleton). The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.
    """

    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        prev_state = env_state
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree.map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, hstate, x)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = jax.tree.map(lambda x: x.squeeze(0), (value, action, log_prob))

        next_obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(rng_step, num_envs), env_state, action, env_params
        )

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        step = (obs, action, reward, done, log_prob, value, info)
        if return_states:
            step += (prev_state,)
        return carry, step

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    x = jax.tree.map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, hstate, x)

    my_obs = traj[0]
    rew = traj[2]

    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj


def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool = True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey):
        train_state (TrainState):
        init_hstate (chex.ArrayTree):
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int):
        n_steps (int):
        n_minibatch (int):
        n_epochs (int):
        clip_eps (float):
        entropy_coeff (float):
        critic_coeff (float):
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch

            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, init_hstate, (obs, last_dones))
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            grad_norm = jnp.linalg.norm(
                jnp.concatenate(jax.tree_util.tree_map(lambda x: x.flatten(), jax.tree_util.tree_flatten(grads)[0]))
            )
            return train_state, (loss, grad_norm)

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0).reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, (losses, grads) = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), (losses, grads)

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)


@partial(jax.jit, static_argnums=(0, 2, 8, 9))
def sample_trajectories_and_learn(
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    config: dict,
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.Array,
    init_obs: Observation,
    init_env_state: EnvState,
    update_grad: bool = True,
    return_states: bool = False,
) -> Tuple[
    Tuple[chex.PRNGKey, TrainState, Observation, EnvState],
    Tuple[
        Observation,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        dict,
        chex.Array,
        chex.Array,
        chex.ArrayTree,
        chex.Array,
    ],
]:
    """This function loops the following:
        - rollout for config['num_steps']
        - learn / update policy

    And it loops it for config['outer_rollout_steps'].
    What is returns is a new carry (rng, train_state, init_obs, init_env_state), and concatenated rollouts. The shape of the rollouts are config['num_steps'] * config['outer_rollout_steps']. In other words, the trajectories returned by this function are the same as if we ran rollouts for config['num_steps'] * config['outer_rollout_steps'] steps, but the agent does perform PPO updates in between.

    Args:
        env (UnderspecifiedEnv):
        env_params (EnvParams):
        config (dict):
        rng (chex.PRNGKey):
        train_state (TrainState):
        init_obs (Observation):
        init_env_state (EnvState):
        update_grad (bool, optional): Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, Observation, EnvState], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict, chex.Array, chex.Array, chex.ArrayTree, chex.Array]]: This returns a tuple:
        (
            (rng, train_state, init_obs, init_env_state),
            (obs, actions, rewards, dones, log_probs, values, info, advantages, targets, losses, grads)
        )
    """

    def single_step(carry, _):
        rng, train_state, init_hstate, init_obs, init_env_state = carry
        ((rng, train_state, new_hstate, last_obs, last_env_state, last_value), traj,) = sample_trajectories_rnn(
            rng,
            env,
            env_params,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            config["num_train_envs"],
            config["num_steps"],
            return_states=return_states,
        )
        if return_states:
            states = traj[-1]
            traj = traj[:-1]

        (obs, actions, rewards, dones, log_probs, values, info) = traj
        advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)

        # Update the policy using trajectories collected from replay levels
        (rng, train_state), (losses, grads) = update_actor_critic_rnn(
            rng,
            train_state,
            init_hstate,
            (obs, actions, dones, log_probs, values, targets, advantages),
            config["num_train_envs"],
            config["num_steps"],
            config["num_minibatches"],
            config["update_epochs"],
            config["clip_eps"],
            config["ent_coef"],
            config["vf_coef"],
            update_grad=update_grad,
        )
        new_carry = (rng, train_state, new_hstate, last_obs, last_env_state)
        step = (obs, actions, rewards, dones, log_probs, values, info, advantages, targets, losses, grads)
        if return_states:
            step += (states,)
        return new_carry, step

    carry = (rng, train_state, init_hstate, init_obs, init_env_state)
    new_carry, all_rollouts = jax.lax.scan(single_step, carry, None, length=config["outer_rollout_steps"])

    all_rollouts = jax.tree_util.tree_map(lambda x: jnp.concatenate(x, axis=0), all_rollouts)
    return new_carry, all_rollouts


def no_op_rollout(
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    rng: chex.PRNGKey,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
    do_random=False,
):

    noop = jnp.array(env.action_type.noop_action())
    zero_action = jnp.repeat(noop[None, ...], num_envs, axis=0)
    SHAPE = zero_action.shape

    def sample_step(carry, _):
        rng, obs, env_state, last_done = carry
        rng, rng_step, _rng = jax.random.split(rng, 3)
        if do_random:
            action = jax.vmap(env.action_space(env_params).sample)(jax.random.split(_rng, num_envs))
        else:
            action = zero_action

        next_obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(rng_step, num_envs), env_state, action, env_params
        )

        carry = (rng, next_obs, env_state, done)
        return carry, (obs, action, reward, done, info)

    (rng, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            init_obs,
            init_env_state,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    info = traj[-1]
    dones = traj[-2]

    returns_per_env = (info["returned_episode_returns"] * dones).sum(axis=0) / jnp.maximum(1, dones.sum(axis=0))
    lens_per_env = (info["returned_episode_lengths"] * dones).sum(axis=0) / jnp.maximum(1, dones.sum(axis=0))
    success_per_env = (info["returned_episode_solved"] * dones).sum(axis=0) / jnp.maximum(1, dones.sum(axis=0))
    return returns_per_env, lens_per_env, success_per_env


def no_op_and_random_rollout(
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    rng: chex.PRNGKey,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
):
    returns_noop, lens_noop, success_noop = no_op_rollout(
        env, env_params, rng, init_obs, init_env_state, num_envs, max_episode_length, do_random=False
    )
    returns_random, lens_random, success_random = no_op_rollout(
        env, env_params, rng, init_obs, init_env_state, num_envs, max_episode_length, do_random=True
    )
    return returns_noop, lens_noop, success_noop, returns_random, lens_random, success_random
