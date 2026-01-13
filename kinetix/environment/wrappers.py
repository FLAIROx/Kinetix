from functools import partial
from typing import Any, Optional, Union

import chex
import jax
from flax import struct

from kinetix.environment.env_state import EnvState


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, env_params=None, override_reset_state: Optional[EnvState] = None):
        obs, env_state = self._env.reset(key, env_params, override_reset_state)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        env_params=None,
        override_reset_state: Optional[EnvState] = None,
    ):
        if isinstance(override_reset_state, LogEnvState):
            override_reset_state = override_reset_state.env_state
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, env_params, override_reset_state
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode_solved"] = info["GoalR"]
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info

    def __hash__(self):
        return hash(self._env)

    def __eq__(self, value):
        return hash(self) == hash(value)
