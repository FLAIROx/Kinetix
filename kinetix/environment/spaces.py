from abc import ABC
from enum import Enum
from typing import Optional, Union
from chex._src.pytypes import PRNGKey
from gymnax.environments import spaces
from gymnax.environments.spaces import Space
from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.environment.utils import (
    convert_continuous_actions,
    convert_discrete_actions,
    convert_multi_discrete_actions,
)


import chex
import jax
import jax.numpy as jnp
import numpy as np

from kinetix.render.renderer_pixels import make_render_pixels_rl
from kinetix.render.renderer_symbolic_entity import make_render_entities
from kinetix.render.renderer_symbolic_flat import make_render_symbolic


class ActionType(Enum):
    CONTINUOUS = 0
    DISCRETE = 1
    MULTI_DISCRETE = 2

    @staticmethod
    def from_string(s: str):
        return {
            "continuous": ActionType.CONTINUOUS,
            "discrete": ActionType.DISCRETE,
            "multi_discrete": ActionType.MULTI_DISCRETE,
        }[s]


class ObservationType(Enum):
    PIXELS = 0
    SYMBOLIC_FLAT = 1
    SYMBOLIC_ENTITY = 2
    BLIND = 3
    SYMBOLIC_FLAT_PADDED = 4

    @staticmethod
    def from_string(s: str):
        return {
            "pixels": ObservationType.PIXELS,
            "symbolic_flat": ObservationType.SYMBOLIC_FLAT,
            "symbolic_entity": ObservationType.SYMBOLIC_ENTITY,
            "blind": ObservationType.BLIND,
            "symbolic_flat_padded": ObservationType.SYMBOLIC_FLAT_PADDED,
        }[s]


class MultiDiscrete(Space):
    def __init__(self, n, number_of_dims_per_distribution):
        self.number_of_dims_per_distribution = number_of_dims_per_distribution
        self.n = n
        self.shape = (number_of_dims_per_distribution.shape[0],)
        self.dtype = jnp.int_

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        uniform_sample = jax.random.uniform(rng, shape=self.shape) * self.number_of_dims_per_distribution
        md_dist = jnp.floor(uniform_sample)
        return md_dist.astype(self.dtype)

    def contains(self, x) -> jnp.ndarray:
        """Check whether specific object is within space."""
        range_cond = jnp.logical_and(x >= 0, (x < self.number_of_dims_per_distribution).all())
        return range_cond


class KinetixAction:
    def __init__(self, env_params: EnvParams, static_env_params: StaticEnvParams):
        # This is the processed, unified action space size that is shared with all action types
        # 1 dim per motor and thruster
        self.unified_action_space_size = static_env_params.num_motor_bindings + static_env_params.num_thruster_bindings

    def action_space(self, env_params: Optional[EnvParams] = None) -> Union[spaces.Discrete, spaces.Box]:
        raise NotImplementedError()

    def process_action(self, action: jnp.ndarray, state: EnvState, static_env_params: StaticEnvParams) -> jnp.ndarray:
        raise NotImplementedError()

    def noop_action(self) -> jnp.ndarray:
        raise NotImplementedError()

    def random_action(self, rng: chex.PRNGKey):
        raise NotImplementedError()


class MultiDiscreteActions(KinetixAction):
    def __init__(self, env_params: EnvParams, static_env_params: StaticEnvParams):
        super().__init__(env_params, static_env_params)

        self.env_params = env_params
        self.static_env_params = static_env_params
        # This is the action space that will be used internally by an agent
        # 3 dims per motor (foward, backward, off) and 2 per thruster (on, off)
        self.n_hot_action_space_size = (
            self.static_env_params.num_motor_bindings * 3 + self.static_env_params.num_thruster_bindings * 2
        )

        def _make_sample_random():
            minval = jnp.zeros(self.unified_action_space_size, dtype=jnp.int32)
            maxval = jnp.ones(self.unified_action_space_size, dtype=jnp.int32) * 3
            maxval = maxval.at[self.static_env_params.num_motor_bindings :].set(2)

            def random(rng):
                return jax.random.randint(rng, shape=(self.unified_action_space_size,), minval=minval, maxval=maxval)

            return random

        self._random = _make_sample_random

        self.number_of_dims_per_distribution = jnp.concatenate(
            [
                np.ones(self.static_env_params.num_motor_bindings) * 3,
                np.ones(self.static_env_params.num_thruster_bindings) * 2,
            ]
        ).astype(np.int32)

    def action_space(self, env_params: Optional[EnvParams] = None) -> MultiDiscrete:
        return MultiDiscrete(self.n_hot_action_space_size, self.number_of_dims_per_distribution)

    def process_action(self, action: jnp.ndarray, state: EnvState, static_env_params: StaticEnvParams) -> jnp.ndarray:
        return convert_multi_discrete_actions(action, state, static_env_params, self.env_params)

    def noop_action(self):
        return jnp.zeros(self.unified_action_space_size, dtype=jnp.int32)

    def random_action(self, rng: chex.PRNGKey):
        return self._random()(rng)


class DiscreteActions(KinetixAction):
    def __init__(self, env_params: EnvParams, static_env_params: StaticEnvParams):
        super().__init__(env_params, static_env_params)

        self.env_params = env_params
        self.static_env_params = static_env_params

        self._n_actions = (
            self.static_env_params.num_motor_bindings * 2 + 1 + self.static_env_params.num_thruster_bindings
        )

    def action_space(self, env_params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(self._n_actions)

    def process_action(self, action: jnp.ndarray, state: EnvState, static_env_params: StaticEnvParams) -> jnp.ndarray:
        return convert_discrete_actions(action, state, static_env_params, self.env_params)

    def noop_action(self) -> int:
        return self.static_env_params.num_motor_bindings * 2

    def random_action(self, rng: chex.PRNGKey):
        return jax.random.randint(rng, shape=(), minval=0, maxval=self._n_actions)


class ContinuousActions(KinetixAction):
    def __init__(self, env_params: EnvParams, static_env_params: StaticEnvParams):
        super().__init__(env_params, static_env_params)

        self.env_params = env_params
        self.static_env_params = static_env_params

    def action_space(self, env_params: EnvParams | None = None) -> spaces.Discrete | spaces.Box:
        return spaces.Box(
            low=jnp.ones(self.unified_action_space_size) * -1.0,
            high=jnp.ones(self.unified_action_space_size) * 1.0,
            shape=(self.unified_action_space_size,),
        )

    def process_action(self, action: PRNGKey, state: EnvState, static_env_params: StaticEnvParams) -> PRNGKey:
        return convert_continuous_actions(action, state, static_env_params, self.env_params)

    def noop_action(self) -> jnp.ndarray:
        return jnp.zeros(self.unified_action_space_size, dtype=jnp.float32)

    def random_action(self, rng: chex.PRNGKey) -> jnp.ndarray:
        actions = jax.random.uniform(rng, shape=(self.unified_action_space_size,), minval=-1.0, maxval=1.0)
        # Motors between -1 and 1, thrusters between 0 and 1
        actions = actions.at[self.static_env_params.num_motor_bindings :].set(
            jnp.abs(actions[self.static_env_params.num_motor_bindings :])
        )

        return actions


class KinetixObservation(ABC):
    def __init__(self, env_params: EnvParams, static_env_params: StaticEnvParams):
        self.env_params = env_params
        self.static_env_params = static_env_params

    def get_obs(self, state: EnvState):
        raise NotImplementedError()

    def observation_space(self, env_params: EnvParams):
        raise NotImplementedError()


class SymbolicPaddedObservations(KinetixObservation):
    def __init__(
        self,
        env_params: EnvParams,
        static_env_params: StaticEnvParams,
    ):
        super().__init__(env_params, static_env_params)
        self.render_function = make_render_symbolic(env_params, static_env_params, True)

    def get_obs(self, state: EnvState):
        return self.render_function(state)


class BlindObservations(KinetixObservation):
    def __init__(self, env_params: EnvParams, static_env_params: StaticEnvParams):
        super().__init__(env_params, static_env_params)

    def get_obs(self, state: EnvState):
        return jax.nn.one_hot(state.timestep, self.env_params.max_timesteps + 1)


class EntityObservations(KinetixObservation):
    def __init__(self, env_params: EnvParams, static_env_params: StaticEnvParams, ignore_mask: bool = False):
        super().__init__(env_params, static_env_params)
        self.render_function = make_render_entities(env_params, static_env_params, ignore_attention_mask=ignore_mask)

    def get_obs(self, state: EnvState):
        return self.render_function(state)

    def observation_space(self, env_params: EnvParams) -> spaces.Dict:
        n_shapes = self.static_env_params.num_polygons + self.static_env_params.num_circles

        def _box(*shape, dtype=jnp.float32, low=-np.inf, high=np.inf):

            return spaces.Box(
                low,
                high,
                shape,
                dtype=dtype,
            )

        return spaces.Dict(
            dict(
                circles=_box(self.static_env_params.num_circles, 19),
                polygons=_box(self.static_env_params.num_polygons, 27),
                joints=_box(self.static_env_params.num_joints * 2, 22),
                thrusters=_box(self.static_env_params.num_thrusters, 8),
                circle_mask=_box(self.static_env_params.num_circles, dtype=bool, low=0, high=1),
                polygon_mask=_box(self.static_env_params.num_polygons, dtype=bool, low=0, high=1),
                joint_mask=_box(self.static_env_params.num_joints * 2, dtype=bool, low=0, high=1),
                thruster_mask=_box(self.static_env_params.num_thrusters, dtype=bool, low=0, high=1),
                attention_mask=_box(4, n_shapes, n_shapes, dtype=bool, low=0, high=1),
                joint_indexes=_box(self.static_env_params.num_joints * 2, 2, dtype=jnp.int32, low=0, high=n_shapes - 1),
                thruster_indexes=_box(self.static_env_params.num_thrusters, dtype=jnp.int32, low=0, high=n_shapes - 1),
            )
        )


class SymbolicObservations(KinetixObservation):
    def __init__(self, env_params: EnvParams, static_env_params: StaticEnvParams):
        super().__init__(env_params, static_env_params)
        self.render_function = make_render_symbolic(env_params, static_env_params)

    def get_obs(self, state: EnvState):
        return self.render_function(state)

    def observation_space(self, env_params: EnvParams) -> spaces.Box:
        n_shapes = self.static_env_params.num_polygons + self.static_env_params.num_circles
        n_features = (
            (self.static_env_params.num_polygons - 3) * 26
            + self.static_env_params.num_circles * 18
            + self.static_env_params.num_joints * (22 + n_shapes * 2)
            + self.static_env_params.num_thrusters * (8 + n_shapes)
            + 1
        )
        return spaces.Box(
            -np.inf,
            np.inf,
            (n_features,),
            dtype=jnp.float32,
        )


class PixelObservations(KinetixObservation):
    def __init__(self, env_params: EnvParams, static_env_params: StaticEnvParams):
        super().__init__(env_params, static_env_params)
        self.render_function = make_render_pixels_rl(env_params, static_env_params)

    def get_obs(self, state: EnvState):
        return self.render_function(state)

    def observation_space(self, env_params: EnvParams) -> spaces.Box:
        return spaces.Box(
            0.0,
            1.0,
            tuple(a // self.static_env_params.downscale for a in self.static_env_params.screen_dim) + (3,),
            dtype=jnp.float32,
        )
