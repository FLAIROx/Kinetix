from abc import ABC
from enum import Enum
from functools import partial
from typing import Optional, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex._src.pytypes import PRNGKey
from gymnax.environments import spaces
from gymnax.environments.spaces import Space
from jax2d.engine import calculate_collision_matrix, create_empty_sim
from jax2d.sim_state import SimState

from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.render.renderer_pixels import make_render_pixels_rl
from kinetix.render.renderer_symbolic_entity import make_render_entities
from kinetix.render.renderer_symbolic_flat import make_render_symbolic


@partial(jax.jit, static_argnums=(0,))
def create_empty_env(static_env_params):
    sim_state = create_empty_sim(static_env_params)
    return EnvState(
        timestep=0,
        last_distance=-1.0,
        thruster_bindings=jnp.zeros(static_env_params.num_thrusters, dtype=jnp.int32),
        motor_bindings=jnp.zeros(static_env_params.num_joints, dtype=jnp.int32),
        motor_auto=jnp.zeros(static_env_params.num_joints, dtype=bool),
        polygon_shape_roles=jnp.zeros(static_env_params.num_polygons, dtype=jnp.int32),
        circle_shape_roles=jnp.zeros(static_env_params.num_circles, dtype=jnp.int32),
        polygon_highlighted=jnp.zeros(static_env_params.num_polygons, dtype=bool),
        circle_highlighted=jnp.zeros(static_env_params.num_circles, dtype=bool),
        polygon_densities=jnp.ones(static_env_params.num_polygons, dtype=jnp.float32),
        circle_densities=jnp.ones(static_env_params.num_circles, dtype=jnp.float32),
        **sim_state.__dict__,
    )


@jax.jit
def index_motor_actions(
    action: jnp.ndarray,
    state: EnvState,
    clip_min=None,
    clip_max=None,
):
    # Expand the motor actions to all joints with the same colour
    return jnp.clip(action[state.motor_bindings], clip_min, clip_max)


@jax.jit
def index_thruster_actions(
    action: jnp.ndarray,
    state: EnvState,
    clip_min=None,
    clip_max=None,
):
    # Expand the thruster actions to all joints with the same colour
    return jnp.clip(action[state.thruster_bindings], clip_min, clip_max)


@partial(jax.jit, static_argnums=(2,))
def convert_continuous_actions(
    action: jnp.ndarray, state: SimState, static_env_params: StaticEnvParams, env_params: EnvParams
):
    action_motor = action[: static_env_params.num_motor_bindings]
    action_thruster = action[static_env_params.num_motor_bindings :]
    action_motor = index_motor_actions(action_motor, state, -1, 1)
    action_thruster = index_thruster_actions(action_thruster, state, 0, 1)

    action_motor = jnp.where(state.motor_auto, jnp.ones_like(action_motor), action_motor)

    action_to_perform = jnp.concatenate([action_motor, action_thruster], axis=0)
    return action_to_perform


@partial(jax.jit, static_argnums=(2,))
def convert_discrete_actions(action: int, state: SimState, static_env_params: StaticEnvParams, env_params: EnvParams):
    # so, we have
    # 0 to NJC * 2 - 1: Joint Actions
    # NJC * 2: No-op
    # NJC * 2 + 1 to NJC * 2 + 1 + NTC - 1: Thruster Actions
    # action here is a categorical action
    which_idx = action // 2
    which_dir = action % 2
    actions = (
        jnp.zeros(static_env_params.num_motor_bindings + static_env_params.num_thruster_bindings)
        .at[which_idx]
        .set(which_dir * 2 - 1)
    )
    actions = actions * (
        1 - (action >= static_env_params.num_motor_bindings * 2)
    )  # if action is the last one, set it to zero, i.e., a no-op. Alternatively, if the action is larger than NJC * 2, then it is a thruster action and we shouldn't control the joints.

    actions = jax.lax.select(
        action > static_env_params.num_motor_bindings * 2,
        actions.at[action - static_env_params.num_motor_bindings * 2 - 1 + static_env_params.num_motor_bindings].set(1),
        actions,
    )

    action_motor = index_motor_actions(actions[: static_env_params.num_motor_bindings], state, -1, 1)
    action_motor = jnp.where(state.motor_auto, jnp.ones_like(action_motor), action_motor)
    action_thruster = index_thruster_actions(actions[static_env_params.num_motor_bindings :], state, 0, 1)
    action_to_perform = jnp.concatenate([action_motor, action_thruster], axis=0)
    return action_to_perform


@partial(jax.jit, static_argnums=(2,))
def convert_multi_discrete_actions(
    action: jnp.ndarray, state: SimState, static_env_params: StaticEnvParams, env_params: EnvParams
):
    # Comes in with each action being in {0,1,2} for joints and {0,1} for thrusters
    # Convert to [-1., 1.] for joints and [0., 1.] for thrusters

    def _single_motor_action(act):
        return jax.lax.switch(
            act,
            [lambda: 0.0, lambda: 1.0, lambda: -1.0],
        )

    def _single_thruster_act(act):
        return jax.lax.select(
            act == 0,
            0.0,
            1.0,
        )

    action_motor = jax.vmap(_single_motor_action)(action[: static_env_params.num_motor_bindings])
    action_thruster = jax.vmap(_single_thruster_act)(action[static_env_params.num_motor_bindings :])

    action_motor = index_motor_actions(action_motor, state, -1, 1)
    action_thruster = index_thruster_actions(action_thruster, state, 0, 1)

    action_motor = jnp.where(state.motor_auto, jnp.ones_like(action_motor), action_motor)

    action_to_perform = jnp.concatenate([action_motor, action_thruster], axis=0)
    return action_to_perform


@partial(jax.jit, static_argnums=(2,))
def permute_state(rng: chex.PRNGKey, env_state: EnvState, static_env_params: StaticEnvParams):
    idxs_circles = jnp.arange(static_env_params.num_circles)
    idxs_polygons = jnp.arange(static_env_params.num_polygons)
    idxs_joints = jnp.arange(static_env_params.num_joints)
    idxs_thrusters = jnp.arange(static_env_params.num_thrusters)

    rng, *_rngs = jax.random.split(rng, 5)
    idxs_circles_permuted = jax.random.permutation(_rngs[0], idxs_circles, independent=True)
    idxs_polygons_permuted = idxs_polygons.at[static_env_params.num_static_fixated_polys :].set(
        jax.random.permutation(_rngs[1], idxs_polygons[static_env_params.num_static_fixated_polys :], independent=True)
    )

    idxs_joints_permuted = jax.random.permutation(_rngs[2], idxs_joints, independent=True)
    idxs_thrusters_permuted = jax.random.permutation(_rngs[3], idxs_thrusters, independent=True)

    combined = jnp.concatenate([idxs_polygons_permuted, idxs_circles_permuted + static_env_params.num_polygons])
    # Change the ordering of the shapes, and also remember to change the indices associated with the joints

    inverse_permutation = jnp.argsort(combined)

    env_state = env_state.replace(
        polygon_shape_roles=env_state.polygon_shape_roles[idxs_polygons_permuted],
        circle_shape_roles=env_state.circle_shape_roles[idxs_circles_permuted],
        polygon_highlighted=env_state.polygon_highlighted[idxs_polygons_permuted],
        circle_highlighted=env_state.circle_highlighted[idxs_circles_permuted],
        polygon_densities=env_state.polygon_densities[idxs_polygons_permuted],
        circle_densities=env_state.circle_densities[idxs_circles_permuted],
        polygon=jax.tree.map(lambda x: x[idxs_polygons_permuted], env_state.polygon),
        circle=jax.tree.map(lambda x: x[idxs_circles_permuted], env_state.circle),
        joint=env_state.joint.replace(
            a_index=inverse_permutation[env_state.joint.a_index],
            b_index=inverse_permutation[env_state.joint.b_index],
        ),
        thruster=env_state.thruster.replace(
            object_index=inverse_permutation[env_state.thruster.object_index],
        ),
    )

    # And now permute the thrusters and joints
    env_state = env_state.replace(
        thruster_bindings=env_state.thruster_bindings[idxs_thrusters_permuted],
        motor_bindings=env_state.motor_bindings[idxs_joints_permuted],
        motor_auto=env_state.motor_auto[idxs_joints_permuted],
        joint=jax.tree.map(lambda x: x[idxs_joints_permuted], env_state.joint),
        thruster=jax.tree.map(lambda x: x[idxs_thrusters_permuted], env_state.thruster),
    )
    # and collision matrix
    env_state = env_state.replace(collision_matrix=calculate_collision_matrix(static_env_params, env_state.joint))
    return env_state


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


class KinetixObservation(ABC):
    def __init__(self, env_params: EnvParams, static_env_params: StaticEnvParams):
        self.env_params = env_params
        self.static_env_params = static_env_params

    def get_obs(self, state: EnvState):
        raise NotImplementedError()

    def observation_space(self, env_params: EnvParams):
        raise NotImplementedError()


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


class BlindObservations(KinetixObservation):
    def __init__(self, env_params: EnvParams, static_env_params: StaticEnvParams):
        super().__init__(env_params, static_env_params)

    def get_obs(self, state: EnvState):
        return jax.nn.one_hot(state.timestep, self.env_params.max_timesteps + 1)


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
