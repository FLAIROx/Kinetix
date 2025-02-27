import functools
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex._src.pytypes import PRNGKey
from gymnax.environments import environment, spaces
from gymnax.environments.environment import TEnvParams, TEnvState
from gymnax.environments.spaces import Space
from jax import lax

from jax2d.engine import PhysicsEngine, create_empty_sim, recalculate_mass_and_inertia
from jax2d.sim_state import CollisionManifold, SimState
from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.environment.wrappers import (
    AutoReplayWrapper,
    AutoResetWrapper,
    UnderspecifiedToGymnaxWrapper,
    DenseRewardWrapper,
    LogWrapper,
)

from kinetix.pcg.pcg import env_state_to_pcg_state, sample_pcg_state
from kinetix.pcg.pcg_state import PCGState
from kinetix.render.renderer_symbolic_entity import make_render_entities
from kinetix.render.renderer_pixels import make_render_pixels, make_render_pixels_rl
from kinetix.render.renderer_symbolic_flat import make_render_symbolic

from kinetix.util.saving import load_pcg_state_pickle
from jaxued.environments import UnderspecifiedEnv


def create_empty_env(static_env_params):
    sim_state = create_empty_sim(static_env_params)
    return EnvState(
        timestep=0,
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


def index_motor_actions(
    action: jnp.ndarray,
    state: EnvState,
    clip_min=None,
    clip_max=None,
):
    # Expand the motor actions to all joints with the same colour
    return jnp.clip(action[state.motor_bindings], clip_min, clip_max)


def index_thruster_actions(
    action: jnp.ndarray,
    state: EnvState,
    clip_min=None,
    clip_max=None,
):
    # Expand the thruster actions to all joints with the same colour
    return jnp.clip(action[state.thruster_bindings], clip_min, clip_max)


def convert_continuous_actions(
    action: jnp.ndarray, state: SimState, static_env_params: StaticEnvParams, params: EnvParams
):
    action_motor = action[: static_env_params.num_motor_bindings]
    action_thruster = action[static_env_params.num_motor_bindings :]
    action_motor = index_motor_actions(action_motor, state, -1, 1)
    action_thruster = index_thruster_actions(action_thruster, state, 0, 1)

    action_motor = jnp.where(state.motor_auto, jnp.ones_like(action_motor), action_motor)

    action_to_perform = jnp.concatenate([action_motor, action_thruster], axis=0)
    return action_to_perform


def convert_discrete_actions(action: int, state: SimState, static_env_params: StaticEnvParams, params: EnvParams):
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


def convert_multi_discrete_actions(
    action: jnp.ndarray, state: SimState, static_env_params: StaticEnvParams, params: EnvParams
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


def make_kinetix_env_from_args(
    obs_type, action_type, reset_type, static_env_params=None, auto_reset_fn=None, dense_reward_scale=1.0
):
    if obs_type == "entity":
        if action_type == "multidiscrete":
            env = KinetixEntityMultiDiscreteActions(should_do_pcg_reset=True, static_env_params=static_env_params)
        elif action_type == "discrete":
            env = KinetixEntityDiscreteActions(should_do_pcg_reset=True, static_env_params=static_env_params)
        elif action_type == "continuous":
            env = KinetixEntityContinuousActions(should_do_pcg_reset=True, static_env_params=static_env_params)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    elif obs_type == "symbolic":
        if action_type == "multidiscrete":
            env = KinetixSymbolicMultiDiscreteActions(should_do_pcg_reset=True, static_env_params=static_env_params)
        elif action_type == "discrete":
            env = KinetixSymbolicDiscreteActions(should_do_pcg_reset=True, static_env_params=static_env_params)
        elif action_type == "continuous":
            env = KinetixSymbolicContinuousActions(should_do_pcg_reset=True, static_env_params=static_env_params)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    elif obs_type == "pixels":
        if action_type == "multidiscrete":
            env = KinetixPixelsMultiDiscreteActions(should_do_pcg_reset=True, static_env_params=static_env_params)
        elif action_type == "discrete":
            env = KinetixPixelsDiscreteActions(should_do_pcg_reset=True, static_env_params=static_env_params)
        elif action_type == "continuous":
            env = KinetixPixelsContinuousActions(should_do_pcg_reset=True, static_env_params=static_env_params)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    elif obs_type == "blind":
        if action_type == "discrete":
            env = KinetixBlindDiscreteActions(should_do_pcg_reset=True, static_env_params=static_env_params)
        elif action_type == "continuous":
            env = KinetixBlindContinuousActions(should_do_pcg_reset=True, static_env_params=static_env_params)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    else:
        raise ValueError(f"Unknown observation type: {obs_type}")

    # Wrap
    if reset_type == "replay":
        env = AutoReplayWrapper(env)
    elif reset_type == "reset":
        env = AutoResetWrapper(env, sample_level=auto_reset_fn)
    else:
        raise ValueError(f"Unknown reset type {reset_type}")

    env = UnderspecifiedToGymnaxWrapper(env)
    env = DenseRewardWrapper(env, dense_reward_scale=dense_reward_scale)
    env = LogWrapper(env)

    return env


def make_kinetix_env_from_name(name, static_env_params=None):
    kwargs = dict(filename_to_use_for_reset=None, should_do_pcg_reset=True, static_env_params=static_env_params)
    values = {
        "Kinetix-Pixels-MultiDiscrete-v1": KinetixPixelsMultiDiscreteActions,
        "Kinetix-Pixels-Discrete-v1": KinetixPixelsDiscreteActions,
        "Kinetix-Pixels-Continuous-v1": KinetixPixelsContinuousActions,
        "Kinetix-Symbolic-MultiDiscrete-v1": KinetixSymbolicMultiDiscreteActions,
        "Kinetix-Symbolic-Discrete-v1": KinetixSymbolicDiscreteActions,
        "Kinetix-Symbolic-Continuous-v1": KinetixSymbolicContinuousActions,
        "Kinetix-Blind-Discrete-v1": KinetixBlindDiscreteActions,
        "Kinetix-Blind-Continuous-v1": KinetixBlindContinuousActions,
        "Kinetix-Entity-Discrete-v1": KinetixEntityDiscreteActions,
        "Kinetix-Entity-Continuous-v1": KinetixEntityContinuousActions,
        "Kinetix-Entity-MultiDiscrete-v1": KinetixEntityMultiDiscreteActions,
    }

    return values[name](**kwargs)


class ObservationSpace:
    def __init__(self, params: EnvParams, static_env_params: StaticEnvParams):
        pass

    def get_obs(self, state: EnvState):
        raise NotImplementedError()

    def observation_space(self, params: EnvParams):
        raise NotImplementedError()


class PixelObservations(ObservationSpace):
    def __init__(self, params: EnvParams, static_env_params: StaticEnvParams):
        self.render_function = make_render_pixels_rl(params, static_env_params)
        self.static_env_params = static_env_params

    def get_obs(self, state: EnvState):
        return self.render_function(state)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            0.0,
            1.0,
            tuple(a // self.static_env_params.downscale for a in self.static_env_params.screen_dim) + (3,),
            dtype=jnp.float32,
        )


class SymbolicObservations(ObservationSpace):
    def __init__(self, params: EnvParams, static_env_params: StaticEnvParams):
        self.render_function = make_render_symbolic(params, static_env_params)
        self.static_env_params = static_env_params

    def get_obs(self, state: EnvState):
        return self.render_function(state)

    def observation_space(self, params: EnvParams) -> spaces.Box:
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


class EntityObservations(ObservationSpace):
    def __init__(self, params: EnvParams, static_env_params: StaticEnvParams):
        self.render_function = make_render_entities(params, static_env_params)
        self.static_env_params = static_env_params

    def get_obs(self, state: EnvState):
        return self.render_function(state)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        n_shapes = self.static_env_params.num_polygons + self.static_env_params.num_circles
        n_features = (
            (self.static_env_params.num_polygons - 3) * 26
            + self.static_env_params.num_circles * 18
            + self.static_env_params.num_joints * (22 + n_shapes * 2)
            + self.static_env_params.num_thrusters * (8 + n_shapes)
            + 1
        )

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


class BlindObservations(ObservationSpace):
    def __init__(self, params: EnvParams, static_env_params: StaticEnvParams):
        self.params = params

    def get_obs(self, state: EnvState):
        return jax.nn.one_hot(state.timestep, self.params.max_timesteps + 1)


def get_observation_space_from_name(name: str, params, static_env_params):
    if "Pixels" in name:
        return PixelObservations(params, static_env_params)
    elif "Symbolic" in name:
        return SymbolicObservations(params, static_env_params)
    elif "Entity" in name:
        return EntityObservations(params, static_env_params)
    if "Blind" in name:
        return BlindObservations(params, static_env_params)
    else:
        raise ValueError(f"Unknown name {name}")


class ActionType:
    def __init__(self, params: EnvParams, static_env_params: StaticEnvParams):
        # This is the processed, unified action space size that is shared with all action types
        # 1 dim per motor and thruster
        self.unified_action_space_size = static_env_params.num_motor_bindings + static_env_params.num_thruster_bindings

    def action_space(self, params: Optional[EnvParams] = None) -> Union[spaces.Discrete, spaces.Box]:
        raise NotImplementedError()

    def process_action(self, action: jnp.ndarray, state: EnvState, static_env_params: StaticEnvParams) -> jnp.ndarray:
        raise NotImplementedError()

    def noop_action(self) -> jnp.ndarray:
        raise NotImplementedError()

    def random_action(self, rng: chex.PRNGKey):
        raise NotImplementedError()


class ActionTypeContinuous(ActionType):
    def __init__(self, params: EnvParams, static_env_params: StaticEnvParams):
        super().__init__(params, static_env_params)

        self.params = params
        self.static_env_params = static_env_params

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete | spaces.Box:
        return spaces.Box(
            low=jnp.ones(self.unified_action_space_size) * -1.0,
            high=jnp.ones(self.unified_action_space_size) * 1.0,
            shape=(self.unified_action_space_size,),
        )

    def process_action(self, action: PRNGKey, state: EnvState, static_env_params: StaticEnvParams) -> PRNGKey:
        return convert_continuous_actions(action, state, static_env_params, self.params)

    def noop_action(self) -> jnp.ndarray:
        return jnp.zeros(self.unified_action_space_size, dtype=jnp.float32)

    def random_action(self, rng: chex.PRNGKey) -> jnp.ndarray:
        actions = jax.random.uniform(rng, shape=(self.unified_action_space_size,), minval=-1.0, maxval=1.0)
        # Motors between -1 and 1, thrusters between 0 and 1
        actions = actions.at[self.static_env_params.num_motor_bindings :].set(
            jnp.abs(actions[self.static_env_params.num_motor_bindings :])
        )

        return actions


class ActionTypeDiscrete(ActionType):
    def __init__(self, params: EnvParams, static_env_params: StaticEnvParams):
        super().__init__(params, static_env_params)

        self.params = params
        self.static_env_params = static_env_params

        self._n_actions = (
            self.static_env_params.num_motor_bindings * 2 + 1 + self.static_env_params.num_thruster_bindings
        )

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(self._n_actions)

    def process_action(self, action: jnp.ndarray, state: EnvState, static_env_params: StaticEnvParams) -> jnp.ndarray:
        return convert_discrete_actions(action, state, static_env_params, self.params)

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


class ActionTypeMultiDiscrete(ActionType):
    def __init__(self, params: EnvParams, static_env_params: StaticEnvParams):
        super().__init__(params, static_env_params)

        self.params = params
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

    def action_space(self, params: Optional[EnvParams] = None) -> MultiDiscrete:
        return MultiDiscrete(self.n_hot_action_space_size, self.number_of_dims_per_distribution)

    def process_action(self, action: jnp.ndarray, state: EnvState, static_env_params: StaticEnvParams) -> jnp.ndarray:
        return convert_multi_discrete_actions(action, state, static_env_params, self.params)

    def noop_action(self):
        return jnp.zeros(self.unified_action_space_size, dtype=jnp.int32)

    def random_action(self, rng: chex.PRNGKey):
        return self._random()(rng)


class BasePhysicsEnv(UnderspecifiedEnv):
    def __init__(
        self,
        action_type: ActionType,
        observation_space: ObservationSpace,
        static_env_params: StaticEnvParams,
        target_index: int = 0,
        filename_to_use_for_reset=None,  # "worlds/games/bipedal_v1",
        should_do_pcg_reset: bool = False,
    ):
        super().__init__()
        self.target_index = target_index
        self.static_env_params = static_env_params
        self.action_type = action_type
        self._observation_space = observation_space
        self.physics_engine = PhysicsEngine(self.static_env_params)
        self.should_do_pcg_reset = should_do_pcg_reset

        self.filename_to_use_for_reset = filename_to_use_for_reset
        if self.filename_to_use_for_reset is not None:
            self.reset_state = load_pcg_state_pickle(filename_to_use_for_reset)
        else:
            env_state = create_empty_env(self.static_env_params)
            self.reset_state = env_state_to_pcg_state(env_state)

    # Action / Observations
    def action_space(self, params: Optional[EnvParams] = None) -> Union[spaces.Discrete, spaces.Box]:
        return self.action_type.action_space(params)

    def observation_space(self, params: Any):
        return self._observation_space.observation_space(params)

    def get_obs(self, state: EnvState):
        return self._observation_space.get_obs(state)

    def step_env(self, rng, state, action: jnp.ndarray, params):
        action_processed = self.action_type.process_action(action, state, self.static_env_params)
        return self.engine_step(state, action_processed, params)

    def reset_env(self, rng, params):
        # Wrap in AutoResetWrapper or AutoReplayWrapper
        raise NotImplementedError()

    def reset_env_to_level(self, rng, state: EnvState, params):
        if isinstance(state, PCGState):
            return self.reset_env_to_pcg_level(rng, state, params)
        return self.get_obs(state), state

    def reset_env_to_pcg_level(self, rng, state: PCGState, params):
        env_state = sample_pcg_state(rng, state, params, self.static_env_params)
        return self.get_obs(env_state), env_state

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def compute_reward_info(
        self, state: EnvState, manifolds: tuple[CollisionManifold, CollisionManifold, CollisionManifold]
    ) -> float:
        def get_active(manifold: CollisionManifold) -> jnp.ndarray:
            return manifold.active

        def dist(a, b):
            return jnp.linalg.norm(a - b)

        @jax.vmap
        def dist_rr(idxa, idxb):
            return dist(state.polygon.position[idxa], state.polygon.position[idxb])

        @jax.vmap
        def dist_cc(idxa, idxb):
            return dist(state.circle.position[idxa], state.circle.position[idxb])

        @jax.vmap
        def dist_cr(idxa, idxb):
            return dist(state.circle.position[idxa], state.polygon.position[idxb])

        info = {
            "GoalR": False,
        }
        negative_reward = 0
        reward = 0
        distance = 0
        rs = state.polygon_shape_roles * state.polygon.active
        cs = state.circle_shape_roles * state.circle.active

        # Polygon Polygon
        r1 = rs[self.physics_engine.poly_poly_pairs[:, 0]]
        r2 = rs[self.physics_engine.poly_poly_pairs[:, 1]]
        reward += ((r1 * r2 == 2) * get_active(manifolds[0])).sum()
        negative_reward += ((r1 * r2 == 3) * get_active(manifolds[0])).sum()

        distance += (
            (r1 * r2 == 2)
            * dist_rr(self.physics_engine.poly_poly_pairs[:, 0], self.physics_engine.poly_poly_pairs[:, 1])
        ).sum()

        # Circle Polygon
        c1 = cs[self.physics_engine.circle_poly_pairs[:, 0]]
        r2 = rs[self.physics_engine.circle_poly_pairs[:, 1]]
        reward += ((c1 * r2 == 2) * get_active(manifolds[1])).sum()
        negative_reward += ((c1 * r2 == 3) * get_active(manifolds[1])).sum()

        t = dist_cr(self.physics_engine.circle_poly_pairs[:, 0], self.physics_engine.circle_poly_pairs[:, 1])
        distance += ((c1 * r2 == 2) * t).sum()

        # Circle Circle
        c1 = cs[self.physics_engine.circle_circle_pairs[:, 0]]
        c2 = cs[self.physics_engine.circle_circle_pairs[:, 1]]
        reward += ((c1 * c2 == 2) * get_active(manifolds[2])).sum()
        negative_reward += ((c1 * c2 == 3) * get_active(manifolds[2])).sum()

        distance += (
            (c1 * c2 == 2)
            * dist_cc(self.physics_engine.circle_circle_pairs[:, 0], self.physics_engine.circle_circle_pairs[:, 1])
        ).sum()

        reward = jax.lax.select(
            negative_reward > 0,
            -1.0,
            jax.lax.select(
                reward > 0,
                1.0,
                0.0,
            ),
        )

        info["GoalR"] = reward > 0
        info["distance"] = distance
        return reward, info

    @partial(jax.jit, static_argnums=(0,))
    def engine_step(self, env_state, action_to_perform, env_params):
        def _single_step(env_state, unused):
            env_state, mfolds = self.physics_engine.step(
                env_state,
                env_params,
                action_to_perform,
            )

            reward, info = self.compute_reward_info(env_state, mfolds)

            done = reward != 0

            info = {"rr_manifolds": None, "cr_manifolds": None} | info

            return env_state, (reward, done, info)

        env_state, (rewards, dones, infos) = jax.lax.scan(
            _single_step, env_state, xs=None, length=self.static_env_params.frame_skip
        )
        env_state = env_state.replace(timestep=env_state.timestep + 1)

        reward = rewards.max()
        done = dones.sum() > 0 | jax.tree.reduce(
            jnp.logical_or, jax.tree.map(lambda x: jnp.isnan(x).any(), env_state), False
        )
        done |= env_state.timestep >= env_params.max_timesteps

        info = jax.tree.map(lambda x: x[-1], infos)

        return (
            lax.stop_gradient(self.get_obs(env_state)),
            lax.stop_gradient(env_state),
            reward,
            done,
            info,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: TEnvState,
        action: Union[int, float, chex.Array],
        params: Optional[TEnvParams] = None,
    ) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        raise NotImplementedError()


class KinetixPixelsDiscreteActions(BasePhysicsEnv):
    def __init__(
        self,
        static_env_params: StaticEnvParams | None = None,
        **kwargs,
    ):

        params = self.default_params
        static_env_params = static_env_params or self.default_static_params()
        super().__init__(
            action_type=ActionTypeDiscrete(params, static_env_params),
            observation_space=PixelObservations(params, static_env_params),
            static_env_params=static_env_params,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "Kinetix-Pixels-Discrete-v1"


class KinetixPixelsContinuousActions(BasePhysicsEnv):
    def __init__(
        self,
        static_env_params: StaticEnvParams | None = None,
        **kwargs,
    ):
        params = self.default_params
        static_env_params = static_env_params or self.default_static_params()
        super().__init__(
            action_type=ActionTypeContinuous(params, static_env_params),
            observation_space=PixelObservations(params, static_env_params),
            static_env_params=static_env_params,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "Kinetix-Pixels-Continuous-v1"


class KinetixPixelsMultiDiscreteActions(BasePhysicsEnv):
    def __init__(
        self,
        static_env_params: StaticEnvParams | None = None,
        **kwargs,
    ):
        params = self.default_params
        static_env_params = static_env_params or self.default_static_params()
        super().__init__(
            action_type=ActionTypeMultiDiscrete(params, static_env_params),
            observation_space=PixelObservations(params, static_env_params),
            static_env_params=static_env_params,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "Kinetix-Pixels-MultiDiscrete-v1"


class KinetixSymbolicDiscreteActions(BasePhysicsEnv):
    def __init__(
        self,
        static_env_params: StaticEnvParams | None = None,
        **kwargs,
    ):
        params = self.default_params
        static_env_params = static_env_params or self.default_static_params()
        super().__init__(
            action_type=ActionTypeDiscrete(params, static_env_params),
            observation_space=SymbolicObservations(params, static_env_params),
            static_env_params=static_env_params,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "Kinetix-Symbolic-Discrete-v1"


class KinetixSymbolicContinuousActions(BasePhysicsEnv):
    def __init__(
        self,
        static_env_params: StaticEnvParams | None = None,
        **kwargs,
    ):
        params = self.default_params
        static_env_params = static_env_params or self.default_static_params()
        super().__init__(
            action_type=ActionTypeContinuous(params, static_env_params),
            observation_space=SymbolicObservations(params, static_env_params),
            static_env_params=static_env_params,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "Kinetix-Symbolic-Continuous-v1"


class KinetixSymbolicMultiDiscreteActions(BasePhysicsEnv):
    def __init__(
        self,
        static_env_params: StaticEnvParams | None = None,
        **kwargs,
    ):
        params = self.default_params
        static_env_params = static_env_params or self.default_static_params()
        super().__init__(
            action_type=ActionTypeMultiDiscrete(params, static_env_params),
            observation_space=SymbolicObservations(params, static_env_params),
            static_env_params=static_env_params,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "Kinetix-Symbolic-MultiDiscrete-v1"


class KinetixEntityDiscreteActions(BasePhysicsEnv):
    def __init__(
        self,
        static_env_params: StaticEnvParams | None = None,
        **kwargs,
    ):
        params = self.default_params
        static_env_params = static_env_params or self.default_static_params()
        super().__init__(
            action_type=ActionTypeDiscrete(params, static_env_params),
            observation_space=EntityObservations(params, static_env_params),
            static_env_params=static_env_params,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "Kinetix-Entity-Discrete-v1"


class KinetixEntityContinuousActions(BasePhysicsEnv):
    def __init__(
        self,
        static_env_params: StaticEnvParams | None = None,
        **kwargs,
    ):
        params = self.default_params
        static_env_params = static_env_params or self.default_static_params()
        super().__init__(
            action_type=ActionTypeContinuous(params, static_env_params),
            observation_space=EntityObservations(params, static_env_params),
            static_env_params=static_env_params,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "Kinetix-Entity-Continuous-v1"


class KinetixEntityMultiDiscreteActions(BasePhysicsEnv):
    def __init__(
        self,
        static_env_params: StaticEnvParams | None = None,
        **kwargs,
    ):
        params = self.default_params
        static_env_params = static_env_params or self.default_static_params()
        super().__init__(
            action_type=ActionTypeMultiDiscrete(params, static_env_params),
            observation_space=EntityObservations(params, static_env_params),
            static_env_params=static_env_params,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "Kinetix-Entity-MultiDiscrete-v1"


class KinetixBlindDiscreteActions(BasePhysicsEnv):
    def __init__(
        self,
        static_env_params: StaticEnvParams | None = None,
        **kwargs,
    ):
        params = self.default_params
        static_env_params = static_env_params or self.default_static_params()
        super().__init__(
            action_type=ActionTypeDiscrete(params, static_env_params),
            observation_space=BlindObservations(params, static_env_params),
            static_env_params=static_env_params,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "Kinetix-Blind-Discrete-v1"


class KinetixBlindContinuousActions(BasePhysicsEnv):
    def __init__(
        self,
        static_env_params: StaticEnvParams | None = None,
        **kwargs,
    ):
        params = self.default_params
        static_env_params = static_env_params or self.default_static_params()
        super().__init__(
            action_type=ActionTypeContinuous(params, static_env_params),
            observation_space=BlindObservations(params, static_env_params),
            static_env_params=static_env_params,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "Kinetix-Blind-Continuous-v1"
