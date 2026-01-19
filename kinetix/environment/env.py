from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from gymnax.environments.environment import Environment, TEnvParams
from gymnax.environments.spaces import Box, Discrete
from jax2d.engine import PhysicsEngine
from jax2d.sim_state import CollisionManifold

from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.environment.spaces import (
    ActionType,
    ContinuousActions,
    DiscreteActions,
    EntityObservations,
    KinetixAction,
    KinetixObservation,
    MultiDiscreteActions,
    PixelObservations,
    SymbolicObservations,
    SymbolicPaddedObservations,
    ObservationType,
)

from kinetix.environment.utils import create_empty_env


class KinetixEnv(Environment):
    def __init__(
        self,
        action_type: KinetixAction,
        observation_type: KinetixObservation,
        static_env_params: StaticEnvParams,
        reset_function: Callable[[chex.PRNGKey], EnvState] = None,
        physics_engine: PhysicsEngine = None,
        auto_reset: bool = True,
    ):
        super().__init__()
        self.static_env_params = static_env_params
        self.action_type = action_type
        self.observation_type = observation_type
        self.physics_engine = physics_engine or PhysicsEngine(self.static_env_params)

        self.reset_function = reset_function
        self.auto_reset = auto_reset

    # Overridden functions from Gymnax
    def step_env(self, rng, state, action: jnp.ndarray, env_params):
        action_processed = self.action_type.process_action(action, state, self.static_env_params)
        return self.engine_step(state, action_processed, env_params)

    def reset_env(self, rng, env_params, override_reset_state: EnvState):
        rng, _rng_reset = jax.random.split(rng, 2)
        if override_reset_state is not None:
            env_state = override_reset_state
        elif self.reset_function is not None:
            env_state = self.reset_function(_rng_reset)
        else:
            raise NotImplementedError("No reset function provided")

        return self.get_obs(env_state), env_state

    # Copied from gymnax so we could add kwargs
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        env_params: Optional[TEnvParams] = None,
        override_reset_state: Optional[EnvState] = None,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if env_params is None:
            env_params = self.default_params
        key, key_reset = jax.random.split(key)

        obs_st, state_st, reward, done, info = self.step_env(key, state, action, env_params)
        if self.auto_reset:
            obs_re, state_re = self.reset_env(key_reset, env_params, override_reset_state=override_reset_state)
            # Auto-reset environment based on termination
            state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
            obs = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        else:
            obs = obs_st
            state = state_st

        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
        env_params: Optional[TEnvParams] = None,
        override_reset_state: Optional[EnvState] = None,
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if env_params is None:
            env_params = self.default_params

        obs, state = self.reset_env(key, env_params, override_reset_state=override_reset_state)
        return obs, state

    # Actual Jax2D engine step
    @partial(jax.jit, static_argnums=(0,))
    def engine_step(self, env_state: EnvState, action_to_perform, env_params):
        def _single_step(env_state, unused):
            env_state, mfolds = self.physics_engine.step(
                env_state,
                env_params,
                action_to_perform,
            )

            reward, info = self.compute_reward_info(env_state, mfolds)

            done = reward != 0

            return env_state, (reward, done, info)

        env_state, (rewards, dones, infos) = jax.lax.scan(
            _single_step, env_state, xs=None, length=self.static_env_params.frame_skip
        )
        env_state = env_state.replace(timestep=env_state.timestep + 1)

        has_at_least_one_done = dones.sum() > 0
        first_done_index = dones.argmax()
        # If one of the steps led to a done, we should use the reward of the first one.
        # The rewards.max() has a problem if the first step has a reward of -1 and the second of 0.0
        reward = jax.lax.select(
            has_at_least_one_done,
            rewards[first_done_index],
            rewards.sum(),
        )
        done = has_at_least_one_done | jax.tree.reduce(
            jnp.logical_or, jax.tree.map(lambda x: jnp.isnan(x).any(), env_state), False
        )
        done |= env_state.timestep >= env_params.max_timesteps

        info = jax.tree.map(lambda x: jax.lax.select(has_at_least_one_done, x[first_done_index], x[-1]), infos)

        # dense reward
        delta_dist = (
            -(info["distance"] - env_state.last_distance) * env_params.dense_reward_scale
        )  # if distance got less, then reward is positive

        delta_dist = jnp.nan_to_num(delta_dist, nan=0.0, posinf=0.0, neginf=0.0)
        reward = reward + jax.lax.select(
            (env_state.last_distance == -1) | (env_params.dense_reward_scale == 0.0), 0.0, delta_dist
        )

        distance = jax.lax.select(done, -1.0, info["distance"])
        env_state = env_state.replace(last_distance=distance)

        return (
            jax.lax.stop_gradient(self.get_obs(env_state)),
            jax.lax.stop_gradient(env_state),
            reward,
            done,
            info,
        )

    # Information for dense rewards
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

    # Action / Observations
    def action_space(self, env_params: Optional[EnvParams] = None) -> Union[Discrete, Box]:
        return self.action_type.action_space(env_params)

    def observation_space(self, env_params: EnvParams):
        return self.observation_type.observation_space(env_params)

    def get_obs(self, state: EnvState):
        return self.observation_type.get_obs(state)

    # Default env_params
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def __hash__(self):
        return hash(
            (
                hash(self.static_env_params),
                self.action_type.__class__.__name__,
                self.observation_type.__class__.__name__,
                self.reset_function,
                self.auto_reset,
            )
        )

    def __eq__(self, value):
        return hash(self) == hash(value)


class KinetixDummyEnv(KinetixEnv):
    def step_env(self, rng, state, action: jnp.ndarray, env_params):
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            0.0,
            False,
            {"GoalR": False, "distance": 0.0},
        )

    def reset_env(self, rng, env_params, override_reset_state: EnvState):
        if override_reset_state is not None:
            env_state = override_reset_state
        elif self.reset_function is not None:
            env_state = create_empty_env(self.static_env_params)
        else:
            raise NotImplementedError("No reset function provided")

        return self.get_obs(env_state), env_state

    def __hash__(self):
        return hash(
            (
                hash(self.static_env_params),
                self.action_type.__class__.__name__,
                self.observation_type.__class__.__name__,
                self.reset_function,
                self.auto_reset,
                "dummy",
            )
        )

    def __eq__(self, value):
        return hash(self) == hash(value)


def make_kinetix_env(
    action_type: ActionType,
    observation_type: ObservationType,
    reset_fn: Optional[Callable[[chex.PRNGKey], EnvState]],
    env_params: Optional[EnvParams] = None,
    static_env_params: Optional[StaticEnvParams] = None,
    auto_reset: bool = True,
    create_dummy_env: bool = False,
) -> KinetixEnv:
    """

    Args:
        action_type (ActionType, optional): ActionType.CONTINUOUS or ActionType.MULTI_DISCRETE, overrides the option from the config if given. Defaults to None.
        observation_type (ObservationType, optional): ObservationType.PIXELS, ObservationType.SYMBOLIC_FLAT, ObservationType.SYMBOLIC_ENTITY, ObservationType.SYMBOLIC_FLAT_PADDED, overrides the option from the config if given. Defaults to None.
        reset_func (Callable[[chex.PRNGKey], EnvState], optional): If this is given, this is the function that gets called on reset to provide the starting state for the next episode. Defaults to None, in which case the environment has no auto reset behaviour
        env_params (EnvParams): EnvParams
        static_env_params (StaticEnvParams): StaticEnvParams
        auto_reset (bool): If True, the environment will automatically reset when the episode is done, in the standard gymnax way. Defaults to True.
        create_dummy_env (bool): If true, creates a dummy environment that has no-ops for the step and reset, so that it compiles significantly faster.

    Returns:
        KinetixEnv: The kinetix environment
    """

    if env_params is None:
        env_params = EnvParams()
    if static_env_params is None:
        static_env_params = StaticEnvParams()

    if action_type == ActionType.DISCRETE:
        action_type_cls = DiscreteActions
    elif action_type == ActionType.CONTINUOUS:
        action_type_cls = ContinuousActions
    elif action_type == ActionType.MULTI_DISCRETE:
        action_type_cls = MultiDiscreteActions
    else:
        raise ValueError(f"Invalid action_type '{action_type}', must be one of: [DISCRETE, CONTINUOUS, MULTI_DISCRETE]")
    action_type = action_type_cls(env_params, static_env_params)
    # observations
    if observation_type == ObservationType.PIXELS:
        obs_type_cls = PixelObservations
    elif observation_type == ObservationType.SYMBOLIC_FLAT:
        obs_type_cls = SymbolicObservations
    elif observation_type == ObservationType.SYMBOLIC_ENTITY:
        obs_type_cls = EntityObservations
    elif observation_type == ObservationType.SYMBOLIC_FLAT_PADDED:
        obs_type_cls = SymbolicPaddedObservations
    else:
        raise ValueError(
            f"Invalid observation_type '{observation_type}', must be one of: [pixels, symbolic_flat, symbolic_entity, symbolic_flat_padded]"
        )
    obs_type = obs_type_cls(env_params, static_env_params)

    env_cls = KinetixEnv
    if create_dummy_env:
        env_cls = KinetixDummyEnv
    return env_cls(
        action_type=action_type,
        observation_type=obs_type,
        static_env_params=static_env_params,
        reset_function=reset_fn,
        auto_reset=auto_reset,
    )
