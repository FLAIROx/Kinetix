import jax.numpy as jnp
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
import chex


class ActionEnvStateMask(NamedTuple):
    # expert action at each timestep, shape (..., action_dim)
    action: jnp.ndarray
    # full physics environment state
    env_state: Any
    # True for valid/successful transitions (all-True in offline datasets)
    mask: jnp.ndarray
    # True for action dimensions that are active in this level (existing motors/thrusters)
    action_mask: jnp.ndarray
    # episode termination flags; None for flat batch data, present for trajectories
    done: jnp.ndarray | None = None


class TrainAndValData(NamedTuple):
    train_data: ActionEnvStateMask
    val_data: dict[str, ActionEnvStateMask]


class RunnerState(NamedTuple):
    train_state: TrainState
    update_step: int
    rng: chex.PRNGKey


class RunnerStateAndData(NamedTuple):
    runner_state: RunnerState
    batch_of_data: TrainAndValData
