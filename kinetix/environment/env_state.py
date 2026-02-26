import dataclasses
import jax.numpy as jnp
from flax import struct
from jax2d.sim_state import SimParams, SimState, StaticSimParams


@struct.dataclass
class EnvState(SimState):
    thruster_bindings: jnp.ndarray
    motor_bindings: jnp.ndarray
    motor_auto: jnp.ndarray

    polygon_shape_roles: jnp.ndarray
    circle_shape_roles: jnp.ndarray

    polygon_highlighted: jnp.ndarray
    circle_highlighted: jnp.ndarray

    polygon_densities: jnp.ndarray
    circle_densities: jnp.ndarray

    timestep: int = 0

    last_distance: float = -1.0


@struct.dataclass
class EnvParams(SimParams):
    max_timesteps: int = 256
    pixels_per_unit: int = 100
    dense_reward_scale: float = 0.1
    num_shape_roles: int = 4

    def __hash__(self):
        field_values = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return hash(tuple(sorted(field_values.items())))

    def __eq__(self, value):
        return super().__eq__(value)


@struct.dataclass
class StaticEnvParams(StaticSimParams):
    screen_dim: tuple[int, int] = (500, 500)
    downscale: int = 4

    frame_skip: int = 1
    max_shape_size: int = 2

    num_motor_bindings: int = 4
    num_thruster_bindings: int = 2

    def __hash__(self):
        field_values = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return hash(tuple(sorted(field_values.items())))

    def __eq__(self, value):
        return super().__eq__(value)
