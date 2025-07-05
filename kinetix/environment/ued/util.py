import math
from functools import partial

import jax
import jax.numpy as jnp
from jax2d.engine import PhysicsEngine, calculate_collision_matrix, recalculate_mass_and_inertia, select_shape
from jax2d.sim_state import RigidBody, Thruster

from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams


def sample_dimensions(rng, static_env_params: StaticEnvParams, is_rect: bool, ued_params, max_shape_size=None):
    if max_shape_size is None:
        max_shape_size = static_env_params.max_shape_size
    # Returns (half_dimensions, radius)

    rng, _rng = jax.random.split(rng)
    # Don't want overly small shapes
    min_rect_size = 0.05
    min_circle_size = 0.1
    cap_rect = max_shape_size / 2.0 / jnp.sqrt(2.0)
    cap_circ = max_shape_size / 2.0 * ued_params.circle_max_size_coeff
    half_dimensions = (
        jax.lax.select(is_rect, jax.random.uniform(_rng, shape=(2,)), jnp.zeros(2, dtype=jnp.float32))
        * (cap_rect - min_rect_size)
        + min_rect_size
    )

    rng, _rng, __rng = jax.random.split(rng, 3)
    dim_scale = (
        jnp.ones(2)
        .at[jax.random.randint(_rng, shape=(), minval=0, maxval=2)]
        .set(
            jax.lax.select(
                jax.random.uniform(__rng) < ued_params.large_rect_dim_chance, ued_params.large_rect_dim_scale, 1.0
            )
        )
    )
    half_dimensions *= dim_scale

    vertices = jnp.array(
        [
            half_dimensions * jnp.array([1, 1]),
            half_dimensions * jnp.array([1, -1]),
            half_dimensions * jnp.array([-1, -1]),
            half_dimensions * jnp.array([-1, 1]),
        ]
    )

    rng, _rng = jax.random.split(rng)
    radius = (
        jax.lax.select(is_rect, jnp.zeros((), dtype=jnp.float32), jax.random.uniform(_rng, shape=()))
        * (cap_circ - min_circle_size)
        + min_circle_size
    )
    return vertices, half_dimensions, radius


def count_roles(state: EnvState, static_env_params: StaticEnvParams, role: int, include_static_polys=True) -> int:
    active_to_use = state.polygon.active
    if not include_static_polys:
        active_to_use = active_to_use.at[: static_env_params.num_static_fixated_polys].set(False)
    return ((state.polygon_shape_roles == role) * active_to_use).sum() + (
        (state.circle_shape_roles == role) * state.circle.active
    ).sum()


def random_position_on_triangle(rng, vertices):
    verts = vertices[:3]
    rng, _rng, _rng2 = jax.random.split(rng, 3)
    f1 = jax.random.uniform(_rng)
    f2 = jax.random.uniform(_rng2)
    # https://www.reddit.com/r/godot/comments/mqp29g/how_do_i_get_a_random_position_inside_a_collision/
    return verts[0] + jnp.sqrt(f1) * (-verts[0] + verts[1] + f2 * (verts[2] - verts[1]))


def random_position_on_rectangle(rng, vertices):
    verts = vertices[:4]
    rng, _rng, _rng2 = jax.random.split(rng, 3)
    f1 = jax.random.uniform(_rng)
    f2 = jax.random.uniform(_rng2)

    min_x, max_x = jnp.min(verts[:, 0]), jnp.max(verts[:, 0])
    min_y, max_y = jnp.min(verts[:, 1]), jnp.max(verts[:, 1])
    random_x_pos = min_x + f1 * (max_x - min_x)
    random_y_pos = min_y + f2 * (max_y - min_y)

    return jnp.array([random_x_pos, random_y_pos])


def random_position_on_polygon(rng, vertices, n_vertices, static_env_params: StaticEnvParams):
    assert static_env_params.max_polygon_vertices <= 4, "Only supports up to 4 vertices"
    return jax.lax.select(
        n_vertices <= 3, random_position_on_triangle(rng, vertices), random_position_on_rectangle(rng, vertices)
    )


def random_position_on_circle(rng, radius, on_centre_chance):
    rngs = jax.random.split(rng, 3)

    on_centre = jax.random.uniform(rngs[0]) < on_centre_chance

    local_joint_position_circle_theta = jax.random.uniform(rngs[1], shape=()) * 2 * math.pi
    local_joint_position_circle_r = jax.random.uniform(rngs[2], shape=()) * radius
    local_joint_position_circle = jnp.array(
        [
            local_joint_position_circle_r * jnp.cos(local_joint_position_circle_theta),
            local_joint_position_circle_r * jnp.sin(local_joint_position_circle_theta),
        ]
    )

    return jax.lax.select(on_centre, jnp.array([0.0, 0.0]), local_joint_position_circle)


def get_role(rng, state: EnvState, static_env_params: StaticEnvParams, initial_p=None) -> int:

    if initial_p is None:
        initial_p = jnp.array([1.0, 1.0, 1.0, 1.0])

    needs_ball = count_roles(state, static_env_params, 1) == 0
    needs_goal = count_roles(state, static_env_params, 2) == 0
    needs_lava = count_roles(state, static_env_params, 3) == 0

    # always put goal/ball first.
    prob_of_something_else = (needs_ball == 0) & (needs_goal == 0)
    p = initial_p * jnp.array(
        [prob_of_something_else, needs_ball, needs_goal, prob_of_something_else * needs_lava / 3]
    )  # This ensures we cannot more than one ball or goal.
    return jax.random.choice(rng, jnp.array([0, 1, 2, 3]), p=p)


def is_space_for_shape(state: EnvState):
    return jnp.logical_not(jnp.concatenate([state.polygon.active, state.circle.active])).sum() > 0


def is_space_for_joint(state: EnvState):
    return jnp.logical_not(state.joint.active).sum() > 0


def are_there_shapes_present(state: EnvState, static_env_params: StaticEnvParams):
    m = (
        jnp.concatenate([state.polygon.active, state.circle.active])
        .at[: static_env_params.num_static_fixated_polys]
        .set(False)
    )
    return m.sum() > 0


@partial(jax.jit, static_argnums=(2, 9))
def add_rigidbody_to_state(
    state: EnvState,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    position: jnp.ndarray,
    vertices: jnp.ndarray,
    n_vertices: int,
    radius: float,
    shape_role: int,
    density: float = 1,
    is_circle: bool = False,
):

    new_rigid_body = RigidBody(
        position=position,
        velocity=jnp.array([0.0, 0.0]),
        inverse_mass=1.0,
        inverse_inertia=1.0,
        rotation=0.0,
        angular_velocity=0.0,
        radius=radius,
        active=True,
        friction=1.0,
        vertices=vertices,
        n_vertices=n_vertices,
        collision_mode=1,
        restitution=0.0,
    )

    if is_circle:
        actives = state.circle.active
    else:
        actives = state.polygon.active

    idx = jnp.argmin(actives)

    def noop(state):
        return state

    def replace(state):
        add_func = lambda all, new: all.at[idx].set(new)
        if is_circle:
            state = state.replace(
                circle=jax.tree.map(add_func, state.circle, new_rigid_body),
                circle_densities=state.circle_densities.at[idx].set(density),
                circle_shape_roles=state.circle_shape_roles.at[idx].set(shape_role),
            )
        else:
            state = state.replace(
                polygon=jax.tree.map(add_func, state.polygon, new_rigid_body),
                polygon_densities=state.polygon_densities.at[idx].set(density),
                polygon_shape_roles=state.polygon_shape_roles.at[idx].set(shape_role),
            )

        state = state.replace(
            collision_matrix=calculate_collision_matrix(static_env_params, state.joint),
        )

        state = recalculate_mass_and_inertia(state, static_env_params, state.polygon_densities, state.circle_densities)
        return state

    return jax.lax.cond(jnp.logical_not(actives).sum() > 0, replace, noop, state)


def rectangle_vertices(half_dim):
    return jnp.array(
        [
            half_dim * jnp.array([1, 1]),
            half_dim * jnp.array([1, -1]),
            half_dim * jnp.array([-1, -1]),
            half_dim * jnp.array([-1, 1]),
        ]
    )


# More Manual Control
@partial(jax.jit, static_argnums=(2,))
def add_rectangle_to_state(
    state: EnvState,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    position: jnp.ndarray,
    width: float,
    height: float,
    shape_role: int,
    density: float = 1,
):

    return add_rigidbody_to_state(
        state,
        env_params,
        static_env_params,
        position,
        rectangle_vertices(jnp.array([width, height]) / 2),
        4,
        0.0,
        shape_role,
        density,
        is_circle=False,
    )


@partial(jax.jit, static_argnums=(2,))
def add_circle_to_state(
    state: EnvState,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    position: jnp.ndarray,
    radius: float,
    shape_role: int,
    density: float = 1,
):
    return add_rigidbody_to_state(
        state,
        env_params,
        static_env_params,
        position,
        jnp.array([0.0, 0.0]),
        0,
        radius,
        shape_role,
        density,
        is_circle=True,
    )


@partial(jax.jit, static_argnums=(2,))
def add_thruster_to_object(
    state: EnvState,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    shape_index: int,
    rotation: float,
    colour: int,
    thruster_power_multiplier: float,
):
    def dummy(state):
        return state

    def do_add(state: EnvState):
        thruster_idx = jnp.argmin(state.thruster.active)

        shape = select_shape(state, shape_index, static_env_params)

        thruster = Thruster(
            object_index=shape_index,
            active=True,
            relative_position=jnp.array([0.0, 0.0]),  # a bit of a hack but reasonable.
            rotation=rotation,
            power=1.0 / jax.lax.select(shape.inverse_mass == 0, 1.0, shape.inverse_mass) * thruster_power_multiplier,
            global_position=select_shape(state, shape_index, static_env_params).position,
        )

        state = state.replace(
            thruster=jax.tree.map(lambda y, x: y.at[thruster_idx].set(x), state.thruster, thruster),
            thruster_bindings=state.thruster_bindings.at[thruster_idx].set(colour),
        )

        return state

    return jax.lax.cond(
        (select_shape(state, shape_index, static_env_params).active)
        & (jnp.logical_not(state.thruster.active).sum() > 0),
        do_add,
        dummy,
        state,
    )


def make_velocities_zero(state: EnvState):
    def inner(state):
        return state.replace(
            polygon=state.polygon.replace(
                angular_velocity=state.polygon.angular_velocity * 0,
                velocity=state.polygon.velocity * 0,
            ),
            circle=state.circle.replace(
                angular_velocity=state.circle.angular_velocity * 0,
                velocity=state.circle.velocity * 0,
            ),
        )

    return inner(state)


def make_do_dummy_step(
    env_params: EnvParams, static_sim_params: StaticEnvParams, zero_collisions=True, zero_velocities=True
):
    env = PhysicsEngine(static_sim_params)

    @jax.jit
    def _step_fn(state):
        state, _ = env.step(
            state, env_params, jnp.zeros((static_sim_params.num_joints + static_sim_params.num_thrusters,))
        )
        return state

    def do_dummy_step(state: EnvState) -> EnvState:
        rng = jax.random.PRNGKey(0)
        og_col = state.collision_matrix
        g = state.gravity
        state = state.replace(
            collision_matrix=state.collision_matrix & (not zero_collisions), gravity=state.gravity * 0
        )
        state = _step_fn(state)
        state = state.replace(gravity=g, collision_matrix=og_col)
        if zero_velocities:
            state = make_velocities_zero(state)
        return state

    return do_dummy_step
