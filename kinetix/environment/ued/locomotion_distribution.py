"""Functions for generating procedural locomotion environments in Kinetix."""
from functools import partial
from typing import Any, Callable, Dict

from flax import struct
from flax.serialization import to_state_dict
import jax
import jax.numpy as jnp
from jax2d import scene
from jax2d.engine import rmat, select_shape
from jax2d.engine import RigidBody
from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.environment.ued.ued_state import UEDParams
from kinetix.environment.utils import create_empty_env, permute_state


DEFAULT_UED_PARAMS = UEDParams(
    floor_prob_normal=0.75,
    floor_prob_green=0.0,
    floor_prob_blue=0.0,
    floor_prob_red=0.25,
)
SCENE_WIDTH = 5.0


@struct.dataclass
class AugmentedRigidbody(RigidBody):
    """A RigidBody with an added global_index field."""

    global_index: int


def add_poly_connection(
    state: EnvState,
    static_env_params: StaticEnvParams,
    parent: AugmentedRigidbody,
    child_dims: jnp.ndarray,
    parent_offset: str = "left",
    child_offset: str = "top",
    offset_gap: float = 0.05,
    add_joint: bool = True,
    motor_binding: int | jnp.ndarray = 0,
    relative_rotation: float = 0.0,
    poly_kwargs: Dict[str, Any] = {},
    joint_kwargs: Dict[str, Any] = {},
    add_fixed_joint: bool = False,
) -> tuple[EnvState, AugmentedRigidbody]:
    """Creates a new polygon and optionally connects it to the parent with a joint.

    This function is a core utility for building articulated bodies. It calculates
    the correct position for a new child polygon based on the parent's position,
    rotation, and specified attachment points (`parent_offset`, `child_offset`).
    It then adds the new polygon to the scene and, if requested, adds a revolute
    or fixed joint to connect the two bodies.

    Args:
      state: The current environment state.
      static_env_params: The static environment parameters.
      parent: The parent rigid body to attach the new polygon to.
      child_dims: The dimensions (width, height) of the new child polygon.
      parent_offset: The attachment point on the parent body. One of "left",
        "right", "top", "bottom".
      child_offset: The attachment point on the child body. One of "left",
        "right", "top", "bottom".
      offset_gap: A small gap to leave between the parent and child bodies.
      add_joint: Whether to add a joint connecting the parent and child.
      motor_binding: The index of the motor to bind to the joint, if created.
      relative_rotation: The initial rotation of the child relative to the parent.
      poly_kwargs: Additional keyword arguments to pass to `add_rectangle_to_scene`.
      joint_kwargs: Additional keyword arguments to pass to the joint creation
        function (`add_revolute_joint_to_scene` or `add_fixed_joint_to_scene`).
      add_fixed_joint: If True and `add_joint` is True, a fixed joint is created
        instead of a revolute joint.

    Returns:
      A tuple containing:
        - The updated environment state.
        - The `AugmentedRigidbody` for the newly created child polygon.
    """

    def _get_offset(offset_str, dimensions, center, rotation, inverse=False):
        mat = rmat(rotation)

        def _get_delta():
            if offset_str == "left":
                return jnp.array([offset_gap - dimensions[0] / 2, 0.0])
            elif offset_str == "right":
                return jnp.array([-offset_gap + dimensions[0] / 2, 0.0])
            elif offset_str == "top":
                return jnp.array([0.0, -offset_gap + dimensions[1] / 2])
            elif offset_str == "bottom":
                return jnp.array([0.0, offset_gap - dimensions[1] / 2])
            else:
                raise ValueError(f"Unknown offset: {offset_str}")

        m = mat
        return center + m @ ((-1 if inverse else 1) * _get_delta())

    rotation = parent.rotation + relative_rotation
    a_dims = parent.vertices[1] * 2
    a_offset_pos = _get_offset(parent_offset, a_dims, parent.position, parent.rotation)
    b_center = _get_offset(child_offset, child_dims, a_offset_pos, rotation, inverse=True)

    state, idxs = scene.add_rectangle_to_scene(
        state,
        static_env_params,
        position=b_center,
        dimensions=child_dims,
        rotation=rotation,
        **poly_kwargs,
    )
    if add_joint:
        # now we add a joint to this position where they intersect
        if add_fixed_joint:
            state, joint_index = scene.add_fixed_joint_to_scene(
                state,
                static_env_params,
                parent.global_index,
                idxs[1],
                a_relative_pos=jnp.linalg.inv(rmat(parent.rotation)) @ (a_offset_pos - parent.position),
                b_relative_pos=jnp.linalg.inv(rmat(rotation)) @ (a_offset_pos - b_center),
                **{k: v for k, v in joint_kwargs.items() if k not in ["motor_power"]},
            )
        else:
            state, joint_index = scene.add_revolute_joint_to_scene(
                state,
                static_env_params,
                parent.global_index,
                idxs[1],
                a_relative_pos=jnp.linalg.inv(rmat(parent.rotation)) @ (a_offset_pos - parent.position),
                b_relative_pos=jnp.linalg.inv(rmat(rotation)) @ (a_offset_pos - b_center),
                motor_on=True,
                has_joint_limits=True,
                min_rotation=-jnp.pi / 4,
                max_rotation=jnp.pi / 4,
                **joint_kwargs,
            )
            state = state.replace(motor_bindings=state.motor_bindings.at[joint_index].set(motor_binding))
    rb_to_return = select_shape(state, idxs[1], static_env_params)

    return state, AugmentedRigidbody(**to_state_dict(rb_to_return), global_index=idxs[1])


def fix_out_of_bounds(state, body, static_env_params):
    """Adjusts the agent's position to ensure it's within world bounds.

    This function calculates the bounding box of the agent's dynamic polygons
    and moves the entire agent if it's spawning too close to the floor or the
    side walls. This prevents the agent from getting stuck at the start of an
    episode.

    Args:
      state: The current environment state.
      body: The main body of the agent. Used as a fallback for inactive polygons.
      static_env_params: Static environment parameters.

    Returns:
      The updated environment state with the agent's position corrected.
    """
    min_height_above_ground = 0.1
    ground_height = 0.4
    min_height = ground_height + min_height_above_ground
    min_dist_from_wall = 0.2

    vertices = state.polygon.vertices
    vertices = jnp.where(
        state.polygon.active[:, None, None],
        vertices + state.polygon.position[:, None, :],
        (jnp.zeros_like(vertices) + body.vertices) + body.position,
    )
    vertices = vertices[static_env_params.num_static_fixated_polys :].reshape(-1, 2)
    min_x = vertices[:, 0].min()
    max_x = vertices[:, 0].max()
    min_y = vertices[:, 1].min()
    to_move_right = jnp.maximum(0.0, min_dist_from_wall - min_x)
    to_move_left = jnp.maximum(0.0, max_x - (SCENE_WIDTH - min_dist_from_wall))
    to_move_right -= to_move_left
    to_move_up = jnp.maximum(0.0, min_height - min_y)
    delta = jnp.array([to_move_right, to_move_up])
    state = state.replace(
        polygon=state.polygon.replace(
            position=state.polygon.position.at[static_env_params.num_static_fixated_polys :].add(delta)
        ),
        joint=state.joint.replace(global_position=state.joint.global_position + delta),
    )
    return state


def _compute_initial_goal_position(rng, static_env_params, env_params, no_randomness=False):
    """Computes the initial x, y position for the goal circle.

    The goal is placed at a fixed y-coordinate and a variable x-coordinate.
    The x-coordinate is sampled uniformly within a predefined range on the right
    side of the scene.

    Args:
      rng: JAX random key.
      static_env_params: Static environment parameters.
      env_params: Dynamic environment parameters.
      no_randomness: If True, place the goal at the midpoint of the allowed range
        instead of sampling randomly.

    Returns:
      A jnp.ndarray of shape (2,) representing the goal's position.
    """
    min_allowed_x = 2.0
    circle_y = 1.0
    max_allowed_x = static_env_params.screen_dim[0] / env_params.pixels_per_unit * 0.9
    if no_randomness:
        scaling_factor = 0.5
    else:
        scaling_factor = jax.random.uniform(rng)
    return jnp.array(
        [
            (max_allowed_x - min_allowed_x) * scaling_factor + min_allowed_x,
            circle_y,
        ]
    )


def add_goal(
    rng: jax.Array,
    state: EnvState,
    static_env_params: StaticEnvParams,
    env_params: EnvParams,
    maybe_swap_position_fn: Callable[[jnp.ndarray], jnp.ndarray],
    no_randomness: bool,
) -> EnvState:
    """Adds the goal circle to the environment state.

    Args:
      rng: JAX random key.
      state: The current environment state.
      static_env_params: Static environment parameters.
      env_params: Dynamic environment parameters.
      _maybe_swap_position: A function that may swap the horizontal position of
        the goal, effectively placing it on the left side of the scene.
      no_randomness: If True, the goal is placed at a fixed position.

    Returns:
      The updated environment state with the goal added.
    """
    # add the goal:
    state, (circle_idx, _) = scene.add_circle_to_scene(
        state,
        static_env_params,
        position=maybe_swap_position_fn(
            _compute_initial_goal_position(rng, static_env_params, env_params, no_randomness)
        ),
        radius=0.3,
        fixated=True,
    )
    state = state.replace(
        circle_shape_roles=state.circle_shape_roles.at[circle_idx].set(2),
    )
    return state


def randomise_floor_colour(rng: jax.Array, state: EnvState, ued_params: UEDParams) -> EnvState:
    """Sets the floor color based on UED probabilities.

    Args:
      rng: JAX random key.
      state: The current environment state.
      ued_params: UED parameters containing the probabilities for each floor color.

    Returns:
      The updated environment state with the new floor color.
    """
    prob_of_floor_colour = jnp.array(
        [
            ued_params.floor_prob_normal,
            ued_params.floor_prob_green,
            ued_params.floor_prob_blue,
            ued_params.floor_prob_red,
        ]
    )
    floor_colour = jax.random.choice(rng, jnp.arange(4), p=prob_of_floor_colour)
    state = state.replace(polygon_shape_roles=state.polygon_shape_roles.at[0].set(floor_colour))
    return state


def sample_uniform_and_exclude(
    rng: jax.Array,
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    minval_exclude: jnp.ndarray,
    maxval_exclude: jnp.ndarray,
) -> jnp.ndarray:
    """Samples a random number from [minval, minval_exclude) union (maxval_exclude, maxval).

    Args:
      rng: JAX random key.
      minval: The minimum value of the overall range.
      maxval: The maximum value of the overall range.
      minval_exclude: The minimum value of the excluded range.
      maxval_exclude: The maximum value of the excluded range.

    Returns:
      A random sample from the valid ranges.
    """

    # Define the two valid intervals element-wise
    interval1_start = minval
    interval1_end = minval_exclude

    interval2_start = maxval_exclude
    interval2_end = maxval

    # Calculate the length of each valid interval element-wise
    length1 = interval1_end - interval1_start
    length2 = interval2_end - interval2_start

    # Calculate the total length of the valid sampling range element-wise
    total_length = length1 + length2

    rand_val_scaled = jax.random.uniform(rng, shape=jnp.shape(total_length), minval=0.0, maxval=total_length)

    # Determine which interval each random number falls into and map it back
    sample = jnp.where(
        rand_val_scaled < length1,
        interval1_start + rand_val_scaled,  # Sample from the first interval
        interval2_start + (rand_val_scaled - length1),  # Sample from the second interval
    )

    assert sample.shape == jnp.shape(minval)
    return sample


def sample_uniform(
    rng: jax.Array,
    minval: jnp.ndarray | float,
    maxval: jnp.ndarray | float,
    minval_exclude: jnp.ndarray | None = None,
    maxval_exclude: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Samples from a uniform distribution, optionally excluding a sub-range.

    If `minval_exclude` and `maxval_exclude` are provided, this function samples
    from `[minval, minval_exclude) U (maxval_exclude, maxval)`. Otherwise, it
    samples from `[minval, maxval)`.

    Args:
      rng: JAX random key.
      minval: The minimum value of the overall range.
      maxval: The maximum value of the overall range.
      minval_exclude: The minimum value of the excluded range.
      maxval_exclude: The maximum value of the excluded range.

    Returns:
      A random sample.
    """
    if minval_exclude is None or maxval_exclude is None:
        return jax.random.uniform(rng, shape=jnp.shape(minval), minval=minval, maxval=maxval)
    else:
        return sample_uniform_and_exclude(rng, minval, maxval, minval_exclude, maxval_exclude)


def make_get_dims_fn(
    default: Any,
    sampling_range_ratio: Any,
    sampling_range_ratio_exclude: Any | None = None,
) -> Callable[[jax.Array], jnp.ndarray]:
    """Creates a function that samples dimensions for a body part.

    This is a factory that returns a function `_get_dims(rng)`. The returned
    function samples dimensions from a uniform distribution centered around a
    `default` value. The range of this distribution is controlled by
    `sampling_range_ratio`. An inner range can optionally be excluded from
    sampling.

    Args:
      default: The default dimensions (e.g., `(width, height)`).
      sampling_range_ratio: The ratio of the default value that defines the outer
        sampling range. The range will be `default * (1 +/- sampling_range_ratio)`.
      sampling_range_ratio_exclude: If provided, the ratio that defines an inner
        range to exclude from sampling. The excluded range will be
        `default * (1 +/- sampling_range_ratio_exclude)`.

    Returns:
      A function that takes a JAX random key and returns sampled dimensions.
    """
    default = jnp.array(default)
    sampling_range_ratio = jnp.array(sampling_range_ratio)
    if sampling_range_ratio_exclude is not None:
        sampling_range_ratio_exclude = jnp.array(sampling_range_ratio_exclude)
        kwargs = {
            "minval_exclude": default * (1 - sampling_range_ratio_exclude),
            "maxval_exclude": default * (1 + sampling_range_ratio_exclude),
        }
    else:
        kwargs = {}

    def _get_dims(rng):
        return sample_uniform(
            rng,
            minval=default * (1 - sampling_range_ratio),
            maxval=default * (1 + sampling_range_ratio),
            **kwargs,
        )

    return _get_dims


def sample_locomotion_level(
    rng: jax.Array,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    ued_params: UEDParams | None = None,
    random_motors: bool = False,
    body_rotation_range: float = 0.0,
    body_dimension_sampling_range_ratio: float = 0.5,
    leg_dimension_sampling_range_ratio: float = 0.5,
    ankle_dimension_sampling_range_ratio: float = 0.5,
    body_dimension_sampling_range_ratio_exclude: float | None = None,
    leg_dimension_sampling_range_ratio_exclude: float | None = None,
    ankle_dimension_sampling_range_ratio_exclude: float | None = None,
    body_dimension_default: tuple[float, float] = (0.6, 0.2),
    leg_dimension_default: tuple[float, float] = (0.1, 0.3),
    ankle_dimension_default: tuple[float, float] = (0.1, 0.2),
    do_permute_state: bool = False,
    add_ankles: bool = True,
    num_random_legs_min: int = 1,
    num_random_legs_max: int = 1,
    friction_default: float = 1.0,
    density_default: float = 1.0,
    density_absolute_range: float = 0.0,
    friction_absolute_range: float = 0.0,
    density_absolute_range_exclude: float | None = None,
    friction_absolute_range_exclude: float | None = None,
    ensure_leg_dims_equal: bool = False,
    ensure_ankle_dims_equal: bool = False,
    ensure_limb_thickness_equal: bool = False,
    fix_goal_position: bool = False,
) -> EnvState:
    """Generates a procedural locomotion environment (V2).

    This version provides more granular control over the randomization of the
    agent's morphology and physical properties compared to V1. It allows for
    defining sampling ranges (and exclusion zones) for body part dimensions and
    physical properties like friction and density.

    Args:
      rng: JAX random key.
      env_params: Dynamic environment parameters.
      static_env_params: Static environment parameters.
      ued_params: Parameters for Unsupervised Environment Design.
      random_motors: If True, randomize motor bindings.
      body_rotation_range: Range for uniform sampling of the initial body
        rotation (e.g., `jnp.pi / 8`).
      body_dimension_sampling_range_ratio: Ratio for sampling body dimensions.
      leg_dimension_sampling_range_ratio: Ratio for sampling leg dimensions.
      ankle_dimension_sampling_range_ratio: Ratio for sampling ankle dimensions.
      body_dimension_sampling_range_ratio_exclude: Exclusion ratio for body dims.
      leg_dimension_sampling_range_ratio_exclude: Exclusion ratio for leg dims.
      ankle_dimension_sampling_range_ratio_exclude: Exclusion ratio for ankle dims.
      body_dimension_default: Default dimensions for the main body.
      leg_dimension_default: Default dimensions for the legs.
      ankle_dimension_default: Default dimensions for the ankles.
      do_permute_state: If True, permute entity order in the state.
      add_ankles: If True, add ankle segments to the legs.
      num_random_legs_min: Minimum number of additional random legs to add.
      num_random_legs_max: Maximum number of additional random legs to add.
      friction_default: Default friction for body parts.
      density_default: Default density for body parts.
      density_absolute_range: Absolute range for sampling density.
      friction_absolute_range: Absolute range for sampling friction.
      density_absolute_range_exclude: Absolute exclusion range for density.
      friction_absolute_range_exclude: Absolute exclusion range for friction.
      ensure_leg_dims_equal: If True, both legs will have the same dimensions.
      ensure_ankle_dims_equal: If True, both ankles will have the same dimensions.
      ensure_limb_thickness_equal: If True, all limbs will have the same
        thickness (x-dimension).
      fix_goal_position: If True, the goal is placed at a fixed, non-random
        position.

    Returns:
      The initial `EnvState` for the generated level.
    """
    if ued_params is None:
        ued_params = DEFAULT_UED_PARAMS

    # Start with empty state
    state = create_empty_env(static_env_params)

    # Perhaps swap the position of the goal and the agent
    rng, _rng = jax.random.split(rng)
    should_swap_goal_agent = jax.random.uniform(_rng) < 0.5

    def _maybe_swap_position(position):
        return position.at[0].set(
            (SCENE_WIDTH - position[0]) * should_swap_goal_agent + position[0] * (1 - should_swap_goal_agent)
        )

    def sample_friction_density_kwargs(rng):
        rng_friction, rng_density = jax.random.split(rng)
        friction_kwargs = {}
        density_kwargs = {}
        if friction_absolute_range_exclude is not None:
            friction_kwargs = {
                "minval_exclude": friction_default - friction_absolute_range_exclude,
                "maxval_exclude": friction_default + friction_absolute_range_exclude,
            }
        if density_absolute_range_exclude is not None:
            density_kwargs = {
                "minval_exclude": density_default - density_absolute_range_exclude,
                "maxval_exclude": density_default + density_absolute_range_exclude,
            }
        return {
            "friction": sample_uniform(
                rng_friction,
                minval=friction_default - friction_absolute_range,
                maxval=friction_default + friction_absolute_range,
                **friction_kwargs,
            ),
            "density": sample_uniform(
                rng_density,
                minval=density_default - density_absolute_range,
                maxval=density_default + density_absolute_range,
                **density_kwargs,
            ),
        }

    get_body_dimensions_fn = make_get_dims_fn(
        body_dimension_default,
        body_dimension_sampling_range_ratio,
        body_dimension_sampling_range_ratio_exclude,
    )
    get_leg_dimensions_fn = make_get_dims_fn(
        leg_dimension_default,
        leg_dimension_sampling_range_ratio,
        leg_dimension_sampling_range_ratio_exclude,
    )
    get_ankle_dimensions_fn = make_get_dims_fn(
        ankle_dimension_default,
        ankle_dimension_sampling_range_ratio,
        ankle_dimension_sampling_range_ratio_exclude,
    )

    # get the main body
    rng, _rng_pos, _rng_dims, _rng_rot = jax.random.split(rng, 4)
    body_center_pos = _maybe_swap_position(
        sample_uniform(
            _rng_pos,
            minval=jnp.array([0.0, 0.0]),
            maxval=jnp.array([0.0, 0.0]),
        )
    )

    body_dimensions = get_body_dimensions_fn(_rng_dims)

    body_rotation = sample_uniform(
        _rng_rot,
        minval=-body_rotation_range,
        maxval=body_rotation_range,
    )

    state, (poly_idx, global_idx) = scene.add_rectangle_to_scene(
        state,
        static_env_params,
        position=body_center_pos,
        dimensions=body_dimensions,
        rotation=body_rotation,
        fixated=False,
    )

    # body: should be green.
    state = state.replace(polygon_shape_roles=state.polygon_shape_roles.at[poly_idx].set(1))
    body = AugmentedRigidbody(
        **to_state_dict(select_shape(state, global_idx, static_env_params)),
        global_index=global_idx,
    )
    rng, _rng_legs, _rng_ankles = jax.random.split(rng, 3)
    leg_dims = jax.vmap(get_leg_dimensions_fn)(jax.random.split(_rng_legs, 2))
    ankle_dims = jax.vmap(get_ankle_dimensions_fn)(jax.random.split(_rng_ankles, 2))

    if ensure_leg_dims_equal:
        leg_dims = leg_dims.at[1].set(leg_dims[0])
    if ensure_ankle_dims_equal:
        ankle_dims = ankle_dims.at[1].set(ankle_dims[0])

    if ensure_limb_thickness_equal:
        normal_limb_thickness = leg_dims[0][0]
        leg_dims = leg_dims.at[:, 0].set(normal_limb_thickness)
        ankle_dims = ankle_dims.at[:, 0].set(normal_limb_thickness)

    rng, _rng = jax.random.split(rng)
    if random_motors:
        motor_bindings = jax.random.randint(
            _rng,
            shape=(4,),
            minval=0,
            maxval=static_env_params.num_motor_bindings - 1,
        )
    else:
        motor_bindings = [0, 1, 2, 3]

    default_joint_kwargs = dict(motor_power=3.0)
    # we always need to add at least two legs
    rng, _rng = jax.random.split(rng)
    _rng_friction_density = jax.random.split(_rng, 4)
    state, left_leg = add_poly_connection(
        state,
        static_env_params,
        body,
        leg_dims[0],
        parent_offset="left",
        child_offset="top",
        relative_rotation=-jnp.pi / 8,
        motor_binding=motor_bindings[0],
        joint_kwargs=default_joint_kwargs,
        poly_kwargs=sample_friction_density_kwargs(_rng_friction_density[0]),
    )

    state, right_leg = add_poly_connection(
        state,
        static_env_params,
        body,
        leg_dims[1],
        parent_offset="right",
        child_offset="top",
        motor_binding=motor_bindings[1],
        relative_rotation=jnp.pi / 8,
        joint_kwargs=default_joint_kwargs,
        poly_kwargs=sample_friction_density_kwargs(_rng_friction_density[1]),
    )

    if add_ankles:
        state, left_ankle = add_poly_connection(
            state,
            static_env_params,
            left_leg,
            ankle_dims[0],
            parent_offset="bottom",
            child_offset="top",
            relative_rotation=0,
            motor_binding=motor_bindings[2],
            joint_kwargs=default_joint_kwargs,
            poly_kwargs=sample_friction_density_kwargs(_rng_friction_density[2]),
        )

        state, right_ankle = add_poly_connection(
            state,
            static_env_params,
            right_leg,
            ankle_dims[1],
            parent_offset="bottom",
            child_offset="top",
            relative_rotation=0,
            motor_binding=motor_bindings[3],
            joint_kwargs=default_joint_kwargs,
            poly_kwargs=sample_friction_density_kwargs(_rng_friction_density[3]),
        )
    else:
        left_ankle = left_leg
        right_ankle = right_leg

    def _make_maybe_add_joint(add_probability: float = 0.5):
        def _maybe_add_joint(carry, step):
            rng, mask = step

            def _add(carry, rng):
                state, body, prev_left, prev_right = carry
                _rng_parent, _rng_leg, _rng_motor, _rng_poly = jax.random.split(rng, 4)
                what_to_use_as_parent = jax.random.randint(_rng_parent, shape=(), minval=0, maxval=2)
                parent = jax.lax.switch(
                    what_to_use_as_parent,
                    [lambda: body, lambda: prev_left, lambda: prev_right],
                )

                state, new_body = add_poly_connection(
                    state,
                    static_env_params,
                    parent,
                    get_leg_dimensions_fn(_rng_leg),
                    parent_offset="bottom",
                    child_offset="top",
                    relative_rotation=0,
                    motor_binding=jax.random.randint(
                        _rng_motor,
                        shape=(),
                        minval=0,
                        maxval=static_env_params.num_motor_bindings - 1,
                    ),
                    joint_kwargs=default_joint_kwargs,
                    poly_kwargs=sample_friction_density_kwargs(_rng_poly),
                )
                select = lambda idx, new, old: jax.lax.select(what_to_use_as_parent == idx, new, old)

                body = jax.tree.map(partial(select, 0), new_body, body)
                prev_left = jax.tree.map(partial(select, 1), new_body, prev_left)
                prev_right = jax.tree.map(partial(select, 2), new_body, prev_right)

                return (state, body, prev_left, prev_right)

            def _noop(carry, rng):
                return carry

            rng, rng_choose, rng_sub = jax.random.split(rng, 3)
            return (
                jax.lax.cond(
                    (jax.random.uniform(rng_choose, shape=()) < add_probability) & mask,
                    _add,
                    _noop,
                    carry,
                    rng_sub,
                ),
                None,
            )

        return _maybe_add_joint

    rng, _rng = jax.random.split(rng)
    how_many_legs_to_add = jax.random.randint(
        _rng,
        shape=(),
        minval=num_random_legs_min,
        maxval=num_random_legs_max + 1,
    )

    ankle_factor = 2 if add_ankles else 0
    max_num_legs = min(
        static_env_params.num_polygons
        - static_env_params.num_static_fixated_polys
        - 3  # 3 for body, left leg, right leg
        - ankle_factor,
        static_env_params.num_joints - 2 - ankle_factor,
    )
    rng, _rng = jax.random.split(rng)
    leg_add_mask = jnp.arange(max_num_legs) < how_many_legs_to_add
    add_joint_fn = _make_maybe_add_joint(add_probability=1.0)
    carry = (state, body, left_ankle, right_ankle)
    for key, mask in zip(jax.random.split(_rng, max_num_legs), leg_add_mask):
        carry, _ = add_joint_fn(carry, (key, mask))

    state = carry[0]

    state = fix_out_of_bounds(state, body, static_env_params)

    rng, _rng_goal, _rng_colour, _rng_friction_density, _rng_permute = jax.random.split(rng, 5)
    state = add_goal(
        _rng_goal,
        state,
        static_env_params,
        env_params,
        _maybe_swap_position,
        no_randomness=fix_goal_position,
    )

    state = randomise_floor_colour(_rng_colour, state, ued_params)

    kw = sample_friction_density_kwargs(_rng_friction_density)
    # set the floor friction
    state = state.replace(
        polygon=state.polygon.replace(
            friction=state.polygon.friction.at[0].set(kw["friction"])  # pytype: disable=attribute-error
        )
    )

    if do_permute_state:
        return permute_state(_rng_permute, state, static_env_params)
    else:
        return state


def make_reset_fn_custom_distribution(
    fn_name: str,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    fn_kwargs: Dict[str, Any] = {},
) -> Callable[[jax.Array], EnvState]:
    """Creates a jitted environment reset function from a named distribution.

    Args:
      fn_name: The name of the level generation function to use. Must be a key in
        `functions` or `functions_handmade`.
      env_params: Dynamic environment parameters to be passed to the generation
        function.
      static_env_params: Static environment parameters to be passed to the
        generation function.
      fn_kwargs: A dictionary of keyword arguments to be partially applied to the
        chosen level generation function.

    Returns:
      A JIT-compiled function `reset(rng) -> EnvState` that samples a new level.
    """
    functions = {
        "locomotion": sample_locomotion_level,
    }
    assert fn_name in functions, f"Function {fn_name} not implemented."
    reset_fn = functions[fn_name]

    @jax.jit
    def reset(rng):
        sampled_level = reset_fn(rng, env_params, static_env_params, **fn_kwargs)

        return sampled_level

    return reset
