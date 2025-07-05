import math
from functools import partial

import jax
import jax.numpy as jnp
from jax2d.engine import (
    calc_inverse_inertia_circle,
    calc_inverse_inertia_polygon,
    calc_inverse_mass_circle,
    calc_inverse_mass_polygon,
    calculate_collision_matrix,
    recalculate_mass_and_inertia,
    select_shape,
)
from jax2d.maths import rmat
from jax2d.sim_state import Joint, RigidBody, Thruster

from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.environment.ued.ued_state import UEDParams
from kinetix.environment.ued.util import (
    are_there_shapes_present,
    count_roles,
    get_role,
    is_space_for_joint,
    is_space_for_shape,
    make_do_dummy_step,
    random_position_on_circle,
    random_position_on_polygon,
    sample_dimensions,
)


@partial(jax.jit, static_argnums=(3, 4))
def mutate_add_shape(
    rng,
    state: EnvState,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    ued_params: UEDParams,
    force_no_fixate: bool = False,
):
    def do_dummy(rng, state):
        return state

    def do_add(rng, state):
        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, 9)

        space_for_new_rect = state.polygon.active.astype(int).sum() < static_env_params.num_polygons
        space_for_new_circle = state.circle.active.astype(int).sum() < static_env_params.num_circles

        is_rect_p = jnp.array([space_for_new_rect * 1.0, space_for_new_circle * 1.0])
        is_rect = jax.random.choice(_rngs[0], jnp.array([True, False], dtype=bool), p=is_rect_p)

        rect_index = jnp.argmin(state.polygon.active)
        circle_index = jnp.argmin(state.circle.active)

        shape_role = get_role(_rngs[1], state, static_env_params)

        max_shape_size = (
            jnp.array([1.0, ued_params.goal_body_size_factor, ued_params.goal_body_size_factor, 1.0])[shape_role]
            * ued_params.max_shape_size
        )

        vertices, half_dimensions, radius = sample_dimensions(
            _rngs[2],
            static_env_params,
            is_rect,
            ued_params,
            max_shape_size=max_shape_size,
        )
        n_vertices = jax.lax.select(ued_params.generate_triangles, jax.random.choice(_rngs[3], jnp.array([3, 4])), 4)

        largest = jnp.max(jnp.array([half_dimensions[0] * jnp.sqrt(2), half_dimensions[1] * jnp.sqrt(2), radius]))

        screen_dim_world = (
            static_env_params.screen_dim[0] / env_params.pixels_per_unit,
            static_env_params.screen_dim[1] / env_params.pixels_per_unit,
        )
        min_x = largest
        max_x = screen_dim_world[0] - largest
        min_y = largest + 0.4
        max_y = screen_dim_world[1] - largest

        def _og_minmax():
            return min_x, max_x, min_y, max_y

        def _opposite_minmax():
            return jax.lax.switch(
                shape_role,
                [
                    (lambda: (min_x, max_x, min_y, max_y)),
                    (lambda: (min_x, max_x - screen_dim_world[0] / 2, min_y, max_y)),
                    (lambda: (min_x + screen_dim_world[0] / 2, max_x, min_y, max_y)),
                    (lambda: (min_x, max_x, min_y, max_y)),
                ],
            )

        min_x, max_x, min_y, max_y = jax.lax.cond(
            jax.random.uniform(_rngs[4], shape=()) < ued_params.goal_body_opposide_side_chance,
            _opposite_minmax,
            _og_minmax,
        )

        position = jax.random.uniform(_rngs[5], shape=(2,)) * jnp.array(
            [
                max_x - min_x,
                max_y - min_y,
            ]
        ) + jnp.array([min_x, min_y])

        rotation = jax.random.uniform(_rngs[6], shape=()) * 2 * math.pi
        velocity = jnp.array([0.0, 0.0])
        angular_velocity = 0.0

        density = 1.0
        inverse_mass = jax.lax.select(
            is_rect,
            calc_inverse_mass_polygon(vertices, n_vertices, static_env_params, density)[0],
            calc_inverse_mass_circle(radius, density),
        )

        inverse_inertia = jax.lax.select(
            is_rect,
            calc_inverse_inertia_polygon(vertices, n_vertices, static_env_params, density),
            calc_inverse_inertia_circle(radius, density),
        )

        fixate_chance = ued_params.fixate_chance_min + (1.0 / inverse_mass) * ued_params.fixate_chance_scale
        fixate_chance = jnp.minimum(fixate_chance, ued_params.fixate_chance_max)
        is_fixated = jax.random.uniform(_rngs[7], shape=()) < fixate_chance
        is_fixated &= ~force_no_fixate

        inverse_mass *= 1 - is_fixated
        inverse_inertia *= 1 - is_fixated

        # We want to bias fixated shapes to starting nearer the bottom half of the screen
        fixate_shape_bottom_bias = (
            ued_params.fixate_shape_bottom_bias + ued_params.fixate_shape_bottom_bias_special_role * (shape_role != 0)
        )
        is_forcing_bottom = jax.random.uniform(_rngs[8]) < fixate_shape_bottom_bias

        half_screen_height = (static_env_params.screen_dim[1] / env_params.pixels_per_unit) / 2.0
        position = jax.lax.select(
            is_fixated & is_forcing_bottom & (position[1] >= half_screen_height),
            position.at[1].add(-half_screen_height),
            position,
        )

        # This could be either a rect or a circle
        new_rigid_body = RigidBody(
            position=position,
            velocity=velocity,
            inverse_mass=inverse_mass,
            inverse_inertia=inverse_inertia,
            rotation=rotation,
            angular_velocity=angular_velocity,
            radius=radius,
            active=True,
            friction=1.0,
            vertices=vertices,
            n_vertices=n_vertices,
            collision_mode=1,
            restitution=0.0,
        )

        state = state.replace(
            polygon=jax.tree.map(
                lambda x, y: jax.lax.select(is_rect, y.at[rect_index].set(x), y), new_rigid_body, state.polygon
            ),
            circle=jax.tree.map(
                lambda x, y: jax.lax.select(jnp.logical_not(is_rect), y.at[circle_index].set(x), y),
                new_rigid_body,
                state.circle,
            ),
            polygon_shape_roles=jax.lax.select(
                is_rect,
                state.polygon_shape_roles.at[rect_index].set(shape_role),
                state.polygon_shape_roles,
            ),
            circle_shape_roles=jax.lax.select(
                jnp.logical_not(is_rect),
                state.circle_shape_roles.at[circle_index].set(shape_role),
                state.circle_shape_roles,
            ),
        )
        return recalculate_mass_and_inertia(state, static_env_params, state.polygon_densities, state.circle_densities)

    return jax.lax.cond(is_space_for_shape(state), do_add, do_dummy, rng, state)


@partial(jax.jit, static_argnums=(3, 4))
def mutate_add_connected_shape(
    rng,
    state: EnvState,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    ued_params: UEDParams,
    force_rjoint: bool = False,
):
    def do_dummy(rng, state):
        return state, False

    def do_add(rng, state):
        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, 21)

        # Select a random index amongst the currently active shapes.
        p_rect = state.polygon.active.at[: static_env_params.num_static_fixated_polys].set(False)
        p_circle = state.circle.active

        p_rect = p_rect.astype(jnp.float32)
        p_circle = p_circle.astype(jnp.float32)

        p_rect *= (state.polygon.inverse_mass == 0) * ued_params.connect_to_fixated_prob_coeff + (
            state.polygon.inverse_mass != 0
        ) * 1.0
        p_circle *= (state.circle.inverse_mass == 0) * ued_params.connect_to_fixated_prob_coeff + (
            state.circle.inverse_mass != 0
        ) * 1.0

        # Bias based on number of existing connections
        rect_connections = jnp.zeros(static_env_params.num_polygons)
        circle_connections = jnp.zeros(static_env_params.num_circles)

        rect_connections = rect_connections.at[state.joint.a_index].add(
            jnp.ones(static_env_params.num_joints)
            * state.joint.active
            * (state.joint.a_index < static_env_params.num_polygons)
        )
        rect_connections = rect_connections.at[state.joint.b_index].add(
            jnp.ones(static_env_params.num_joints)
            * state.joint.active
            * (state.joint.b_index < static_env_params.num_polygons)
        )

        circle_connections = circle_connections.at[state.joint.a_index - static_env_params.num_polygons].add(
            jnp.ones(static_env_params.num_joints)
            * state.joint.active
            * (state.joint.a_index >= static_env_params.num_polygons)
        )
        circle_connections = circle_connections.at[state.joint.b_index - static_env_params.num_polygons].add(
            jnp.ones(static_env_params.num_joints)
            * state.joint.active
            * (state.joint.b_index >= static_env_params.num_polygons)
        )

        # Rectangles can have up to 2 connections
        p_rect *= (-rect_connections + 2.0) / 2.0
        p_rect = jnp.maximum(p_rect, 0.0)
        # Circles can have 1 connection
        p_circle *= circle_connections == 0

        # To sample a target rect/circle, we have to have at least one.
        target_rect_p = jnp.array(
            [
                (state.polygon.active.astype(int).sum() > static_env_params.num_static_fixated_polys) * 1.0,
                (state.circle.active.astype(int).sum() > 0) * 1.0,
            ]
        )

        # Don't connect to a circle if no connection-free ones exist
        target_rect_p = target_rect_p.at[1].mul(p_circle.sum() > 0)

        space_for_new_rect = state.polygon.active.astype(int).sum() < static_env_params.num_polygons
        space_for_new_circle = state.circle.active.astype(int).sum() < static_env_params.num_circles

        is_target_rect = jax.random.choice(_rngs[0], jnp.array([True, False], dtype=bool), p=target_rect_p) | (
            ~space_for_new_rect
        )

        is_rect_p = jnp.array([space_for_new_rect * 1.0, space_for_new_circle * 1.0])
        is_rect = jax.random.choice(_rngs[1], jnp.array([True, False], dtype=bool), p=is_rect_p) | (
            ~is_target_rect & space_for_new_rect
        )

        shape_index = jax.lax.select(
            is_rect,
            jnp.argmin(state.polygon.active),
            jnp.argmin(state.circle.active),
        )
        unified_shape_index = shape_index + (~is_rect) * static_env_params.num_polygons

        vertices, half_dimensions, radius = sample_dimensions(
            _rngs[2], static_env_params, is_rect, ued_params, max_shape_size=ued_params.max_shape_size
        )
        n_vertices = jax.lax.select(ued_params.generate_triangles, jax.random.choice(_rngs[3], jnp.array([3, 4])), 4)

        rotation = jax.random.uniform(_rngs[4], shape=()) * 2 * math.pi
        velocity = jnp.array([0.0, 0.0])
        angular_velocity = 0.0

        density = 1.0
        inverse_mass = jax.lax.select(
            is_rect,
            calc_inverse_mass_polygon(vertices, n_vertices, static_env_params, density)[0],
            calc_inverse_mass_circle(radius, density),
        )

        inverse_inertia = jax.lax.select(
            is_rect,
            calc_inverse_inertia_polygon(vertices, n_vertices, static_env_params, density),
            calc_inverse_inertia_circle(radius, density),
        )

        # Joint

        current_num_rjoints = (jnp.logical_not(state.joint.is_fixed_joint) * state.joint.active).sum()
        is_rjoint = jnp.logical_or(
            jnp.logical_or(jax.random.uniform(_rngs[5]) < 0.5, force_rjoint),
            current_num_rjoints < ued_params.min_rjoints_bias,
        )

        joint_index = jnp.argmin(state.joint.active)

        local_joint_position_rect = random_position_on_polygon(_rngs[6], vertices, n_vertices, static_env_params)
        local_joint_position_circle = random_position_on_circle(_rngs[7], radius, on_centre_chance=1.0)

        local_joint_position = jax.lax.select(is_rect, local_joint_position_rect, local_joint_position_circle)

        p_rect = jax.lax.select(p_rect.sum() == 0, state.polygon.active.astype(jnp.float32), p_rect)
        p_circle = jax.lax.select(p_circle.sum() == 0, state.circle.active.astype(jnp.float32), p_circle)

        target_index = jax.lax.select(
            is_target_rect,
            jax.random.choice(
                _rngs[8],
                jnp.arange(static_env_params.num_polygons),
                p=p_rect,
            ),
            jax.random.choice(
                _rngs[9],
                jnp.arange(static_env_params.num_circles),
                p=p_circle,
            ),
        )

        unified_target_index = target_index + jnp.logical_not(is_target_rect) * static_env_params.num_polygons
        target_shape = select_shape(state, unified_target_index, static_env_params)

        target_joint_position_rect = random_position_on_polygon(
            _rngs[10], state.polygon.vertices[target_index], state.polygon.n_vertices[target_index], static_env_params
        )
        target_joint_position_circle = random_position_on_circle(
            _rngs[11], state.circle.radius[target_index], on_centre_chance=1.0
        )

        target_joint_position = jax.lax.select(is_target_rect, target_joint_position_rect, target_joint_position_circle)

        # Calculate the world position of the new shape
        # We know the rotation of the new shape. We also know the position of the current shape, which we want to remain fixed.
        # Set `position` such that local_joint_position is the same as `target_joint_position`
        global_joint_pos = target_shape.position + jnp.matmul(rmat(target_shape.rotation), target_joint_position)
        position = global_joint_pos - jnp.matmul(rmat(rotation), local_joint_position)

        _, pos_diff = calc_inverse_mass_polygon(vertices, n_vertices, static_env_params, density)
        position = jax.lax.select(is_rect, position + pos_diff, position)
        local_joint_position = jax.lax.select(is_rect, local_joint_position - pos_diff, local_joint_position)
        vertices = jax.lax.select(is_rect, vertices - pos_diff[None], vertices)

        target_role = jax.lax.select(
            is_target_rect, state.polygon_shape_roles[target_index], state.circle_shape_roles[target_index]
        )

        # We cannot have role 1 and role 2 being connected.
        p = jnp.array([1.0, 1.0, 1.0, 1.0])
        # If role is 0, keep all probs at 1, otherwise set the target role's complement to 0 prob
        # 3 - role turns 1 to 2 and 2 to 1
        # If the target role is three, we set everything to zero except for the default
        p = jax.lax.select(
            target_role == 0,
            p,
            jax.lax.select(
                target_role <= 2,
                p.at[3 - target_role].set(False).at[3].set(False),
                (p.at[2].set(False).at[1].set(False)),
            ),
        )

        shape_role = get_role(_rngs[12], state, static_env_params, initial_p=p)

        # This could be either a rect or a circle
        new_rigid_body = RigidBody(
            position=position,
            velocity=velocity,
            inverse_mass=inverse_mass,
            inverse_inertia=inverse_inertia,
            rotation=rotation,
            angular_velocity=angular_velocity,
            radius=radius,
            active=True,
            friction=1.0,
            vertices=vertices,
            n_vertices=n_vertices,
            collision_mode=1,
            restitution=0.0,
        )

        # Change the shape indices such that a_index is less than b_index
        a_index = shape_index + (1 - is_rect) * static_env_params.num_polygons
        b_index = target_index + (1 - is_target_rect) * static_env_params.num_polygons

        should_swap = a_index > b_index
        a_index, b_index, local_joint_position, target_joint_position, shape_a, shape_b = jax.lax.cond(
            should_swap,
            lambda x: (x[1], x[0], x[3], x[2], x[5], x[4]),  # pairwise swap
            lambda x: x,
            (a_index, b_index, local_joint_position, target_joint_position, new_rigid_body, target_shape),
        )

        motor_on = jax.random.uniform(_rngs[13], shape=()) < ued_params.motor_on_chance
        joint_colour = jax.random.randint(_rngs[14], shape=(), minval=0, maxval=static_env_params.num_motor_bindings)
        joint_rotation = shape_b.rotation - shape_a.rotation

        motor_speed = jax.random.uniform(
            _rngs[15], shape=(), minval=ued_params.motor_min_speed, maxval=ued_params.motor_max_speed
        )

        motor_power = jax.random.uniform(
            _rngs[16], shape=(), minval=ued_params.motor_min_power, maxval=ued_params.motor_max_power
        )
        wheel_power = jax.random.uniform(
            _rngs[20], shape=(), minval=ued_params.motor_min_power, maxval=ued_params.wheel_max_power
        )

        # High-powered wheels break the physics engine - this is a temporary fix
        motor_power = jax.lax.select(is_rect & is_target_rect, motor_power, wheel_power)

        motor_has_joint_limits = jax.random.uniform(_rngs[17], shape=()) < ued_params.joint_limit_chance
        motor_has_joint_limits &= is_rect & is_target_rect
        joint_limit_min = (
            jax.random.uniform(_rngs[18], shape=(), minval=-ued_params.joint_limit_max, maxval=0.0)
            * motor_has_joint_limits
        )
        joint_limit_max = (
            jax.random.uniform(_rngs[19], shape=(), minval=0.0, maxval=ued_params.joint_limit_max)
            * motor_has_joint_limits
        )

        rjoint = Joint(
            a_index=a_index,
            b_index=b_index,
            a_relative_pos=local_joint_position,
            b_relative_pos=target_joint_position,
            global_position=global_joint_pos,
            active=True,
            motor_speed=motor_speed,
            motor_power=motor_power,
            motor_on=motor_on,
            # colour=joint_colour,
            motor_has_joint_limits=motor_has_joint_limits,
            min_rotation=joint_limit_min,
            max_rotation=joint_limit_max,
            is_fixed_joint=False,
            rotation=0.0,
            acc_impulse=jnp.zeros((2,), dtype=jnp.float32),
            acc_r_impulse=jnp.zeros((), dtype=jnp.float32),
        )

        fjoint = Joint(
            a_index=a_index,
            b_index=b_index,
            a_relative_pos=local_joint_position,
            b_relative_pos=target_joint_position,
            global_position=global_joint_pos,
            active=True,
            rotation=joint_rotation,
            acc_impulse=jnp.zeros((2,), dtype=jnp.float32),
            acc_r_impulse=jnp.zeros((), dtype=jnp.float32),
            is_fixed_joint=True,
            motor_has_joint_limits=False,
            min_rotation=0.0,
            max_rotation=0.0,
            motor_on=False,
            motor_power=0.0,
            motor_speed=0.0,
        )

        state = state.replace(
            polygon=jax.tree.map(
                lambda x, y: jax.lax.select(is_rect, y.at[shape_index].set(x), y), new_rigid_body, state.polygon
            ),
            circle=jax.tree.map(
                lambda x, y: jax.lax.select(jnp.logical_not(is_rect), y.at[shape_index].set(x), y),
                new_rigid_body,
                state.circle,
            ),
            joint=jax.tree.map(
                lambda rj, fj, y: jax.lax.select(is_rjoint, y.at[joint_index].set(rj), y.at[joint_index].set(fj)),
                rjoint,
                fjoint,
                state.joint,
            ),
            polygon_shape_roles=jax.lax.select(
                is_rect,
                state.polygon_shape_roles.at[shape_index].set(shape_role),
                state.polygon_shape_roles,
            ),
            circle_shape_roles=jax.lax.select(
                jnp.logical_not(is_rect),
                state.circle_shape_roles.at[shape_index].set(shape_role),
                state.circle_shape_roles,
            ),
            motor_bindings=state.motor_bindings.at[joint_index].set(joint_colour),
        )

        # We need the new collision matrix.
        state = state.replace(collision_matrix=calculate_collision_matrix(static_env_params, state.joint))
        state = recalculate_mass_and_inertia(state, static_env_params, state.polygon_densities, state.circle_densities)

        # Was this a valid addition?
        # We calculate whether (assuming the possiblity of 360 degree rotation around the joint)
        # both shapes can be visible
        # This is to remove the common degenerate pattern of connected shapes being fully inside each other
        def _get_min_rect_dist(r_id, local_pos):
            rect: RigidBody = jax.tree.map(lambda x: x[r_id], state.polygon)

            half_width = (jnp.max(rect.vertices[:, 0]) - jnp.min(rect.vertices[:, 0])) / 2.0
            half_height = (jnp.max(rect.vertices[:, 1]) - jnp.min(rect.vertices[:, 1])) / 2.0

            dist_x = half_width - jnp.abs(local_pos[0])
            dist_y = half_height - jnp.abs(local_pos[1])

            return jnp.minimum(dist_x, dist_y)

        def _get_max_rect_dist(r_id, local_pos):
            rect: RigidBody = jax.tree.map(lambda x: x[r_id], state.polygon)

            half_width = (jnp.max(rect.vertices[:, 0]) - jnp.min(rect.vertices[:, 0])) / 2.0
            half_height = (jnp.max(rect.vertices[:, 1]) - jnp.min(rect.vertices[:, 1])) / 2.0

            dist_x = jnp.maximum(
                jnp.abs(half_width - local_pos[0]),
                jnp.abs(-half_width - local_pos[0]),
            )

            dist_y = jnp.maximum(
                jnp.abs(half_height - local_pos[1]),
                jnp.abs(-half_height - local_pos[1]),
            )

            return jnp.sqrt(dist_x * dist_x + dist_y * dist_y)

        def are_both_shapes_showing(idx1, idx2, local_pos1, local_pos2):
            def _is_small_shape_showing(small_idx, big_idx, small_local_pos, big_local_pos):
                small_is_poly = small_idx < static_env_params.num_polygons
                big_is_poly = big_idx < static_env_params.num_polygons

                # CC
                cc_result = False

                # CR
                cr_r_dist = _get_min_rect_dist(big_idx, big_local_pos)
                cr_result = (
                    cr_r_dist + ued_params.connect_visibility_min
                    < state.circle.radius[small_idx - static_env_params.num_polygons]
                )

                # RC
                rc_r_dist = _get_max_rect_dist(small_idx, small_local_pos)
                rc_result = (
                    rc_r_dist
                    > state.circle.radius[big_idx - static_env_params.num_polygons] + ued_params.connect_visibility_min
                )

                # RR
                rr_small_dist = _get_max_rect_dist(small_idx, small_local_pos)
                rr_big_dist = _get_min_rect_dist(big_idx, big_local_pos)
                rr_result = rr_small_dist > rr_big_dist + ued_params.connect_visibility_min

                # Select
                return jax.lax.select(
                    small_is_poly,
                    jax.lax.select(big_is_poly, rr_result, rc_result),
                    jax.lax.select(big_is_poly, cr_result, cc_result),
                )

            # Are both shapes showing?
            return _is_small_shape_showing(idx1, idx2, local_pos1, local_pos2) & _is_small_shape_showing(
                idx2, idx1, local_pos2, local_pos1
            )

        valid = are_both_shapes_showing(
            unified_shape_index, unified_target_index, local_joint_position, target_joint_position
        )
        return state, valid

    # To add a connected shape, we must have both at least one existing shape and space
    return jax.lax.cond(
        is_space_for_shape(state) & are_there_shapes_present(state, static_env_params) & is_space_for_joint(state),
        do_add,
        do_dummy,
        rng,
        state,
    )


@partial(jax.jit, static_argnums=(3, 4))
def mutate_add_connected_shape_proper(
    rng,
    state: EnvState,
    env_params: EnvParams,
    static_env_params: StaticEnvParams,
    ued_params: UEDParams,
    force_rjoint: bool = False,
):
    return mutate_add_connected_shape(rng, state, env_params, static_env_params, ued_params, force_rjoint=force_rjoint)[
        0
    ]


@partial(jax.jit, static_argnums=(3, 4))
def mutate_remove_shape(
    rng, state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams, ued_params: UEDParams
):

    can_remove_mask = (
        jnp.concatenate([state.polygon.active, state.circle.active])
        .at[: static_env_params.num_static_fixated_polys]
        .set(False)
    )

    def dummy(rng, state):
        return state

    def do_remove(rng, state: EnvState):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, 2)
        p = can_remove_mask.astype(jnp.float32)
        index_to_remove = jax.random.choice(rngs[0], jnp.arange(can_remove_mask.shape[0]), p=p)
        is_rect = index_to_remove < static_env_params.num_polygons
        state = state.replace(
            polygon=state.polygon.replace(
                active=jax.lax.select(
                    is_rect, state.polygon.active.at[index_to_remove].set(False), state.polygon.active
                )
            ),
            circle=state.circle.replace(
                active=jax.lax.select(
                    jnp.logical_not(is_rect),
                    state.circle.active.at[index_to_remove - static_env_params.num_polygons].set(False),
                    state.circle.active,
                )
            ),
        )
        # We need to now remove any joints connected to this shape
        joints_to_remove = (state.joint.a_index == index_to_remove) | (state.joint.b_index == index_to_remove)

        thrusters_to_remove = state.thruster.object_index == index_to_remove

        state = state.replace(
            joint=state.joint.replace(active=jnp.where(joints_to_remove, False, state.joint.active)),
            thruster=state.thruster.replace(active=jnp.where(thrusters_to_remove, False, state.thruster.active)),
        )
        # Now recalculate collision matrix
        state = state.replace(collision_matrix=calculate_collision_matrix(static_env_params, state.joint))
        return state

    return jax.lax.cond(can_remove_mask.sum() > 0, do_remove, dummy, rng, state)


@partial(jax.jit, static_argnums=(3, 4))
def mutate_remove_joint(
    rng, state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams, ued_params: UEDParams
):
    can_remove_mask = state.joint.active

    def dummy(rng, state):
        return state

    def do_remove(rng, state):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, 2)
        p = can_remove_mask.astype(jnp.float32)
        index_to_remove = jax.random.choice(rngs[0], jnp.arange(can_remove_mask.shape[0]), p=p)
        state = state.replace(joint=state.joint.replace(active=state.joint.active.at[index_to_remove].set(False)))
        # Recalculate collision matrix.
        state = state.replace(collision_matrix=calculate_collision_matrix(static_env_params, state.joint))
        return state

    return jax.lax.cond(can_remove_mask.sum() > 0, do_remove, dummy, rng, state)


@partial(jax.jit, static_argnums=(3, 4))
def mutate_swap_role(
    rng, state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams, ued_params: UEDParams
):
    def _cr(*args):
        return count_roles(*args, include_static_polys=False)

    role_counts = jax.vmap(_cr, (None, None, 0))(state, static_env_params, jnp.arange(4))
    are_there_multiple_roles = (role_counts > 0).sum() > 1

    def dummy(rng, state):
        return state

    def do_swap(rng, state):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, 2)
        all_roles = jnp.concatenate([state.polygon_shape_roles, state.circle_shape_roles])

        p = (
            (jnp.concatenate([state.polygon.active, state.circle.active]))
            .astype(jnp.float32)
            .at[: static_env_params.num_static_fixated_polys]
            .set(0.0)
        )
        shape_idx_a = jax.random.choice(
            rngs[0], jnp.arange(static_env_params.num_polygons + static_env_params.num_circles), p=p
        )
        role_a = all_roles[shape_idx_a]
        p = jnp.where(all_roles == role_a, 0.0, p)
        shape_idx_b = jax.random.choice(
            rngs[1], jnp.arange(static_env_params.num_polygons + static_env_params.num_circles), p=p
        )
        role_b = all_roles[shape_idx_b]
        role_a, role_b = role_b, role_a

        for idx, role in [(shape_idx_a, role_a), (shape_idx_b, role_b)]:
            is_rect = idx < static_env_params.num_polygons
            state = state.replace(
                polygon_shape_roles=jax.lax.select(
                    is_rect, state.polygon_shape_roles.at[idx].set(role), state.polygon_shape_roles
                ),
                circle_shape_roles=jax.lax.select(
                    jnp.logical_not(is_rect),
                    state.circle_shape_roles.at[idx - static_env_params.num_polygons].set(role),
                    state.circle_shape_roles,
                ),
            )
        return state

    return jax.lax.cond(are_there_multiple_roles, do_swap, dummy, rng, state)


@partial(jax.jit, static_argnums=(3, 4))
def mutate_toggle_fixture(
    rng, state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams, ued_params: UEDParams
):
    can_toggle_mask = (
        jnp.concatenate([state.polygon.active, state.circle.active])
        .at[: static_env_params.num_static_fixated_polys]
        .set(False)
    )

    def dummy(rng, state):
        return state

    def do_toggle(rng, state: EnvState):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, 2)
        p = can_toggle_mask.astype(jnp.float32)
        index_to_remove = jax.random.choice(rngs[0], jnp.arange(can_toggle_mask.shape[0]), p=p)
        is_rect = index_to_remove < static_env_params.num_polygons
        is_current_fixed = (
            jax.lax.select(
                is_rect,
                state.polygon.inverse_inertia[index_to_remove],
                state.circle.inverse_inertia[index_to_remove - static_env_params.num_polygons],
            )
            == 0.0
        )

        is_current_fixed = is_current_fixed * 1.0  # if it is fixed, we set it to 1.0 and recalc.
        # If it is not fixed, this is 0.0, and it makes it fixed.

        state = state.replace(
            polygon=state.polygon.replace(
                inverse_inertia=jax.lax.select(
                    is_rect,
                    state.polygon.inverse_inertia.at[index_to_remove].set(is_current_fixed),
                    state.polygon.inverse_inertia,
                ),
                inverse_mass=jax.lax.select(
                    is_rect,
                    state.polygon.inverse_mass.at[index_to_remove].set(is_current_fixed),
                    state.polygon.inverse_mass,
                ),
            ),
            circle=state.circle.replace(
                inverse_inertia=jax.lax.select(
                    jnp.logical_not(is_rect),
                    state.circle.inverse_inertia.at[index_to_remove - static_env_params.num_polygons].set(
                        is_current_fixed
                    ),
                    state.circle.inverse_inertia,
                ),
                inverse_mass=jax.lax.select(
                    jnp.logical_not(is_rect),
                    state.circle.inverse_mass.at[index_to_remove - static_env_params.num_polygons].set(
                        is_current_fixed
                    ),
                    state.circle.inverse_mass,
                ),
            ),
        )

        state = recalculate_mass_and_inertia(state, static_env_params, state.polygon_densities, state.circle_densities)
        return state

    return jax.lax.cond(can_toggle_mask.sum() > 0, do_toggle, dummy, rng, state)


@partial(jax.jit, static_argnums=(3, 4))
def mutate_add_thruster(
    rng, state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams, ued_params: UEDParams
):
    is_fixated = jnp.concatenate([state.polygon.inverse_mass == 0, state.circle.inverse_mass == 0])
    # is_fixated = jnp.zeros_like(is_fixated, dtype=bool)
    is_active = jnp.concatenate([state.polygon.active, state.circle.active])
    can_add_mask = is_active & (~is_fixated)
    can_add_mask = jnp.logical_and(is_active, jnp.logical_not(is_fixated))

    def dummy(rng, state):
        return state

    def do_add(rng, state: EnvState):
        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, 10)
        p = can_add_mask.astype(jnp.float32)
        shape_index = jax.random.choice(_rngs[0], jnp.arange(can_add_mask.shape[0]), p=p)
        thruster_idx = jnp.argmin(state.thruster.active)

        shape = select_shape(state, shape_index, static_env_params)

        position_to_add_thruster = jax.lax.select(
            shape_index < static_env_params.num_polygons,
            random_position_on_polygon(_rngs[1], shape.vertices, shape.n_vertices, static_env_params),
            random_position_on_circle(_rngs[2], shape.radius, on_centre_chance=0.0),
        )

        direction_to_com = ((jax.random.uniform(_rngs[3]) > 0.5) * 2 - 1) * position_to_add_thruster
        direction_to_com = jax.lax.select(
            jnp.linalg.norm(direction_to_com) == 0.0, jnp.array([1.0, 0.0]), direction_to_com
        )

        thruster_angle = jax.lax.select(
            jax.random.uniform(_rngs[4]) < ued_params.thruster_align_com_prob,
            jnp.atan2(direction_to_com[1], direction_to_com[0]),  # test this
            jax.random.uniform(
                _rngs[5],
                (),
            )
            * 2
            * jnp.pi,
        )

        thruster_power = jax.random.uniform(_rngs[6]) * 1.5 + 0.5

        thruster = Thruster(
            object_index=shape_index,
            active=True,
            relative_position=position_to_add_thruster,  # jnp.array([0.0, 0.0]),  # a bit of a hack but reasonable.
            rotation=thruster_angle,  # jax.random.choice(rngs[1], jnp.arange(4) * jnp.pi / 2),
            power=1.0
            / jax.lax.select(shape.inverse_mass == 0, 1.0, shape.inverse_mass)
            * ued_params.thruster_power_multiplier
            * thruster_power,
            global_position=shape.position + jnp.matmul(rmat(shape.rotation), position_to_add_thruster),
        )
        thruster_colour = jax.random.randint(
            _rngs[7], shape=(), minval=0, maxval=static_env_params.num_thruster_bindings
        )

        state = state.replace(
            thruster=jax.tree.map(lambda y, x: y.at[thruster_idx].set(x), state.thruster, thruster),
            thruster_bindings=state.thruster_bindings.at[thruster_idx].set(thruster_colour),
        )

        return state

    return jax.lax.cond(
        jnp.logical_and((can_add_mask.sum() > 0), (jnp.logical_not(state.thruster.active).sum() > 0)),
        do_add,
        dummy,
        rng,
        state,
    )


@partial(jax.jit, static_argnums=(3, 4))
def mutate_change_gravity(
    rng, state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams, ued_params: UEDParams
):
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, 2)
    new_gravity = jax.lax.select(
        jax.random.uniform(rngs[0]) < 0.5,
        jnp.array([0.0, -9.8]),
        jnp.array([0.0, jax.random.uniform(rngs[1], minval=-9.8, maxval=0)]),
    )

    return state.replace(gravity=new_gravity)


@partial(jax.jit, static_argnums=(3, 4))
def mutate_remove_thruster(
    rng, state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams, ued_params: UEDParams
):
    are_there_thrusters = state.thruster.active

    def dummy(rng, state):
        return state

    def do_remove(rng, state):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, 2)
        p = are_there_thrusters.astype(jnp.float32)
        thruster_idx = jax.random.choice(rngs[0], jnp.arange(are_there_thrusters.shape[0]), p=p)
        return state.replace(thruster=state.thruster.replace(active=state.thruster.active.at[thruster_idx].set(False)))

    return jax.lax.cond(are_there_thrusters.sum() > 0, do_remove, dummy, rng, state)


def make_mutate_change_shape_size(env_params, static_env_params):
    do_dummy_step = make_do_dummy_step(env_params, static_env_params)

    @partial(jax.jit, static_argnums=(3, 4))
    def mutate_change_shape_size(
        rng, state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams, ued_params: UEDParams
    ):
        shape_active = jnp.concatenate(
            [state.polygon.active.at[: static_env_params.num_static_fixated_polys].set(False), state.circle.active]
        )

        def dummy(rng, state):
            return state

        def do_change(rng, state):
            rng, _rng = jax.random.split(rng)
            rngs = jax.random.split(_rng, 10)
            p = shape_active.astype(jnp.float32)
            shape_idx = jax.random.choice(rngs[0], jnp.arange(shape_active.shape[0]), p=p)
            is_rect = shape_idx < static_env_params.num_polygons
            vertices, _, radius = sample_dimensions(
                rngs[1], static_env_params, is_rect, ued_params, max_shape_size=ued_params.max_shape_size
            )

            idx_new_top_left = jnp.argmin(vertices[:, 0] * 100 + vertices[:, 1])
            idx_old_top_left = jnp.argmin(
                state.polygon.vertices[shape_idx, :, 0] * 100 + state.polygon.vertices[shape_idx, :, 1]
            )
            scale_rect = (vertices[idx_new_top_left]) / (state.polygon.vertices[shape_idx, idx_old_top_left])
            scale_circle = radius / state.circle.radius[shape_idx - static_env_params.num_polygons]
            vertices = state.polygon.vertices[shape_idx] * scale_rect

            scale = jax.lax.select(
                is_rect,
                scale_rect,
                jnp.array([scale_circle, scale_circle]),
            )

            is_a = ((state.joint.a_index == shape_idx) & state.joint.active)[:, None]
            is_b = ((state.joint.b_index == shape_idx) & state.joint.active)[:, None]
            state = state.replace(
                joint=state.joint.replace(
                    a_relative_pos=(state.joint.a_relative_pos * scale[None]) * is_a
                    + (1 - is_a) * state.joint.a_relative_pos,
                    b_relative_pos=(state.joint.b_relative_pos * scale[None]) * is_b
                    + (1 - is_b) * state.joint.b_relative_pos,
                ),
                polygon=state.polygon.replace(
                    vertices=jax.lax.select(
                        is_rect, state.polygon.vertices.at[shape_idx].set(vertices), state.polygon.vertices
                    ),
                ),
                circle=state.circle.replace(
                    radius=jax.lax.select(
                        jnp.logical_not(is_rect),
                        state.circle.radius.at[shape_idx - static_env_params.num_polygons].set(radius),
                        state.circle.radius,
                    )
                ),
            )

            def _ss(state, _):
                return do_dummy_step(state), None

            state = jax.lax.scan(_ss, state, jnp.arange(5))[0]
            return recalculate_mass_and_inertia(
                state, static_env_params, state.polygon_densities, state.circle_densities
            )

        return jax.lax.cond(shape_active.sum() > 0, do_change, dummy, rng, state)

    return mutate_change_shape_size


@partial(jax.jit, static_argnums=(3, 4))
def mutate_change_shape_location(
    rng, state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams, ued_params: UEDParams
):
    shape_active = jnp.concatenate(
        [state.polygon.active.at[: static_env_params.num_static_fixated_polys].set(False), state.circle.active]
    )

    def dummy(rng, state):
        return state

    def do_change(rng, state):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, 10)
        p = shape_active.astype(jnp.float32)
        shape_idx = jax.random.choice(rngs[0], jnp.arange(shape_active.shape[0]), p=p)
        delta_pos = jax.random.uniform(rngs[1], shape=(2,)) - 0.5  # [-0.5, 0.5]

        positions = jnp.concatenate([state.polygon.position, state.circle.position])

        mask_of_shape_locations_to_change = (
            (state.collision_matrix[shape_idx] == 0).at[: static_env_params.num_static_fixated_polys].set(False)
        )
        # check the new positions, but then maybe revert if any shape becomes out of bounds now.
        new_positions_tentative = positions * (
            1 - mask_of_shape_locations_to_change[:, None]
        ) + mask_of_shape_locations_to_change[:, None] * (positions + delta_pos[None])

        polys = state.polygon
        p_pos = new_positions_tentative[: static_env_params.num_polygons]
        c_pos = new_positions_tentative[static_env_params.num_polygons :]  # state.circle.position
        rad = state.circle.radius
        rect_vertex_mask = jnp.arange(static_env_params.max_polygon_vertices)[None] < polys.n_vertices[:, None]
        rect_mask = polys.active.at[: static_env_params.num_static_fixated_polys].set(False)
        circ_mask = state.circle.active
        # check if new pos maybe goes out of bounds:
        min_x, max_x, min_y, max_y = (
            jnp.minimum(
                jnp.min(
                    p_pos[:, 0] + jnp.min(polys.vertices[:, :, 0], where=rect_vertex_mask, initial=0, axis=1),
                    where=rect_mask,
                    initial=jnp.inf,
                ),
                jnp.min(c_pos[:, 0] - rad, where=circ_mask, initial=jnp.inf),
            ),
            jnp.maximum(
                jnp.max(
                    p_pos[:, 0] + jnp.max(polys.vertices[:, :, 0], where=rect_vertex_mask, initial=0, axis=1),
                    where=rect_mask,
                    initial=-jnp.inf,
                ),
                jnp.max(c_pos[:, 0] + rad, where=circ_mask, initial=-jnp.inf),
            ),
            jnp.minimum(
                jnp.min(
                    p_pos[:, 1] + jnp.min(polys.vertices[:, :, 1], where=rect_vertex_mask, initial=0, axis=1),
                    where=rect_mask,
                    initial=jnp.inf,
                ),
                jnp.min(c_pos[:, 1] - rad, where=circ_mask, initial=jnp.inf),
            ),
            jnp.maximum(
                jnp.max(
                    p_pos[:, 1] + jnp.max(polys.vertices[:, :, 1], where=rect_vertex_mask, initial=0, axis=1),
                    where=rect_mask,
                    initial=-jnp.inf,
                ),
                jnp.max(c_pos[:, 1] + rad, where=circ_mask, initial=-jnp.inf),
            ),
        )

        how_much_oob_x_left = jnp.maximum(0, 0 - min_x)
        how_much_oob_x_right = jnp.maximum(0, max_x - static_env_params.screen_dim[0] / env_params.pixels_per_unit)
        how_much_oob_y_down = jnp.maximum(0, 0.4 - min_y)  # this is for the floor
        how_much_oob_y_up = jnp.maximum(0, max_y - static_env_params.screen_dim[1] / env_params.pixels_per_unit)

        # correct by out of bounds factor
        positions = (
            new_positions_tentative
            + jnp.array(
                [
                    how_much_oob_x_left - how_much_oob_x_right,
                    how_much_oob_y_down - how_much_oob_y_up,
                ]
            )[None]
            * mask_of_shape_locations_to_change[:, None]
        )

        state = state.replace(
            polygon=state.polygon.replace(
                position=positions[: static_env_params.num_polygons],
            ),
            circle=state.circle.replace(
                position=positions[static_env_params.num_polygons :],
            ),
        )
        return recalculate_mass_and_inertia(state, static_env_params, state.polygon_densities, state.circle_densities)

    return jax.lax.cond(shape_active.sum() > 0, do_change, dummy, rng, state)


def make_mutate_change_shape_rotation(env_params, static_env_params):
    do_dummy_step = make_do_dummy_step(env_params, static_env_params)

    @partial(jax.jit, static_argnums=(3, 4))
    def mutate_change_shape_rotation(
        rng, state: EnvState, env_params: EnvParams, static_env_params: StaticEnvParams, ued_params: UEDParams
    ):
        shape_active = jnp.concatenate(
            [state.polygon.active.at[: static_env_params.num_static_fixated_polys].set(False), state.circle.active]
        )

        def dummy(rng, state):
            return state

        def do_change(rng, state):
            rng, _rng = jax.random.split(rng)
            rngs = jax.random.split(_rng, 10)
            p = shape_active.astype(jnp.float32)
            shape_idx = jax.random.choice(rngs[0], jnp.arange(shape_active.shape[0]), p=p)
            is_rect = shape_idx < static_env_params.num_polygons

            rotation_delta = jax.random.uniform(rngs[1], shape=()) * math.pi / 2

            has_fixed_joint_a = (state.joint.a_index == shape_idx) & state.joint.is_fixed_joint & state.joint.active
            has_fixed_joint_b = (state.joint.b_index == shape_idx) & state.joint.is_fixed_joint & state.joint.active

            state = state.replace(
                joint=state.joint.replace(
                    rotation=jax.lax.select(
                        has_fixed_joint_a,
                        state.joint.rotation - rotation_delta,
                        jax.lax.select(
                            has_fixed_joint_b,
                            state.joint.rotation + rotation_delta,
                            state.joint.rotation,
                        ),
                    )
                ),
                polygon=state.polygon.replace(
                    rotation=jax.lax.select(
                        is_rect, state.polygon.rotation.at[shape_idx].add(rotation_delta), state.polygon.rotation
                    ),
                ),
                circle=state.circle.replace(
                    rotation=jax.lax.select(
                        jnp.logical_not(is_rect),
                        state.circle.rotation.at[shape_idx - static_env_params.num_polygons].add(rotation_delta),
                        state.circle.rotation,
                    )
                ),
            )

            def _ss(state, _):
                return do_dummy_step(state), None

            state = jax.lax.scan(_ss, state, jnp.arange(5))[0]
            return recalculate_mass_and_inertia(
                state, static_env_params, state.polygon_densities, state.circle_densities
            )

        return jax.lax.cond(shape_active.sum() > 0, do_change, dummy, rng, state)

    return mutate_change_shape_rotation
