import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax2d.engine import select_shape
from jax2d.maths import rmat
from jaxgl.maths import dist_from_line
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import fragment_shader_edged_quad, nearest_neighbour

from kinetix.environment.env_state import EnvState, StaticEnvParams
from kinetix.render.textures import FJOINT_TEXTURE_6_RGBA, RJOINT_TEXTURE_6_RGBA, THRUSTER_TEXTURE_16_RGBA


def make_render_pixels(
    env_params,
    static_params: StaticEnvParams,
):
    screen_dim = static_params.screen_dim
    downscale = static_params.downscale

    joint_tex_size = 6
    thruster_tex_size = 16

    FIXATED_COLOUR = jnp.array([80, 80, 80])
    JOINT_COLOURS = jnp.array(
        [
            # [0, 0, 255],
            [255, 255, 255],  # yellow
            [255, 255, 0],  # yellow
            [255, 0, 255],  # purple/magenta
            [0, 255, 255],  # cyan
            [255, 153, 51],  # white
        ]
    )

    def colour_thruster_texture(colour):
        return THRUSTER_TEXTURE_16_RGBA.at[:9, :, :3].mul(colour[None, None, :] / 255.0)

    coloured_thruster_textures = jax.vmap(colour_thruster_texture)(JOINT_COLOURS)

    ROLE_COLOURS = jnp.array(
        [
            [160.0, 160.0, 160.0],  # None
            [0.0, 204.0, 0.0],  # Green:    The ball
            [0.0, 102.0, 204.0],  # Blue:   The goal
            [255.0, 102.0, 102.0],  # Red:      Death Objects
        ]
    )

    BACKGROUND_COLOUR = jnp.array([255.0, 255.0, 255.0])

    def _get_colour(shape_role, inverse_inertia):
        base_colour = ROLE_COLOURS[shape_role]
        f = (inverse_inertia == 0) * 1
        is_not_normal = (shape_role != 0) * 1

        return jnp.array(
            [
                base_colour,
                base_colour,
                FIXATED_COLOUR,
                base_colour * 0.5,
            ]
        )[2 * f + is_not_normal]

    # Pixels per unit distance
    ppud = env_params.pixels_per_unit // downscale

    downscaled_screen_dim = (screen_dim[0] // downscale, screen_dim[1] // downscale)

    full_screen_size = (
        downscaled_screen_dim[0] + (static_params.max_shape_size * 2 * ppud),
        downscaled_screen_dim[1] + (static_params.max_shape_size * 2 * ppud),
    )
    cleared_screen = clear_screen(full_screen_size, BACKGROUND_COLOUR)

    def _world_space_to_pixel_space(x):
        return (x + static_params.max_shape_size) * ppud

    def fragment_shader_kinetix_circle(position, current_frag, unit_position, uniform):
        centre, radius, rotation, colour, mask = uniform

        dist = jnp.sqrt(jnp.square(position - centre).sum())
        inside = dist <= radius
        on_edge = dist > radius - 2

        normal = jnp.array([jnp.sin(rotation), -jnp.cos(rotation)])

        dist = dist_from_line(position, centre, centre + normal)

        on_edge |= (dist < 1) & (jnp.dot(normal, position - centre) <= 0)

        fragment = jax.lax.select(on_edge, jnp.zeros(3), colour)

        return jax.lax.select(inside & mask, fragment, current_frag)

    def fragment_shader_kinetix_joint(position, current_frag, unit_position, uniform):
        texture, colour, mask = uniform

        tex_coord = (
            jnp.array(
                [
                    joint_tex_size * unit_position[0],
                    joint_tex_size * unit_position[1],
                ]
            )
            - 0.5
        )

        tex_frag = nearest_neighbour(texture, tex_coord)
        tex_frag = tex_frag.at[3].mul(mask)
        tex_frag = tex_frag.at[:3].mul(colour / 255.0)

        tex_frag = (tex_frag[3] * tex_frag[:3]) + ((1.0 - tex_frag[3]) * current_frag)

        return tex_frag

    thruster_pixel_size = thruster_tex_size // downscale
    thruster_pixel_size_diagonal = (thruster_pixel_size * np.sqrt(2)).astype(jnp.int32) + 1

    def fragment_shader_kinetix_thruster(fragment_position, current_frag, unit_position, uniform):
        thruster_position, rotation, texture, mask = uniform

        tex_position = jnp.matmul(rmat(-rotation), (fragment_position - thruster_position)) / thruster_pixel_size + 0.5

        mask &= (tex_position[0] >= 0) & (tex_position[0] <= 1) & (tex_position[1] >= 0) & (tex_position[1] <= 1)

        eps = 0.001
        tex_coord = (
            jnp.array(
                [
                    thruster_tex_size * tex_position[0],
                    thruster_tex_size * tex_position[1],
                ]
            )
            - 0.5
            + eps
        )

        tex_frag = nearest_neighbour(texture, tex_coord)
        tex_frag = tex_frag.at[3].mul(mask)

        tex_frag = (tex_frag[3] * tex_frag[:3]) + ((1.0 - tex_frag[3]) * current_frag)

        return tex_frag

    patch_size_1d = static_params.max_shape_size * ppud
    patch_size = (patch_size_1d, patch_size_1d)

    circle_renderer = make_renderer(full_screen_size, fragment_shader_kinetix_circle, patch_size, batched=True)
    quad_renderer = make_renderer(full_screen_size, fragment_shader_edged_quad, patch_size, batched=True)
    big_quad_renderer = make_renderer(full_screen_size, fragment_shader_edged_quad, downscaled_screen_dim)

    joint_pixel_size = joint_tex_size // downscale
    joint_renderer = make_renderer(
        full_screen_size, fragment_shader_kinetix_joint, (joint_pixel_size, joint_pixel_size), batched=True
    )

    thruster_renderer = make_renderer(
        full_screen_size,
        fragment_shader_kinetix_thruster,
        (thruster_pixel_size_diagonal, thruster_pixel_size_diagonal),
        batched=True,
    )

    @jax.jit
    def render_pixels(state: EnvState):
        pixels = cleared_screen

        # Floor
        floor_uniform = (
            _world_space_to_pixel_space(state.polygon.position[0, None, :] + state.polygon.vertices[0]),
            _get_colour(state.polygon_shape_roles[0], 0),
            jnp.zeros(3),
            True,
        )

        pixels = big_quad_renderer(pixels, _world_space_to_pixel_space(jnp.zeros(2, dtype=jnp.int32)), floor_uniform)

        # Rectangles
        rectangle_patch_positions = _world_space_to_pixel_space(
            state.polygon.position - (static_params.max_shape_size / 2.0)
        ).astype(jnp.int32)

        rectangle_rmats = jax.vmap(rmat)(state.polygon.rotation)
        rectangle_rmats = jnp.repeat(rectangle_rmats[:, None, :, :], repeats=static_params.max_polygon_vertices, axis=1)
        rectangle_vertices_pixel_space = _world_space_to_pixel_space(
            state.polygon.position[:, None, :] + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, state.polygon.vertices)
        )
        rectangle_colours = jax.vmap(_get_colour)(state.polygon_shape_roles, state.polygon.inverse_mass)
        rectangle_edge_colours = jnp.zeros((static_params.num_polygons, 3))

        rectangle_uniforms = (
            rectangle_vertices_pixel_space,
            rectangle_colours,
            rectangle_edge_colours,
            state.polygon.active,
        )

        pixels = quad_renderer(pixels, rectangle_patch_positions, rectangle_uniforms)

        # Circles
        circle_positions_pixel_space = _world_space_to_pixel_space(state.circle.position)
        circle_radii_pixel_space = state.circle.radius * ppud
        circle_patch_positions = _world_space_to_pixel_space(
            state.circle.position - (static_params.max_shape_size / 2.0)
        ).astype(jnp.int32)

        circle_colours = jax.vmap(_get_colour)(state.circle_shape_roles, state.circle.inverse_mass)

        circle_uniforms = (
            circle_positions_pixel_space,
            circle_radii_pixel_space,
            state.circle.rotation,
            circle_colours,
            state.circle.active,
        )

        pixels = circle_renderer(pixels, circle_patch_positions, circle_uniforms)

        # Joints
        joint_patch_positions = jnp.round(
            _world_space_to_pixel_space(state.joint.global_position) - (joint_pixel_size // 2)
        ).astype(jnp.int32)
        joint_textures = jax.vmap(jax.lax.select, in_axes=(0, None, None))(
            state.joint.is_fixed_joint, FJOINT_TEXTURE_6_RGBA, RJOINT_TEXTURE_6_RGBA
        )
        joint_colours = JOINT_COLOURS[
            (state.motor_bindings + 1) * (state.joint.motor_on & (~state.joint.is_fixed_joint))
        ]

        joint_uniforms = (joint_textures, joint_colours, state.joint.active)

        pixels = joint_renderer(pixels, joint_patch_positions, joint_uniforms)

        # Thrusters
        thruster_positions = jnp.round(_world_space_to_pixel_space(state.thruster.global_position)).astype(jnp.int32)
        thruster_patch_positions = thruster_positions - (thruster_pixel_size_diagonal // 2)
        thruster_textures = coloured_thruster_textures[state.thruster_bindings + 1]
        thruster_rotations = (
            state.thruster.rotation
            + jax.vmap(select_shape, in_axes=(None, 0, None))(
                state, state.thruster.object_index, static_params
            ).rotation
        )
        thruster_uniforms = (thruster_positions, thruster_rotations, thruster_textures, state.thruster.active)

        pixels = thruster_renderer(pixels, thruster_patch_positions, thruster_uniforms)

        # Crop out the sides
        crop_amount = static_params.max_shape_size * ppud
        return pixels[crop_amount:-crop_amount, crop_amount:-crop_amount]

    return render_pixels


@struct.dataclass
class PixelsObservation:
    image: jnp.ndarray
    global_info: jnp.ndarray


def make_render_pixels_rl(env_params, static_params: StaticEnvParams):
    render_fn = make_render_pixels(env_params, static_params)

    def inner(state):
        pixels = render_fn(state) / 255.0
        return PixelsObservation(
            image=pixels,
            global_info=jnp.array([state.gravity[1] / 10.0]),
        )

    return inner
