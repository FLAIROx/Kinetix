import itertools
import time
from timeit import default_timer as tmr

import optax
from PIL import Image
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState

from matplotlib import pyplot as plt

from kinetix.environment.ued.distributions import sample_kinetix_level
from kinetix.models import make_network_from_config
from kinetix.models.actor_critic import ScannedRNN
from kinetix.render.renderer_symbolic_entity import make_render_entities

ss = tmr()

import jax

jax.config.update("jax_compilation_cache_dir", ".cache-location")

import hydra
from omegaconf import OmegaConf

from kinetix.environment.ued.mutators import (
    make_mutate_change_shape_rotation,
    mutate_add_connected_shape,
    mutate_add_shape,
    make_mutate_change_shape_size,
    mutate_change_shape_location,
    mutate_swap_role,
    mutate_remove_shape,
    mutate_remove_joint,
    mutate_toggle_fixture,
    mutate_add_thruster,
    mutate_remove_thruster,
)
from kinetix.environment.ued.ued import make_mutate_env, ALL_MUTATION_FNS
from kinetix.environment.ued.ued_state import UEDParams
from kinetix.environment.ued.util import rectangle_vertices
from kinetix.util.config import generate_params_from_config, normalise_config


import argparse
import os
import sys

sys.path.append("editor")
import tkinter
import tkinter.filedialog
from enum import Enum
from timeit import default_timer as tmr

import jax.numpy as jnp
import numpy as np
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from pygame_widgets.toggle import Toggle

from jax2d.engine import (
    calc_inverse_inertia_circle,
    calc_inverse_inertia_polygon,
    calc_inverse_mass_circle,
    calc_inverse_mass_polygon,
    calculate_collision_matrix,
    recalculate_mass_and_inertia,
    recompute_global_joint_positions,
    select_shape,
)
from jax2d.maths import rmat
from jax2d.sim_state import RigidBody
from kinetix.environment.env import (
    create_empty_env,
    make_kinetix_env_from_name,
)
from kinetix.environment.env_state import EnvParams, EnvState, StaticEnvParams
from kinetix.environment.utils import permute_pcg_state
from kinetix.environment.wrappers import AutoResetWrapper
from kinetix.pcg.pcg import env_state_to_pcg_state, sample_pcg_state
from kinetix.pcg.pcg_state import PCGState
from kinetix.render.renderer_pixels import make_render_pixels
from kinetix.render.textures import (
    CIRCLE_TEXTURE_RGBA,
    EDIT_TEXTURE_RGBA,
    PLAY_TEXTURE_RGBA,
    RECT_TEXTURE_RGBA,
    RJOINT_TEXTURE_RGBA,
    SELECT_TEXTURE_RGBA,
    THRUSTER_TEXTURE_RGBA,
    TRIANGLE_TEXTURE_RGBA,
)
from kinetix.util.saving import (
    expand_pcg_state,
    export_env_state_to_json,
    get_pcg_state_from_json,
    load_from_json_file,
    load_pcg_state_pickle,
    load_world_state_pickle,
    save_pickle,
    load_params_from_wandb_artifact_path,
    load_train_state_from_wandb_artifact_path,
)
from kinetix.util.timing import time_function
from tkinter import Tk

root = Tk()
root.destroy()

ee = tmr()
print(f"Imported in {ee - ss} seconds")

editor = None
outer_timer = tmr()
EMPTY_ENV = False


class ObjectType(Enum):
    POLYGON = 0
    CIRCLE = 1
    JOINT = 2
    THRUSTER = 3


class EditMode(Enum):
    ADD_CIRCLE = 0
    ADD_RECTANGLE = 1
    ADD_JOINT = 2
    SELECT = 3
    ADD_TRIANGLE = 4
    ADD_THRUSTER = 5


TOTAL_DUMMY_STEPS_TO_SNAP = 0
SNAPPING_DIST = 0.1


def select_object(state: EnvState, type: int, index: int):
    if type is None:
        type = ObjectType.POLYGON
    li = {0: state.polygon, 1: state.circle, 2: state.joint, 3: state.thruster}[type.value]
    return jax.tree.map(lambda x: x[index], li)


def snap_to_center(shape: RigidBody, position: jnp.ndarray):
    if jnp.linalg.norm(shape.position - position) < SNAPPING_DIST:
        return shape.position
    return position


def snap_to_polygon_center_line(polygon: RigidBody, position: jnp.ndarray):
    # Snap to the center line
    r = rmat(polygon.rotation)
    x = jnp.matmul(r, position - polygon.position)
    if jnp.abs(x[0]) < SNAPPING_DIST:
        x = x.at[0].set(0.0)
    if jnp.abs(x[1]) < SNAPPING_DIST:
        x = x.at[1].set(0.0)
    x = jnp.matmul(r.transpose(), x)
    return x + polygon.position


def snap_to_circle_center_line(circle: RigidBody, position: jnp.ndarray):
    # Snap to the center line, i.e. on the edge of the circle, if the position is close enough to directly below the circle, etc., snap the position to that
    x = position - circle.position
    if jnp.linalg.norm(x) < SNAPPING_DIST:
        return circle.position
    angle = (jnp.arctan2(x[1], x[0]) + 2 * jnp.pi) % (2 * jnp.pi)

    for i in range(0, 8):
        if jnp.abs(angle - i * jnp.pi / 4) < jnp.radians(25):  # 25 degrees
            angle = i * jnp.pi / 4
            break
    x = jnp.array([jnp.cos(angle), jnp.sin(angle)]) * circle.radius
    return x + circle.position


def prompt_file(save=False):
    dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "worlds")
    """Create a Tk file dialog and cleanup when finished"""
    top = tkinter.Tk()
    top.withdraw()  # hide window
    if save:
        file_name = tkinter.filedialog.asksaveasfilename(parent=top, initialdir=dir)
    else:
        file_name = tkinter.filedialog.askopenfilename(parent=top, initialdir=dir)
    top.destroy()
    return file_name


def get_numeric_key_pressed(pygame_events, is_mod=False):
    for event in pygame_events:
        if not is_mod:
            if event.type == pygame.KEYDOWN and event.unicode.isdigit():
                return int(event.unicode)
        else:
            if event.type == pygame.KEYDOWN:
                pass
            if event.type == pygame.KEYDOWN and event.key in [pygame.K_0 + i for i in range(10)]:
                return int(event.key - pygame.K_0)
    return None


def new_env(static_env_params):
    return create_empty_env(static_env_params)


myrng = jax.random.PRNGKey(0)


def make_reset_function(static_env_params):
    def reset(rng):
        return env_state_to_pcg_state(create_empty_env(static_env_params))

    return reset


def new_pcg_env(static_env_params):
    global myrng
    if EMPTY_ENV:
        env_state = create_empty_env(static_env_params)
    else:
        return get_pcg_state_from_json("worlds/l/h0_angrybirds.json")
    return env_state_to_pcg_state(env_state)


class Editor:
    def __init__(self, env, env_params, config, upscale=1):
        self.env = env
        self.upscale = upscale
        self.env_params = env_params
        self.static_env_params = env.static_env_params
        self.ued_params = UEDParams()

        self.side_panel_width = env.static_env_params.screen_dim[0] // 3

        self.rng = jax.random.PRNGKey(0)
        self.config = config
        self.pcg_state = new_pcg_env(self.static_env_params)

        self.rng, _rng = jax.random.split(self.rng)
        self.play_state = sample_pcg_state(_rng, self.pcg_state, self.env_params, self.static_env_params)
        self.last_played_level = None

        self.pygame_events = []

        self.mutate_world = make_mutate_env(env.static_env_params, env_params, self.ued_params)

        self.num_triangle_clicks = 0
        self.triangle_order = jnp.array([0, 1, 2])

        # Init rendering
        pygame.init()
        pygame.key.set_repeat(250, 75)

        self.screen_surface = pygame.display.set_mode(
            tuple(
                (t + extra) * self.upscale
                for t, extra in zip(self.static_env_params.screen_dim, (self.side_panel_width, 0))
            ),
            display=0,
        )
        self.all_widgets = {}
        self._setup_side_panel()
        self.has_done_action = False

        self._setup_rendering(self.static_env_params, env_params)

        self._render_edit_overlay_fn = jax.jit(self._render_edit_overlay)
        self._step_fn = jax.jit(env.step)

        # Agent
        if self.config["agent_taking_actions"]:
            self._entity_renderer = jax.jit(make_render_entities(env_params, self.env.static_env_params))
            self.network = make_network_from_config(env, env_params, config)

            rng = jax.random.PRNGKey(0)
            dones = jnp.zeros((config["num_train_envs"]), dtype=jnp.bool_)
            rng, _rng = jax.random.split(rng)
            init_hstate = ScannedRNN.initialize_carry(config["num_train_envs"])
            obsv = self._entity_renderer(self.play_state)
            obsv = jax.tree.map(lambda x: jnp.repeat(x[None, ...], repeats=config["num_train_envs"], axis=0), obsv)
            init_x = jax.tree.map(lambda x: x[None, ...], (obsv, dones))
            network_params = self.network.init(_rng, init_hstate, init_x)

            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(config["lr"], eps=1e-5),
            )
            self.train_state = TrainState.create(
                apply_fn=self.network.apply,
                params=network_params,
                tx=tx,
            )

            self.train_state = load_train_state_from_wandb_artifact_path(self.train_state, config["agent_wandb_path"])

            self.apply_fn = jax.jit(self.network.apply)

        # JIT Compile
        def _jit_step():
            rng = jax.random.PRNGKey(0)
            ans = self._step_fn(
                rng,
                self.env.reset_env_to_level(rng, self.play_state, self.env_params)[1],
                jnp.zeros(
                    env.static_env_params.num_motor_bindings + env.static_env_params.num_thruster_bindings, dtype=int
                ),
                self.env_params,
            )

        def _jit_render():
            self._render_fn(self.play_state)
            self._render_fn_edit(self.play_state)

        time_function(_jit_step, "_jit_step")
        time_function(_jit_render, "_jit_render")

        # self._step_fn(rng, self.play_state, 0, self.env_params)

        # Editing
        self.is_editing = True

        self.edit_shape_mode = EditMode.ADD_CIRCLE
        self.creating_shape = False
        self.create_shape_position = jnp.array([0.0, 0.0])
        self.creating_shape_index = 0
        self.selected_shape_index = -1
        self.selected_shape_type = ObjectType.POLYGON

        self.all_selected_shapes = []

        self.rng = jax.random.PRNGKey(0)
        time_function(self._jit, "self._jit")
        self._put_state_values_into_gui(self.pcg_state)

        self.mutate_change_shape_size = make_mutate_change_shape_size(self.env_params, self.static_env_params)
        self.mutate_change_shape_rotation = make_mutate_change_shape_rotation(self.env_params, self.static_env_params)

    def _setup_rendering(self, static_env_params: StaticEnvParams, env_params: EnvParams):
        def _make_render(should_do_edit_additions=False):
            def _render(env_state):
                side_panel = self._render_side_panel()
                render_pixels = make_render_pixels(params=env_params, static_params=static_env_params)
                pixels = render_pixels(
                    env_state,
                )
                pixels = jnp.concatenate([side_panel, pixels], axis=0)

                pixels = jnp.repeat(pixels, repeats=static_env_params.downscale * self.upscale, axis=0)
                pixels = jnp.repeat(pixels, repeats=static_env_params.downscale * self.upscale, axis=1)

                return pixels[:, ::-1, :]

            return _render

        def _make_screenshot_render():
            def _render(env_state):
                px_upscale = 4

                static_params = static_env_params.replace(screen_dim=(500 * px_upscale, 500 * px_upscale))
                ss_env_params = env_params.replace(
                    pixels_per_unit=100 * px_upscale,
                )

                render_pixels = make_render_pixels(
                    params=ss_env_params,
                    static_params=static_params,
                    render_rjoint_sectors=False,
                    pixel_upscale=2 * px_upscale,
                )
                pixels = render_pixels(
                    env_state,
                )

                return pixels[:, ::-1, :]

            return _render

        self._render_fn_edit = jax.jit(_make_render(True))
        self._render_fn = jax.jit(_make_render(False))
        self._render_fn_screenshot = jax.jit(_make_screenshot_render())

    def _jit(self):
        self._get_circles_on_mouse(self.pcg_state.env_state)
        self._get_polygons_on_mouse(self.pcg_state.env_state)
        self._get_revolute_joints_on_mouse(self.pcg_state.env_state)
        self._get_thrusters_on_mouse(self.pcg_state.env_state)
        self.pygame_events = list(pygame.event.get())
        self._handle_events(do_dummy=True)

        state = self.play_state
        for mutation_fn in ALL_MUTATION_FNS:
            mutation_fn(jax.random.PRNGKey(0), state, self.env_params, self.static_env_params, self.ued_params)

    def update(self, rng):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.has_done_action = False
                    self.is_editing = not self.is_editing
                    if not self.is_editing:
                        self.pcg_state = self._discard_shape_being_created(self.pcg_state)
                        self.pcg_state = self._reset_select_shape(self.pcg_state)
                        self.pcg_state = self.pcg_state.replace(
                            env_state=self.pcg_state.env_state.replace(
                                collision_matrix=calculate_collision_matrix(
                                    self.static_env_params, self.pcg_state.env_state.joint
                                ),
                            ),
                            env_state_pcg_mask=self.pcg_state.env_state_pcg_mask.replace(
                                collision_matrix=jnp.zeros_like(self.pcg_state.env_state_pcg_mask.collision_matrix)
                            ),
                        )
                        self.rng, _rng = jax.random.split(self.rng)
                        self.play_state = sample_pcg_state(
                            _rng, self.pcg_state, self.env_params, self.static_env_params
                        )
                        self.last_played_level = self.play_state
                elif event.key == pygame.K_s and not self.is_editing:
                    self.take_screenshot()

        if self.is_editing:
            self.pcg_state = self.edit()
        else:
            rng, _rng = jax.random.split(rng)
            # action = []
            action = jnp.zeros(
                self.env.static_env_params.num_motor_bindings + self.env.static_env_params.num_thruster_bindings,
                dtype=jnp.int32,
            )

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = action.at[0].set(1)
            if keys[pygame.K_RIGHT]:
                action = action.at[0].set(2)
            if keys[pygame.K_UP]:
                action = action.at[1].set(1)
            if keys[pygame.K_DOWN]:
                action = action.at[1].set(2)

            if keys[pygame.K_1]:
                action = action.at[0 + self.env.static_env_params.num_motor_bindings].set(1)
            if keys[pygame.K_2]:
                action = action.at[1 + self.env.static_env_params.num_motor_bindings].set(1)
            if keys[pygame.K_3]:
                action = action.at[2 + self.env.static_env_params.num_motor_bindings].set(1)
            if keys[pygame.K_4]:
                action = action.at[3 + self.env.static_env_params.num_motor_bindings].set(1)
            # if self.has_done_action: action = action * 0
            self.has_done_action = self.has_done_action | (action != 0).any()

            if self.config["agent_taking_actions"]:
                obs = self._entity_renderer(self.play_state)
                obs = jax.tree.map(lambda x: x[None, ...], obs)

                last_done = jnp.zeros((1, 1), dtype=bool)
                ac_in = (jax.tree.map(lambda x: x[np.newaxis, :], obs), last_done[np.newaxis, :])
                hstate = ScannedRNN.initialize_carry(1)
                hstate, pi, value = self.apply_fn(self.train_state.params, hstate, ac_in)
                rng, _rng = jax.random.split(rng)
                action = pi.sample(seed=_rng)
                action = action[0, 0]

            _rng, __rng = jax.random.split(_rng)
            obs, self.play_state, reward, done, info = self._step_fn(
                _rng, self.env.reset_to_level(__rng, self.play_state, self.env_params)[1], action, self.env_params
            )
            if done:
                self.rng, _rng = jax.random.split(self.rng)
                self.play_state = sample_pcg_state(_rng, self.pcg_state, self.env_params, self.static_env_params)
        state_to_render = self.pcg_state.env_state if self.is_editing else self.play_state

        self.render(state_to_render)
        self._handle_events()

        # Update screen
        pygame.display.flip()

    def take_screenshot(self):
        print("screenshot!")

        pixels = self._render_fn_screenshot(self.play_state)
        mtime = round(time.time() * 1000)

        pixels = pixels.transpose((1, 0, 2))

        # Black border
        border_thickness = 5
        pixels = pixels.at[:, :border_thickness].set(0.0)
        pixels = pixels.at[:, -border_thickness:].set(0.0)
        pixels = pixels.at[:border_thickness, :].set(0.0)
        pixels = pixels.at[-border_thickness:, :].set(0.0)

        im = Image.fromarray(np.array(pixels.astype(jnp.uint8)))
        im.save(f"results/screenshot_{str(mtime)}.png")

    def _get_selected_shape_global_indices(self):
        def _idx(idx, type):
            if type == ObjectType.CIRCLE:
                return idx + self.static_env_params.num_polygons
            return idx

        indices_to_use = jnp.array([_idx(idx, type) for idx, type in self.all_selected_shapes])
        return indices_to_use

    # flag1
    def _handle_events(self, do_dummy=False):
        pygame_widgets.update(self.pygame_events)
        if do_dummy or self.selected_shape_index < 0:
            gravity_main = self.all_widgets[None]["sldGravity"].getValue()
            gravity_max = self.all_widgets[None]["sldMaxGravity"].getValue()
            gravity_main, gravity_max = min(gravity_main, gravity_max), max(gravity_main, gravity_max)
            gravity_pcg_mask = self.all_widgets[None]["pcgTglGravity"].getValue()

            def _set_single_global(state, gravity):
                return state.replace(
                    gravity=state.gravity.at[1].set(gravity),
                )

            env_state = _set_single_global(self.pcg_state.env_state, gravity_main)
            env_state_max = _set_single_global(self.pcg_state.env_state_max, gravity_max)
            env_state_pcg_mask = _set_single_global(self.pcg_state.env_state_pcg_mask, gravity_pcg_mask)
            if not do_dummy:
                self.pcg_state = self.pcg_state.replace(
                    env_state=env_state,
                    env_state_max=env_state_max,
                    env_state_pcg_mask=env_state_pcg_mask,
                )

        if self.edit_shape_mode == EditMode.SELECT or do_dummy:  # is on the hand.
            if do_dummy or len(self.all_selected_shapes) > 1:
                # this processes the tying together.
                indices_to_use = self._get_selected_shape_global_indices()
                if len(indices_to_use) > 1:
                    toggle_val = self.all_widgets["TIE_TOGETHER"]["tglTieTogether"].getValue()
                    idxs = itertools.product(indices_to_use, indices_to_use)
                    idxs_a = []
                    idxs_b = []
                    for i, j in idxs:
                        idxs_a.append(i)
                        idxs_b.append(j)
                    idxs = jnp.array(idxs_a), jnp.array(idxs_b)
                    if toggle_val:
                        self.pcg_state = self.pcg_state.replace(
                            tied_together=self.pcg_state.tied_together.at[idxs[0], idxs[1]].set(True)
                        )
                    else:
                        self.pcg_state = self.pcg_state.replace(
                            tied_together=self.pcg_state.tied_together.at[idxs[0], idxs[1]].set(False)
                        )
            if self.selected_shape_index < 0 and not do_dummy:
                return
            if do_dummy or self.selected_shape_type in [ObjectType.POLYGON, ObjectType.CIRCLE]:  # rigidbody
                shape_main = select_object(
                    self.pcg_state.env_state, self.selected_shape_type, self.selected_shape_index
                )
                shape_max = select_object(
                    self.pcg_state.env_state_max, self.selected_shape_type, self.selected_shape_index
                )

                parent_container_main = (
                    self.pcg_state.env_state.circle
                    if self.selected_shape_type == ObjectType.CIRCLE
                    else self.pcg_state.env_state.polygon
                )
                parent_container_max = (
                    self.pcg_state.env_state_max.circle
                    if self.selected_shape_type == ObjectType.CIRCLE
                    else self.pcg_state.env_state_max.polygon
                )
                parent_container_pcg_mask = (
                    self.pcg_state.env_state_pcg_mask.circle
                    if self.selected_shape_type == ObjectType.CIRCLE
                    else self.pcg_state.env_state_pcg_mask.polygon
                )

                new_density_main = self.all_widgets[self.selected_shape_type]["sldDensity"].getValue()
                new_density_max = self.all_widgets[self.selected_shape_type]["sldMaxDensity"].getValue()
                density_pcg_mask = self.all_widgets[self.selected_shape_type]["pcgTglDensity"].getValue()
                if density_pcg_mask:
                    new_density_main, new_density_max = min(new_density_main, new_density_max), max(
                        new_density_main, new_density_max
                    )
                fixated = self.all_widgets[self.selected_shape_type]["tglFixate"].getValue()

                fix_val = 0.0 if fixated else 1.0

                def _density_calcs(base, new_density):
                    inv_mass = jax.lax.select(
                        self.selected_shape_type == ObjectType.CIRCLE,
                        calc_inverse_mass_circle(base.radius, new_density),
                        calc_inverse_mass_polygon(base.vertices, base.n_vertices, self.static_env_params, new_density)[
                            0
                        ],
                    )
                    inv_inertia = jax.lax.select(
                        self.selected_shape_type == ObjectType.CIRCLE,
                        calc_inverse_inertia_circle(base.radius, new_density),
                        calc_inverse_inertia_polygon(
                            base.vertices, base.n_vertices, self.static_env_params, new_density
                        ),
                    )

                    return inv_mass, inv_inertia

                inv_mass_main, inv_inertia_main = _density_calcs(shape_main, new_density_main)
                inv_mass_max, inv_inertia_max = _density_calcs(shape_max, new_density_max)

                friction_main = self.all_widgets[self.selected_shape_type]["sldFriction"].getValue()
                friction_max = self.all_widgets[self.selected_shape_type]["sldMaxFriction"].getValue()
                friction_pcg_mask = self.all_widgets[self.selected_shape_type]["pcgTglFriction"].getValue()
                if friction_pcg_mask:
                    friction_main, friction_max = min(friction_main, friction_max), max(friction_main, friction_max)

                restitution = self.all_widgets[self.selected_shape_type]["sldRestitution"].getValue()

                position_main = jnp.array(
                    [
                        self.all_widgets[self.selected_shape_type]["sldPosition_X"].getValue(),
                        self.all_widgets[self.selected_shape_type]["sldPosition_Y"].getValue(),
                    ]
                )

                position_max = jnp.array(
                    [
                        self.all_widgets[self.selected_shape_type]["sldMaxPosition_X"].getValue(),
                        self.all_widgets[self.selected_shape_type]["sldMaxPosition_Y"].getValue(),
                    ]
                )
                position_pcg_mask = self.all_widgets[self.selected_shape_type]["pcgTglPosition_X"].getValue()
                if position_pcg_mask:
                    position_main, position_max = jnp.minimum(position_main, position_max), jnp.maximum(
                        position_main, position_max
                    )

                rotation_main = self.all_widgets[self.selected_shape_type]["sldRotation"].getValue()
                rotation_max = self.all_widgets[self.selected_shape_type]["sldMaxRotation"].getValue()
                rotation_pcg_mask = self.all_widgets[self.selected_shape_type]["pcgTglRotation"].getValue()
                if rotation_pcg_mask:
                    rotation_main, rotation_max = min(rotation_main, rotation_max), max(rotation_main, rotation_max)

                velocity_main = jnp.array(
                    [
                        self.all_widgets[self.selected_shape_type]["sldVelocity_X"].getValue(),
                        self.all_widgets[self.selected_shape_type]["sldVelocity_Y"].getValue(),
                    ]
                )

                velocity_max = jnp.array(
                    [
                        self.all_widgets[self.selected_shape_type]["sldMaxVelocity_X"].getValue(),
                        self.all_widgets[self.selected_shape_type]["sldMaxVelocity_Y"].getValue(),
                    ]
                )
                velocity_main, velocity_max = jnp.minimum(velocity_main, velocity_max), jnp.maximum(
                    velocity_main, velocity_max
                )
                velocity_pcg_mask = self.all_widgets[self.selected_shape_type]["pcgTglVelocity_X"].getValue()

                angular_velocity_main = self.all_widgets[self.selected_shape_type]["sldAngular_Velocity"].getValue()
                angular_velocity_max = self.all_widgets[self.selected_shape_type]["sldMaxAngular_Velocity"].getValue()
                angular_velocity_pcg_mask = self.all_widgets[self.selected_shape_type][
                    "pcgTglAngular_Velocity"
                ].getValue()
                if angular_velocity_pcg_mask:
                    angular_velocity_main, angular_velocity_max = min(angular_velocity_main, angular_velocity_max), max(
                        angular_velocity_main, angular_velocity_max
                    )

                # Circle stuff
                radius_main, radius_max, radius_pcg_mask = None, None, None
                if self.selected_shape_type == ObjectType.CIRCLE:
                    radius_main = self.all_widgets[self.selected_shape_type]["sldRadius"].getValue()
                    radius_max = self.all_widgets[self.selected_shape_type]["sldMaxRadius"].getValue()
                    radius_pcg_mask = self.all_widgets[self.selected_shape_type]["pcgTglRadius"].getValue()

                # Poly stuff
                vertices_main, vertices_max, vertices_pcg_mask = None, None, None
                if self.selected_shape_type == ObjectType.POLYGON:
                    # Triangle
                    new_size_main = self.all_widgets[self.selected_shape_type]["sldSize"].getValue()
                    new_size_max = self.all_widgets[self.selected_shape_type]["sldMaxSize"].getValue()

                    current_size = jnp.abs(self.pcg_state.env_state.polygon.vertices[self.selected_shape_index]).max()
                    scale_main = new_size_main / current_size
                    scale_max = new_size_max / current_size

                    vertices_main = self.pcg_state.env_state.polygon.vertices[self.selected_shape_index] * scale_main
                    vertices_max = self.pcg_state.env_state.polygon.vertices[self.selected_shape_index] * scale_max
                    vertices_pcg_mask = (
                        jnp.ones_like(vertices_main, dtype=bool)
                        * self.all_widgets[self.selected_shape_type]["pcgTglSize"].getValue()
                    )

                def _set_single_state_rbody(
                    state,
                    parent_container,
                    density,
                    inv_mass,
                    inv_inertia,
                    friction,
                    position,
                    radius,
                    rotation,
                    velocity,
                    angular_velocity,
                    vertices,
                ):
                    new = {
                        "friction": parent_container.friction.at[self.selected_shape_index].set(friction),
                        "collision_mode": parent_container.collision_mode.at[self.selected_shape_index].set(
                            int(self.all_widgets[self.selected_shape_type]["sldCollidability"].getValue())
                        ),
                        "inverse_mass": parent_container.inverse_mass.at[self.selected_shape_index].set(
                            inv_mass * fix_val
                        ),
                        "inverse_inertia": parent_container.inverse_inertia.at[self.selected_shape_index].set(
                            inv_inertia * fix_val
                        ),
                        "position": parent_container.position.at[self.selected_shape_index].set(position),
                        "rotation": parent_container.rotation.at[self.selected_shape_index].set(rotation),
                        "velocity": parent_container.velocity.at[self.selected_shape_index].set(velocity),
                        "angular_velocity": parent_container.angular_velocity.at[self.selected_shape_index].set(
                            angular_velocity
                        ),
                        "restitution": parent_container.restitution.at[self.selected_shape_index].set(restitution),
                    }

                    if self.selected_shape_type == ObjectType.CIRCLE:
                        state = state.replace(
                            circle=state.circle.replace(
                                **new,
                                radius=parent_container.radius.at[self.selected_shape_index].set(radius),
                            ),
                            circle_shape_roles=state.circle_shape_roles.at[self.selected_shape_index].set(
                                int(self.all_widgets[self.selected_shape_type]["sldRole"].getValue())
                            ),
                            circle_densities=state.circle_densities.at[self.selected_shape_index].set(density),
                        )
                    else:
                        state = state.replace(
                            polygon=state.polygon.replace(
                                **new,
                                vertices=parent_container.vertices.at[self.selected_shape_index].set(vertices),
                            ),
                            polygon_shape_roles=state.polygon_shape_roles.at[self.selected_shape_index].set(
                                int(self.all_widgets[self.selected_shape_type]["sldRole"].getValue())
                            ),
                            polygon_densities=state.polygon_densities.at[self.selected_shape_index].set(density),
                        )

                    return state

                position_delta = position_main - shape_main.position
                env_state = _set_single_state_rbody(
                    self.pcg_state.env_state,
                    parent_container_main,
                    new_density_main,
                    inv_mass_main,
                    inv_inertia_main,
                    friction_main,
                    position_main,
                    radius_main,
                    rotation_main,
                    velocity_main,
                    angular_velocity_main,
                    vertices_main,
                )
                env_state_max = _set_single_state_rbody(
                    self.pcg_state.env_state_max,
                    parent_container_max,
                    new_density_max,
                    inv_mass_max,
                    inv_inertia_max,
                    friction_max,
                    position_max,
                    radius_max,
                    rotation_max,
                    velocity_max,
                    angular_velocity_max,
                    vertices_max,
                )
                env_state_pcg_mask = _set_single_state_rbody(
                    self.pcg_state.env_state_pcg_mask,
                    parent_container_pcg_mask,
                    density_pcg_mask,
                    density_pcg_mask,
                    density_pcg_mask,
                    friction_pcg_mask,
                    position_pcg_mask,
                    radius_pcg_mask,
                    rotation_pcg_mask,
                    velocity_pcg_mask,
                    angular_velocity_pcg_mask,
                    vertices_pcg_mask,
                )

                # Now, we must also set all of the tied shape's positions

                correct_index = self.selected_shape_index + (
                    self.static_env_params.num_polygons if self.selected_shape_type == ObjectType.CIRCLE else 0
                )

                nonzero_indices = set(
                    jnp.nonzero(self.pcg_state.tied_together[correct_index].at[correct_index].set(False))[0].tolist()
                )
                for idx in nonzero_indices:
                    if idx < self.static_env_params.num_polygons:
                        env_state = env_state.replace(
                            polygon=env_state.polygon.replace(
                                position=env_state.polygon.position.at[idx].set(
                                    env_state.polygon.position[idx] + position_delta
                                )
                            )
                        )
                    else:
                        idx = idx - self.static_env_params.num_polygons
                        env_state = env_state.replace(
                            circle=env_state.circle.replace(
                                position=env_state.circle.position.at[idx].set(
                                    env_state.circle.position[idx] + position_delta
                                )
                            )
                        )

                if not do_dummy:
                    self.pcg_state = PCGState(
                        env_state=env_state,
                        env_state_max=env_state_max,
                        env_state_pcg_mask=env_state_pcg_mask,
                        tied_together=self.pcg_state.tied_together,
                    )

            if do_dummy or self.selected_shape_type == ObjectType.JOINT:  # rjoint
                speed_main = self.all_widgets[ObjectType.JOINT]["sldSpeed"].getValue()
                speed_max = self.all_widgets[ObjectType.JOINT]["sldMaxSpeed"].getValue()
                speed_pcg_mask = self.all_widgets[ObjectType.JOINT]["pcgTglSpeed"].getValue()
                if speed_pcg_mask:
                    speed_main, speed_max = min(speed_main, speed_max), max(speed_main, speed_max)

                power_main = self.all_widgets[ObjectType.JOINT]["sldPower"].getValue()
                power_max = self.all_widgets[ObjectType.JOINT]["sldMaxPower"].getValue()
                power_pcg_mask = self.all_widgets[ObjectType.JOINT]["pcgTglPower"].getValue()
                if power_pcg_mask:
                    power_main, power_max = min(power_main, power_max), max(power_main, power_max)

                motor_binding_val_min = int(self.all_widgets[ObjectType.JOINT]["sldColour"].getValue())
                motor_binding_val_max = int(self.all_widgets[ObjectType.JOINT]["sldMaxColour"].getValue())
                colour_pcg_mask = self.all_widgets[ObjectType.JOINT]["pcgTglColour"].getValue()
                if colour_pcg_mask:
                    motor_binding_val_min, motor_binding_val_max = min(
                        motor_binding_val_min, motor_binding_val_max
                    ), max(motor_binding_val_min, motor_binding_val_max)

                auto_motor_val = self.all_widgets[ObjectType.JOINT]["tglAutoMotor"].getValue()
                joint_limits_val = self.all_widgets[ObjectType.JOINT]["tglJointLimits"].getValue()
                is_fixed_val = self.all_widgets[ObjectType.JOINT]["tglIsFixedJoint"].getValue()
                is_motor_on = self.all_widgets[ObjectType.JOINT]["tglIsMotorOn"].getValue()

                min_rot_val = jnp.deg2rad(self.all_widgets[ObjectType.JOINT]["sldMin_Rotation"].getValue())
                max_rot_val = jnp.deg2rad(self.all_widgets[ObjectType.JOINT]["sldMax_Rotation"].getValue())
                # ensure the min is less than the max
                min_rot_val, max_rot_val = min(min_rot_val, max_rot_val), max(min_rot_val, max_rot_val)

                def _set_single_state_joint(state, speed, power, colour):
                    state = state.replace(
                        joint=state.joint.replace(
                            motor_speed=state.joint.motor_speed.at[self.selected_shape_index].set(speed),
                            motor_power=state.joint.motor_power.at[self.selected_shape_index].set(power),
                            motor_has_joint_limits=state.joint.motor_has_joint_limits.at[self.selected_shape_index].set(
                                joint_limits_val
                            ),
                            min_rotation=state.joint.min_rotation.at[self.selected_shape_index].set(min_rot_val),
                            max_rotation=state.joint.max_rotation.at[self.selected_shape_index].set(max_rot_val),
                            is_fixed_joint=state.joint.is_fixed_joint.at[self.selected_shape_index].set(is_fixed_val),
                            motor_on=state.joint.motor_on.at[self.selected_shape_index].set(is_motor_on),
                        ),
                        motor_bindings=state.motor_bindings.at[self.selected_shape_index].set(colour),
                        motor_auto=state.motor_auto.at[self.selected_shape_index].set(auto_motor_val),
                    )
                    return state

                env_state = _set_single_state_joint(
                    self.pcg_state.env_state,
                    speed_main,
                    power_main,
                    motor_binding_val_min,
                )

                env_state_max = _set_single_state_joint(
                    self.pcg_state.env_state_max,
                    speed_max,
                    power_max,
                    motor_binding_val_max,
                )

                env_state_pcg_mask = _set_single_state_joint(
                    self.pcg_state.env_state_pcg_mask, speed_pcg_mask, power_pcg_mask, colour_pcg_mask
                )
                if not do_dummy:
                    self.pcg_state = self.pcg_state.replace(
                        env_state=env_state,
                        env_state_max=env_state_max,
                        env_state_pcg_mask=env_state_pcg_mask,
                    )

            if do_dummy or self.selected_shape_type == ObjectType.THRUSTER:  # thruster
                power_main = self.all_widgets[ObjectType.THRUSTER]["sldPower"].getValue()
                power_max = self.all_widgets[ObjectType.THRUSTER]["sldMaxPower"].getValue()
                power_pcg_mask = self.all_widgets[ObjectType.THRUSTER]["pcgTglPower"].getValue()
                if power_pcg_mask:
                    power_main, power_max = min(power_main, power_max), max(power_main, power_max)

                def _set_single_state_thruster(state, power):
                    return state.replace(
                        thruster=state.thruster.replace(
                            power=state.thruster.power.at[self.selected_shape_index].set(power),
                        ),
                        thruster_bindings=state.thruster_bindings.at[self.selected_shape_index].set(
                            int(self.all_widgets[ObjectType.THRUSTER]["sldColour"].getValue())
                        ),
                    )

                env_state = _set_single_state_thruster(self.pcg_state.env_state, power_main)
                env_state_max = _set_single_state_thruster(self.pcg_state.env_state_max, power_max)
                env_state_pcg_mask = _set_single_state_thruster(self.pcg_state.env_state_pcg_mask, power_pcg_mask)
                if not do_dummy:
                    self.pcg_state = self.pcg_state.replace(
                        env_state=env_state,
                        env_state_max=env_state_max,
                        env_state_pcg_mask=env_state_pcg_mask,
                    )

    # flag2
    def _put_state_values_into_gui(self, pcg_state=None):
        def _set_toggle_val(toggle, val):
            if toggle.getValue() != val:
                toggle.toggle()

        def _enable_slider(slider):
            slider.enable()
            slider.colour = (200, 200, 200)
            slider.handleColour = (0, 0, 0)

        def _disable_slider(slider):
            slider.disable()
            slider.colour = (255, 0, 0)
            slider.handleColour = (255, 0, 0)

        if pcg_state is None:
            # state = self.edit_state
            raise ValueError

        def pcg_text(main_val, max_val, pcg_mask):
            if pcg_mask:
                return f"{main_val:.2f} - {max_val:.2f}"
            else:
                return f"{main_val:.2f}"

        # global ones
        gravity_pcg_mask = pcg_state.env_state_pcg_mask.gravity[1]
        self.all_widgets[None]["lblGravity"].setText(
            f"Gravity: {pcg_text(pcg_state.env_state.gravity[1], pcg_state.env_state_max.gravity[1], pcg_state.env_state_pcg_mask.gravity[1])}"
        )
        self.all_widgets[None]["sldGravity"].setValue(pcg_state.env_state.gravity[1])
        self.all_widgets[None]["sldMaxGravity"].setValue(pcg_state.env_state_max.gravity[1])
        if not gravity_pcg_mask:
            _disable_slider(self.all_widgets[None]["sldMaxGravity"])
        else:
            _enable_slider(self.all_widgets[None]["sldMaxGravity"])

        if self.edit_shape_mode != EditMode.SELECT or self.selected_shape_index < 0:
            return

        obj_main = select_object(pcg_state.env_state, self.selected_shape_type, self.selected_shape_index)
        obj_max = select_object(pcg_state.env_state_max, self.selected_shape_type, self.selected_shape_index)
        obj_pcg_mask = select_object(pcg_state.env_state_pcg_mask, self.selected_shape_type, self.selected_shape_index)

        if len(self.all_selected_shapes) > 1:
            indices_to_use = self._get_selected_shape_global_indices()
            are_all_tied = pcg_state.tied_together[indices_to_use.min(), indices_to_use].all()
            _set_toggle_val(self.all_widgets["TIE_TOGETHER"]["tglTieTogether"], are_all_tied)

        if self.selected_shape_type == ObjectType.JOINT:
            self.all_widgets[ObjectType.JOINT]["lblSpeed"].setText(
                f"Speed: {pcg_text(obj_main.motor_speed, obj_max.motor_speed, obj_pcg_mask.motor_speed)}"
            )
            self.all_widgets[ObjectType.JOINT]["sldSpeed"].setValue(obj_main.motor_speed)
            self.all_widgets[ObjectType.JOINT]["sldMaxSpeed"].setValue(obj_max.motor_speed)
            _set_toggle_val(self.all_widgets[ObjectType.JOINT]["pcgTglSpeed"], obj_pcg_mask.motor_speed)
            if obj_pcg_mask.motor_speed:
                _enable_slider(self.all_widgets[ObjectType.JOINT]["sldMaxSpeed"])
            else:
                _disable_slider(self.all_widgets[ObjectType.JOINT]["sldMaxSpeed"])

            self.all_widgets[ObjectType.JOINT]["lblPower"].setText(
                f"Power: {pcg_text(obj_main.motor_power, obj_max.motor_power, obj_pcg_mask.motor_power)}"
            )
            self.all_widgets[ObjectType.JOINT]["sldPower"].setValue(obj_main.motor_power)
            self.all_widgets[ObjectType.JOINT]["sldMaxPower"].setValue(obj_max.motor_power)
            _set_toggle_val(self.all_widgets[ObjectType.JOINT]["pcgTglPower"], obj_pcg_mask.motor_power)
            if obj_pcg_mask.motor_power:
                _enable_slider(self.all_widgets[ObjectType.JOINT]["sldMaxPower"])
            else:
                _disable_slider(self.all_widgets[ObjectType.JOINT]["sldMaxPower"])

            self.all_widgets[ObjectType.JOINT]["lblColour"].setText(
                f"Colour: {pcg_state.env_state.motor_bindings[self.selected_shape_index]}"
            )
            self.all_widgets[ObjectType.JOINT]["sldColour"].setValue(
                pcg_state.env_state.motor_bindings[self.selected_shape_index]
            )

            self.all_widgets[ObjectType.JOINT]["sldMaxColour"].setValue(
                pcg_state.env_state_max.motor_bindings[self.selected_shape_index]
            )

            colour_pcg_mask = pcg_state.env_state_pcg_mask.motor_bindings[self.selected_shape_index]
            if not colour_pcg_mask:
                _disable_slider(self.all_widgets[ObjectType.JOINT]["sldMaxColour"])
            else:
                _enable_slider(self.all_widgets[ObjectType.JOINT]["sldMaxColour"])

            self.all_widgets[ObjectType.JOINT]["lblJointLimits"].setText(
                f"Joint Limits: {obj_main.motor_has_joint_limits}"
            )
            widget_motor_has_joint_limits = self.all_widgets[ObjectType.JOINT]["tglJointLimits"].getValue()
            if obj_main.motor_has_joint_limits != widget_motor_has_joint_limits:  # Update the toggle
                self.all_widgets[ObjectType.JOINT]["tglJointLimits"].toggle()

            mini, maxi = jnp.rad2deg(obj_main.min_rotation), jnp.rad2deg(obj_main.max_rotation)
            self.all_widgets[ObjectType.JOINT]["lblMin_Rotation"].setText(f"Min Rot: {int(mini)}")
            self.all_widgets[ObjectType.JOINT]["sldMin_Rotation"].setValue(mini)
            self.all_widgets[ObjectType.JOINT]["lblMax_Rotation"].setText(f"Max Rot: {int(maxi)}")
            self.all_widgets[ObjectType.JOINT]["sldMax_Rotation"].setValue(maxi)

            if not obj_main.motor_has_joint_limits:
                for k in ["min_rotation", "max_rotation"]:
                    self.all_widgets[self.selected_shape_type][f"sld{k.title()}"].disable()
                    self.all_widgets[self.selected_shape_type][f"sld{k.title()}"].colour = (255, 0, 0)
                    self.all_widgets[self.selected_shape_type][f"sld{k.title()}"].handleColour = (255, 0, 0)
            else:
                for k in ["min_rotation", "max_rotation"]:
                    self.all_widgets[self.selected_shape_type][f"sld{k.title()}"].enable()
                    self.all_widgets[self.selected_shape_type][f"sld{k.title()}"].colour = (200, 200, 200)
                    self.all_widgets[self.selected_shape_type][f"sld{k.title()}"].handleColour = (0, 0, 0)

            self.all_widgets[ObjectType.JOINT]["lblAutoMotor"].setText(
                f"Auto: {pcg_state.env_state.motor_auto[self.selected_shape_index]}"
            )
            widget_is_auto_motor = self.all_widgets[ObjectType.JOINT]["tglAutoMotor"].getValue()
            if pcg_state.env_state.motor_auto[self.selected_shape_index] != widget_is_auto_motor:  # Update the toggle
                self.all_widgets[ObjectType.JOINT]["tglAutoMotor"].toggle()

            self.all_widgets[ObjectType.JOINT]["lblIsFixedJoint"].setText(f"Fixed: {obj_main.is_fixed_joint}")
            widget_is_motor_on = self.all_widgets[ObjectType.JOINT]["tglIsFixedJoint"].getValue()
            if obj_main.is_fixed_joint != widget_is_motor_on:  # Update the toggle
                self.all_widgets[ObjectType.JOINT]["tglIsFixedJoint"].toggle()

            self.all_widgets[ObjectType.JOINT]["lblIsMotorOn"].setText(f"Motor On: {obj_main.motor_on}")
            widget_is_motor_on = self.all_widgets[ObjectType.JOINT]["tglIsMotorOn"].getValue()
            if obj_main.motor_on != widget_is_motor_on:  # Update the toggle
                self.all_widgets[ObjectType.JOINT]["tglIsMotorOn"].toggle()

        elif self.selected_shape_type == ObjectType.THRUSTER:
            # thruster
            self.all_widgets[ObjectType.THRUSTER]["lblPower"].setText(
                f"Power: {pcg_text(obj_main.power, obj_max.power, obj_pcg_mask.power)}"
            )
            self.all_widgets[ObjectType.THRUSTER]["sldPower"].setValue(obj_main.power)
            self.all_widgets[ObjectType.THRUSTER]["sldMaxPower"].setValue(obj_max.power)
            _set_toggle_val(self.all_widgets[ObjectType.THRUSTER]["pcgTglPower"], obj_pcg_mask.power)
            if obj_pcg_mask.power:
                _enable_slider(self.all_widgets[ObjectType.THRUSTER]["sldMaxPower"])
            else:
                _disable_slider(self.all_widgets[ObjectType.THRUSTER]["sldMaxPower"])
            self.all_widgets[ObjectType.THRUSTER]["sldColour"].setValue(
                pcg_state.env_state.thruster_bindings[self.selected_shape_index]
            )
            self.all_widgets[ObjectType.THRUSTER]["lblColour"].setText(
                f"Colour: {pcg_state.env_state.thruster_bindings[self.selected_shape_index]}"
            )

        elif self.selected_shape_type in [ObjectType.POLYGON, ObjectType.CIRCLE]:
            # rigidbody

            # Position
            # We use the mask for position_x for the entire position vector
            self.all_widgets[self.selected_shape_type]["lblPosition_X"].setText(
                f"Position X: {pcg_text(obj_main.position[0], obj_max.position[0], obj_pcg_mask.position[0])}"
            )
            self.all_widgets[self.selected_shape_type]["sldPosition_X"].setValue(obj_main.position[0])
            self.all_widgets[self.selected_shape_type]["sldMaxPosition_X"].setValue(obj_max.position[0])
            _set_toggle_val(self.all_widgets[self.selected_shape_type]["pcgTglPosition_X"], obj_pcg_mask.position[0])

            self.all_widgets[self.selected_shape_type]["lblPosition_Y"].setText(
                f"Position Y: {pcg_text(obj_main.position[1], obj_max.position[1], obj_pcg_mask.position[0])}"
            )
            self.all_widgets[self.selected_shape_type]["sldPosition_Y"].setValue(obj_main.position[1])
            self.all_widgets[self.selected_shape_type]["sldMaxPosition_Y"].setValue(obj_max.position[1])

            if obj_pcg_mask.position[0]:
                _enable_slider(self.all_widgets[self.selected_shape_type]["sldMaxPosition_X"])
                _enable_slider(self.all_widgets[self.selected_shape_type]["sldMaxPosition_Y"])
            else:
                _disable_slider(self.all_widgets[self.selected_shape_type]["sldMaxPosition_X"])
                _disable_slider(self.all_widgets[self.selected_shape_type]["sldMaxPosition_Y"])

            # Velocity
            # We use the mask for velocity_x for the entire velocity vector
            self.all_widgets[self.selected_shape_type]["lblVelocity_X"].setText(
                f"Velocity X: {pcg_text(obj_main.velocity[0], obj_max.velocity[0], obj_pcg_mask.velocity[0])}"
            )
            self.all_widgets[self.selected_shape_type]["sldVelocity_X"].setValue(obj_main.velocity[0])
            self.all_widgets[self.selected_shape_type]["sldMaxVelocity_X"].setValue(obj_max.velocity[0])
            _set_toggle_val(self.all_widgets[self.selected_shape_type]["pcgTglVelocity_X"], obj_pcg_mask.velocity[0])

            self.all_widgets[self.selected_shape_type]["lblVelocity_Y"].setText(
                f"Velocity Y: {pcg_text(obj_main.velocity[1], obj_max.velocity[1], obj_pcg_mask.velocity[0])}"
            )
            self.all_widgets[self.selected_shape_type]["sldVelocity_Y"].setValue(obj_main.velocity[1])
            self.all_widgets[self.selected_shape_type]["sldMaxVelocity_Y"].setValue(obj_max.velocity[1])

            if obj_pcg_mask.velocity[0]:
                _enable_slider(self.all_widgets[self.selected_shape_type]["sldMaxVelocity_X"])
                _enable_slider(self.all_widgets[self.selected_shape_type]["sldMaxVelocity_Y"])
            else:
                _disable_slider(self.all_widgets[self.selected_shape_type]["sldMaxVelocity_X"])
                _disable_slider(self.all_widgets[self.selected_shape_type]["sldMaxVelocity_Y"])

            # Density
            is_fixated = obj_main.inverse_mass == 0

            def _calc_density(state):
                if self.selected_shape_type == ObjectType.POLYGON:
                    return state.polygon_densities[self.selected_shape_index]
                elif self.selected_shape_type == ObjectType.CIRCLE:
                    return state.circle_densities[self.selected_shape_index]
                else:
                    raise ValueError

            density_main = _calc_density(pcg_state.env_state)
            density_max = _calc_density(pcg_state.env_state_max)
            density_pcg_mask = obj_pcg_mask.inverse_mass

            self.all_widgets[self.selected_shape_type]["lblDensity"].setText(
                f"Density: {pcg_text(density_main, density_max, density_pcg_mask)}"
            )
            self.all_widgets[self.selected_shape_type]["sldDensity"].setValue(density_main)
            self.all_widgets[self.selected_shape_type]["sldMaxDensity"].setValue(density_max)
            _set_toggle_val(self.all_widgets[self.selected_shape_type]["pcgTglDensity"], density_pcg_mask)
            if is_fixated:
                _disable_slider(self.all_widgets[self.selected_shape_type]["sldDensity"])
            else:
                _enable_slider(self.all_widgets[self.selected_shape_type]["sldDensity"])

            if is_fixated or (not density_pcg_mask):
                _disable_slider(self.all_widgets[self.selected_shape_type]["sldMaxDensity"])
            else:
                _enable_slider(self.all_widgets[self.selected_shape_type]["sldMaxDensity"])

            # Friction
            self.all_widgets[self.selected_shape_type]["lblFriction"].setText(
                f"Friction: {pcg_text(obj_main.friction, obj_max.friction, obj_pcg_mask.friction)}"
            )
            self.all_widgets[self.selected_shape_type]["sldFriction"].setValue(obj_main.friction)
            self.all_widgets[self.selected_shape_type]["sldMaxFriction"].setValue(obj_max.friction)
            _set_toggle_val(self.all_widgets[self.selected_shape_type]["pcgTglFriction"], obj_pcg_mask.friction)

            if not obj_pcg_mask.friction:
                _disable_slider(self.all_widgets[self.selected_shape_type]["sldMaxFriction"])
            else:
                _enable_slider(self.all_widgets[self.selected_shape_type]["sldMaxFriction"])

            # Restitution
            self.all_widgets[self.selected_shape_type]["lblRestitution"].setText(
                f"Restitution: {obj_main.restitution:.2f}"
            )
            self.all_widgets[self.selected_shape_type]["sldRestitution"].setValue(obj_main.restitution)

            # Rotation
            self.all_widgets[self.selected_shape_type]["lblRotation"].setText(
                f"Rotation: {pcg_text(obj_main.rotation, obj_max.rotation, obj_pcg_mask.rotation)}"
            )
            self.all_widgets[self.selected_shape_type]["sldRotation"].setValue(obj_main.rotation)
            self.all_widgets[self.selected_shape_type]["sldMaxRotation"].setValue(obj_max.rotation)
            _set_toggle_val(self.all_widgets[self.selected_shape_type]["pcgTglRotation"], obj_pcg_mask.rotation)

            if not obj_pcg_mask.rotation:
                _disable_slider(self.all_widgets[self.selected_shape_type]["sldMaxRotation"])
            else:
                _enable_slider(self.all_widgets[self.selected_shape_type]["sldMaxRotation"])

            # Angular_Velocity
            self.all_widgets[self.selected_shape_type]["lblAngular_Velocity"].setText(
                f"Angular_Velocity: {pcg_text(obj_main.angular_velocity, obj_max.angular_velocity, obj_pcg_mask.angular_velocity)}"
            )
            self.all_widgets[self.selected_shape_type]["sldAngular_Velocity"].setValue(obj_main.angular_velocity)
            self.all_widgets[self.selected_shape_type]["sldMaxAngular_Velocity"].setValue(obj_max.angular_velocity)
            _set_toggle_val(
                self.all_widgets[self.selected_shape_type]["pcgTglAngular_Velocity"], obj_pcg_mask.angular_velocity
            )

            if not obj_pcg_mask.angular_velocity:
                _disable_slider(self.all_widgets[self.selected_shape_type]["sldMaxAngular_Velocity"])
            else:
                _enable_slider(self.all_widgets[self.selected_shape_type]["sldMaxAngular_Velocity"])

            # Collision mode
            self.all_widgets[self.selected_shape_type]["lblCollidability"].setText(
                f"Collidability: {obj_main.collision_mode}"
            )
            self.all_widgets[self.selected_shape_type]["sldCollidability"].setValue(obj_main.collision_mode)

            # Shape role
            if self.selected_shape_type == ObjectType.POLYGON:
                shape_role = pcg_state.env_state.polygon_shape_roles[self.selected_shape_index]
            else:
                shape_role = pcg_state.env_state.circle_shape_roles[self.selected_shape_index]

            self.all_widgets[self.selected_shape_type]["sldRole"].setValue(shape_role)
            self.all_widgets[self.selected_shape_type]["lblRole"].setText(f"Role: {shape_role}")

            # Fixate
            self.all_widgets[self.selected_shape_type]["lblFixate"].setText(f"Fixate: {is_fixated}")
            widget_is_fixed = self.all_widgets[self.selected_shape_type]["tglFixate"].getValue()
            if is_fixated != widget_is_fixed:  # Update the toggle
                self.all_widgets[self.selected_shape_type]["tglFixate"].toggle()

            # Radius
            if self.selected_shape_type == ObjectType.CIRCLE:
                self.all_widgets[self.selected_shape_type]["lblRadius"].setText(
                    f"Radius: {pcg_text(obj_main.radius, obj_max.radius, obj_pcg_mask.radius)}"
                )
                self.all_widgets[self.selected_shape_type]["sldRadius"].setValue(obj_main.radius)
                self.all_widgets[self.selected_shape_type]["sldMaxRadius"].setValue(obj_max.radius)
                _set_toggle_val(self.all_widgets[self.selected_shape_type]["pcgTglRadius"], obj_pcg_mask.radius)

                if not obj_pcg_mask.radius:
                    _disable_slider(self.all_widgets[self.selected_shape_type]["sldMaxRadius"])
                else:
                    _enable_slider(self.all_widgets[self.selected_shape_type]["sldMaxRadius"])

            elif self.selected_shape_type == ObjectType.POLYGON:
                size_main = jnp.abs(obj_main.vertices).max()
                size_max = jnp.abs(obj_max.vertices).max()
                size_pcg_mask = obj_pcg_mask.vertices.any()

                self.all_widgets[self.selected_shape_type]["lblSize"].setText(
                    f"Size: {pcg_text(size_main, size_max, size_pcg_mask)}"
                )
                self.all_widgets[self.selected_shape_type]["sldSize"].setValue(size_main)
                self.all_widgets[self.selected_shape_type]["sldMaxSize"].setValue(size_max)
                _set_toggle_val(self.all_widgets[self.selected_shape_type]["pcgTglSize"], size_pcg_mask)

                if not size_pcg_mask:
                    _disable_slider(self.all_widgets[self.selected_shape_type]["sldMaxSize"])
                else:
                    _enable_slider(self.all_widgets[self.selected_shape_type]["sldMaxSize"])

    def _render_side_panel(self):
        arr = jnp.ones((self.side_panel_width, self.static_env_params.screen_dim[1], 3)) * (
            jnp.array([135.0, 206.0, 235.0])[None, None] + 20
        )
        return arr

    def make_label_and_slider(
        self,
        start_y,
        label_text,
        slider_min=0.0,
        slider_max=1.0,
        slider_step=0.05,
        font_size=18,
        is_toggle=False,
        is_pcg=False,
        add_pcg_toggle=True,
    ):
        Wl = round(self.W * 0.7)
        pcg_lbl = None
        pcg_toggle = None
        widget_max = None

        label = TextBox(
            self.screen_surface,
            self.MARGIN,
            start_y,
            Wl,
            20,
            fontSize=font_size,
            margin=0,
            placeholderText=label_text,
            font=pygame.font.SysFont("sans-serif", font_size),
        )
        label.disable()  # Act as label instead of textbox
        if is_toggle:
            widget = Toggle(self.screen_surface, self.W // 2 - 20, start_y + 23, 20, 13)
        else:
            widget = Slider(
                self.screen_surface,
                self.MARGIN,
                start_y + 23,
                Wl,
                13,
                min=slider_min,
                max=slider_max,
                step=slider_step,
            )

            if is_pcg:
                widget_max = Slider(
                    self.screen_surface,
                    self.MARGIN,
                    start_y + 50,
                    Wl,
                    13,
                    min=slider_min,
                    max=slider_max,
                    step=slider_step,
                )

                pcg_lbl = TextBox(
                    self.screen_surface,
                    self.MARGIN + Wl + 30,
                    start_y,
                    60,
                    20,
                    fontSize=font_size,
                    margin=0,
                    placeholderText="PCG",
                    font=pygame.font.SysFont("sans-serif", font_size),
                )
                if add_pcg_toggle:
                    pcg_toggle = Toggle(self.screen_surface, Wl + 80, start_y + 23, 20, 13)

        return label, widget, (pcg_toggle, pcg_lbl, widget_max)

    def _setup_side_panel(self):

        W = self.W = int(self.side_panel_width * self.upscale * 0.8)
        MARGIN = self.MARGIN = (self.side_panel_width * self.upscale - W) // 2

        # global values
        G = {}
        gravity_label, gravity_slider, (pcg_toggle, pcg_lbl, slider_max) = self.make_label_and_slider(
            150, "Gravity", -20.0, 0.0, 0.2, is_pcg=True
        )
        G["lblGravity"] = gravity_label
        G["sldGravity"] = gravity_slider
        G["sldMaxGravity"] = slider_max
        G["pcgLblGravity"] = pcg_lbl
        G["pcgTglGravity"] = pcg_toggle

        # thruster values
        T = {}
        thruster_power_label, thruster_power_slider, (pcg_toggle, pcg_lbl, slider_max) = self.make_label_and_slider(
            150, "Power", slider_max=3.0, is_pcg=True
        )
        T["lblPower"] = thruster_power_label
        T["sldPower"] = thruster_power_slider
        T["sldMaxPower"] = slider_max
        T["pcgLblPower"] = pcg_lbl
        T["pcgTglPower"] = pcg_toggle
        thruster_colour_label, thruster_colour_slider, _ = self.make_label_and_slider(
            250, "Colour", 0, self.static_env_params.num_thruster_bindings - 1, 1
        )
        T["lblColour"] = thruster_colour_label
        T["sldColour"] = thruster_colour_slider

        # joints
        D = {}
        for i, (name, (mini, maxi, step), is_pcg) in enumerate(
            zip(
                ["speed", "power", "colour", "min_rotation", "max_rotation"],
                [
                    (-3, 3, 0.05),
                    (0, 3, 0.05),
                    (0, self.static_env_params.num_motor_bindings - 1, 1),
                    (-180, 180, 5),
                    (-180, 180, 5),
                ],
                [True, True, True, False, False],
            )
        ):
            label, slider, (pcg_toggle, pcg_lbl, slider_max) = self.make_label_and_slider(
                150 + 80 * i,
                f"Motor {name.title()}",
                slider_min=mini,
                slider_max=maxi,
                slider_step=step,
                is_pcg=is_pcg,
            )
            D["lbl" + name.title()] = label
            D["sld" + name.title()] = slider
            if is_pcg:
                D["sldMax" + name.title()] = slider_max
                D["pcgLbl" + name.title()] = pcg_lbl
                D["pcgTgl" + name.title()] = pcg_toggle

        label, toggle, _ = self.make_label_and_slider(150 + 80 * 5, "Joint Limits", is_toggle=True)
        D["lblJointLimits"] = label
        D["tglJointLimits"] = toggle

        label, toggle, _ = self.make_label_and_slider(150 + 80 * 6, "Auto", is_toggle=True)
        D["lblAutoMotor"] = label
        D["tglAutoMotor"] = toggle

        label, toggle, _ = self.make_label_and_slider(150 + 80 * 7, "Fixed", is_toggle=True)
        D["lblIsFixedJoint"] = label
        D["tglIsFixedJoint"] = toggle

        label, toggle, _ = self.make_label_and_slider(150 + 80 * 8, "Motor On", is_toggle=True)
        D["lblIsMotorOn"] = label
        D["tglIsMotorOn"] = toggle

        def _create_rigid_body_base_gui():
            D_rigid = {}
            # rigidbodies
            total_toggles = 0
            total_non_toggles = 0
            for i, (name, bounds, is_pcg, add_pcg_toggle) in enumerate(
                zip(
                    [
                        "position_x",
                        "position_y",
                        "rotation",
                        "velocity_x",
                        "velocity_y",
                        "angular_velocity",
                        "density",
                        "friction",
                        "restitution",
                        "collidability",
                        "role",
                    ],
                    [
                        (0, 5.0, 0.01),
                        (0, 5.0, 0.01),
                        (-2 * jnp.pi, 2 * jnp.pi, 0.01),
                        (-10.0, 10.0, 0.1),
                        (-10.0, 10.0, 0.1),
                        (-6, 6.0, 0.01),
                        (0.1, 5.0, 0.1),
                        (0.02, 1.0, 0.02),
                        (0.0, 0.8, 0.02),
                        (0, 2, 1),
                        (0, 3, 1),
                    ],
                    [True, True, True, True, True, True, True, True, False, False, False],
                    [True, False, True, True, False, True, True, True, False, False, False],
                )
            ):
                location = 50 + 80 * total_toggles + 40 * total_non_toggles
                label, slider, (pcg_toggle, pcg_lbl, slider_max) = self.make_label_and_slider(
                    location, name.title(), *bounds, is_pcg=is_pcg, add_pcg_toggle=add_pcg_toggle
                )
                total_toggles += is_pcg
                total_non_toggles += not is_pcg
                D_rigid["lbl" + name.title()] = label
                D_rigid["sld" + name.title()] = slider

                if is_pcg:
                    D_rigid["sldMax" + name.title()] = slider_max
                    if add_pcg_toggle:
                        D_rigid["pcgLbl" + name.title()] = pcg_lbl
                        D_rigid["pcgTgl" + name.title()] = pcg_toggle

            location = 50 + 80 * total_toggles + 40 * total_non_toggles
            # toggles:
            label, toggle, _ = self.make_label_and_slider(location, "Fixate", is_toggle=True)
            D_rigid["lblFixate"] = label
            D_rigid["tglFixate"] = toggle

            return D_rigid, location

        # Circle extras
        D_circle, location = _create_rigid_body_base_gui()
        label, slider, (pcg_toggle, pcg_lbl, slider_max) = self.make_label_and_slider(
            location + 40, "Radius", slider_min=0.1, slider_max=1.0, slider_step=0.02, is_pcg=True, add_pcg_toggle=True
        )
        D_circle["lblRadius"] = label
        D_circle["sldRadius"] = slider
        D_circle["sldMaxRadius"] = slider_max
        D_circle["pcgLblRadius"] = pcg_lbl
        D_circle["pcgTglRadius"] = pcg_toggle

        # Polygon extras
        D_poly, location = _create_rigid_body_base_gui()
        label, slider, (pcg_toggle, pcg_lbl, slider_max) = self.make_label_and_slider(
            location + 40, "Size", slider_min=0.1, slider_max=2.0, slider_step=0.02, is_pcg=True, add_pcg_toggle=True
        )
        D_poly["lblSize"] = label
        D_poly["sldSize"] = slider
        D_poly["sldMaxSize"] = slider_max
        D_poly["pcgLblSize"] = pcg_lbl
        D_poly["pcgTglSize"] = pcg_toggle

        label, toggle, _ = self.make_label_and_slider(150 + 80 * 5, "Tie Positions Together", is_toggle=True)
        TIE_TOGETHER = {
            "lblTieTogether": label,
            "tglTieTogether": toggle,
        }

        self.all_widgets = {
            ObjectType.THRUSTER: T,
            ObjectType.JOINT: D,
            "GENERAL": {
                "lblGeneral": TextBox(
                    self.screen_surface,
                    MARGIN,
                    10,
                    W,
                    30,
                    fontSize=20,
                    margin=0,
                    placeholderText="General",
                    font=pygame.font.SysFont("sans-serif", 35),
                ),
            },
            ObjectType.POLYGON: D_poly,
            ObjectType.CIRCLE: D_circle,
            None: G,
            "TIE_TOGETHER": TIE_TOGETHER,
        }

        self._hide_all_widgets()

    def _render_edit_overlay(self, pixels, is_editing, edit_shape_mode):
        is_editing_texture = jax.lax.select(is_editing, EDIT_TEXTURE_RGBA, PLAY_TEXTURE_RGBA)
        is_editing_texture = jnp.repeat(jnp.repeat(is_editing_texture, self.upscale, axis=0), self.upscale, axis=1)

        offset = self.side_panel_width * self.upscale
        w = 64 * self.upscale
        offset2 = int(w * 1.25)
        offset_y = 16 * self.upscale

        play_tex_with_background = (1 - is_editing_texture[:, :, 3:]) * pixels[
            offset + 0 : offset + w, 0:w
        ] + is_editing_texture[:, :, 3:] * is_editing_texture[:, :, :3]
        pixels = pixels.at[offset : offset + w, 0:w].set(play_tex_with_background)

        edit_shape_texture = jax.lax.switch(
            edit_shape_mode,
            [
                lambda: CIRCLE_TEXTURE_RGBA,
                lambda: RECT_TEXTURE_RGBA,
                lambda: RJOINT_TEXTURE_RGBA,
                lambda: SELECT_TEXTURE_RGBA,
                lambda: TRIANGLE_TEXTURE_RGBA,
                lambda: THRUSTER_TEXTURE_RGBA,
            ],
        )

        edit_shape_texture = jnp.repeat(jnp.repeat(edit_shape_texture, self.upscale, axis=0), self.upscale, axis=1)

        edit_shape_texture_alpha = edit_shape_texture[:, :, 3:] * is_editing

        w = 32 * self.upscale
        edit_shape_texture_with_background = (1 - edit_shape_texture_alpha) * pixels[
            offset + offset2 : offset + offset2 + w, offset_y : offset_y + w
        ] + edit_shape_texture_alpha * edit_shape_texture[:, :, :3]
        pixels = pixels.at[offset + offset2 : offset + offset2 + w, offset_y : offset_y + w].set(
            edit_shape_texture_with_background
        )

        return pixels

    def edit(self):
        self.rng, _rng = jax.random.split(self.rng)
        pcg_state = self.pcg_state

        left_click = False
        right_click = False

        keys = []
        keys_up_this_frame = set()
        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if (
                    event.key == pygame.K_s
                    and (pygame.key.get_mods() & pygame.KMOD_CTRL)
                    and (pygame.key.get_mods() & pygame.KMOD_SHIFT)
                ):
                    filename = prompt_file(save=True)
                    if filename:
                        filename += ".level.pkl"
                        save_pickle(filename, self.last_played_level)
                        print(f"Saved last sampled level to {filename}")
                elif event.key == pygame.K_s and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    pcg_state = self._reset_select_shape(pcg_state)
                    filename = prompt_file(save=True)
                    if filename:
                        if filename.endswith(".json"):
                            export_env_state_to_json(
                                filename, pcg_state.env_state, self.static_env_params, self.env_params
                            )
                        elif not filename.endswith(".pcg.pkl"):
                            filename += ".pcg.pkl"
                            save_pickle(filename, pcg_state)
                        print(f"Saved PCG state to {filename}")
                elif event.key == pygame.K_o and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    filename = prompt_file(save=False)
                    if filename:
                        self._reset_select_shape(pcg_state)
                        if filename.endswith(".pcg.pkl"):
                            pcg_state = load_pcg_state_pickle(filename)
                            pcg_state = expand_pcg_state(pcg_state, self.static_env_params)
                            print(f"Loaded PCG state from {filename}")
                        elif filename.endswith(".level.pkl"):
                            env_state = load_world_state_pickle(filename)
                            pcg_state = env_state_to_pcg_state(env_state)
                            print(f"Converted level state to PCG state from {filename}")
                        elif filename.endswith(".json"):
                            env_state, new_static_env_params, new_env_params = load_from_json_file(filename)
                            self._update_params(new_static_env_params, new_env_params)
                            pcg_state = env_state_to_pcg_state(env_state)
                    self._reset_triangles()
                elif event.key == pygame.K_n and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    self._reset_select_shape(pcg_state)
                    pcg_state = new_pcg_env(self.static_env_params)
                    self._reset_triangles()
                else:
                    keys.append(event.key)
            elif event.type == pygame.KEYUP:
                keys_up_this_frame.add(event.key)
            if event.type == pygame.MOUSEBUTTONDOWN and self._get_mouse_position_world_space()[0] >= 0:
                if event.button == 1:
                    left_click = True
                if event.button == 3:
                    right_click = True

            if event.type == pygame.MOUSEWHEEL:
                pcg_state = self._handle_scroll_wheel(pcg_state, event.y)

        if self.selected_shape_index == -1:
            num = get_numeric_key_pressed(self.pygame_events)
            if num is not None:
                self.edit_shape_mode = EditMode(num % len(EditMode))

        # We have to do these checks outside the loop, otherwise they get triggered multiple times per key press.
        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
            state = pcg_state.env_state
            if pygame.K_m in keys_up_this_frame:
                state = self.mutate_world(_rng, state, 1)
            if pygame.K_c in keys_up_this_frame:
                state, _ = mutate_add_connected_shape(
                    _rng, state, self.env_params, self.static_env_params, self.ued_params
                )
            elif pygame.K_s in keys_up_this_frame:
                state = mutate_add_shape(_rng, state, self.env_params, self.static_env_params, self.ued_params)
            elif pygame.K_p in keys_up_this_frame:
                state = mutate_swap_role(_rng, state, self.env_params, self.static_env_params, self.ued_params)
            elif pygame.K_r in keys_up_this_frame:
                state = mutate_remove_shape(_rng, state, self.env_params, self.static_env_params, self.ued_params)
            elif pygame.K_j in keys_up_this_frame:
                state = mutate_remove_joint(_rng, state, self.env_params, self.static_env_params, self.ued_params)
            elif pygame.K_t in keys_up_this_frame:
                state = mutate_toggle_fixture(_rng, state, self.env_params, self.static_env_params, self.ued_params)
            elif pygame.K_g in keys_up_this_frame:
                state = mutate_add_thruster(_rng, state, self.env_params, self.static_env_params, self.ued_params)
            elif pygame.K_l in keys_up_this_frame:
                state = mutate_remove_thruster(_rng, state, self.env_params, self.static_env_params, self.ued_params)
            elif pygame.K_b in keys_up_this_frame:
                state = self.mutate_change_shape_size(
                    _rng, state, self.env_params, self.static_env_params, self.ued_params
                )
            elif pygame.K_x in keys_up_this_frame:
                state = mutate_change_shape_location(
                    _rng, state, self.env_params, self.static_env_params, self.ued_params
                )
            elif pygame.K_k in keys_up_this_frame:
                state = self.mutate_change_shape_rotation(
                    _rng, state, self.env_params, self.static_env_params, self.ued_params
                )
            pcg_state = pcg_state.replace(env_state=state)

        if pygame.K_p in keys_up_this_frame:
            global myrng
            myrng, _rng = jax.random.split(myrng)
            # use the same rng
            pcg_state = permute_pcg_state(_rng, pcg_state, self.static_env_params)
        if self.edit_shape_mode == EditMode.SELECT:  # select a shape
            pcg_state = self._edit_select_shape(pcg_state, left_click, right_click, keys)
            self._put_state_values_into_gui(pcg_state)
            pcg_state = self._select_shape_keyboard_shortcuts(pcg_state, left_click, keys)
        else:
            pcg_state = self._reset_select_shape(pcg_state)  # don't highlight
            self._put_state_values_into_gui(pcg_state)
            self._show_correct_widgets(None)

        if self.edit_shape_mode != EditMode.ADD_TRIANGLE or not self.creating_shape:
            self.num_triangle_clicks = 0

        if self.edit_shape_mode == EditMode.ADD_CIRCLE:
            pcg_state = self._edit_circle(pcg_state, left_click, right_click)
        elif self.edit_shape_mode == EditMode.ADD_RECTANGLE:
            pcg_state = self._edit_rect(pcg_state, left_click, right_click)
        elif self.edit_shape_mode == EditMode.ADD_JOINT:
            pcg_state = self._edit_joint(pcg_state, left_click, right_click)
        elif self.edit_shape_mode == EditMode.ADD_TRIANGLE:
            pcg_state = self._edit_triangle(pcg_state, left_click, right_click)
        elif self.edit_shape_mode == EditMode.ADD_THRUSTER:
            pcg_state = self._edit_thruster(pcg_state, left_click, right_click)

        pcg_state = pcg_state.replace(
            env_state=recompute_global_joint_positions(
                pcg_state.env_state.replace(
                    collision_matrix=calculate_collision_matrix(self.static_env_params, pcg_state.env_state.joint),
                ),
                self.static_env_params,
            ),
            env_state_pcg_mask=pcg_state.env_state_pcg_mask.replace(
                collision_matrix=jnp.zeros_like(pcg_state.env_state_pcg_mask.collision_matrix)
            ),
        )
        return pcg_state

    def _update_params(self, new_static_env_params: StaticEnvParams, new_env_params: EnvParams):
        self.static_env_params = new_static_env_params.replace(
            frame_skip=self.config["frame_skip"], downscale=self.config["downscale"]
        )
        self.env_params = new_env_params
        env = make_kinetix_env_from_name("Kinetix-Entity-MultiDiscrete-v1", static_env_params=self.static_env_params)
        self.env = AutoResetWrapper(env, make_reset_function(self.static_env_params))
        self._setup_rendering(self.static_env_params, self.env_params)

    def _discard_shape_being_created(self, pcg_state):
        env_state = pcg_state.env_state
        if self.creating_shape:
            if self.edit_shape_mode == EditMode.ADD_CIRCLE:
                env_state = env_state.replace(
                    circle=env_state.circle.replace(
                        active=env_state.circle.active.at[self.creating_shape_index].set(False)
                    )
                )

            elif self.edit_shape_mode == EditMode.ADD_RECTANGLE:
                env_state = env_state.replace(
                    polygon=env_state.polygon.replace(
                        active=env_state.polygon.active.at[self.creating_shape_index].set(False)
                    )
                )

        self.creating_shape = False

        return pcg_state.replace(env_state=env_state)

    def _handle_scroll_wheel(self, pcg_state, y):
        if y == 0:
            return pcg_state

        state = self._discard_shape_being_created(pcg_state)

        self.edit_shape_mode = EditMode((self.edit_shape_mode.value + y) % len(EditMode))

        return state

    def _get_mouse_position_world_space(self):
        mouse_pos = pygame.mouse.get_pos()
        return (
            jnp.array(
                [
                    mouse_pos[0] / self.upscale - self.side_panel_width,
                    self.static_env_params.screen_dim[1] - mouse_pos[1] / self.upscale,
                ]
            )
            / self.env_params.pixels_per_unit
        )

    def _get_circles_on_mouse(self, state):
        mouse_pos = self._get_mouse_position_world_space()

        cis = []

        for ci in jnp.arange(self.static_env_params.num_circles)[::-1]:
            circle = jax.tree.map(lambda x: x[ci], state.circle)

            if not circle.active:
                continue

            dist = jnp.linalg.norm(mouse_pos - circle.position)
            if dist <= circle.radius:
                cis.append(ci)

        return cis

    def _get_revolute_joints_on_mouse(self, state: EnvState):
        mouse_pos = self._get_mouse_position_world_space()

        ris = []

        for ri in jnp.arange(self.static_env_params.num_joints)[::-1]:
            joint = jax.tree.map(lambda x: x[ri], state.joint)

            if not joint.active:
                continue

            dist = jnp.linalg.norm(mouse_pos - joint.global_position)
            if dist <= 10 / 100:  # arbitrary
                ris.append(ri)

        return ris

    def _get_thrusters_on_mouse(self, state: EnvState):
        mouse_pos = self._get_mouse_position_world_space()

        ris = []

        for ri in jnp.arange(self.static_env_params.num_thrusters)[::-1]:
            thruster = jax.tree.map(lambda x: x[ri], state.thruster)

            if not thruster.active:
                continue

            dist = jnp.linalg.norm(mouse_pos - thruster.global_position)
            if dist <= 16 / 100:  # arbitrary
                ris.append(ri)

        return ris

    def _get_joints_attached_to_shape(self, state, shape_index):
        r_a = jnp.arange(self.static_env_params.num_joints)[state.joint.a_index == shape_index]
        r_b = jnp.arange(self.static_env_params.num_joints)[state.joint.b_index == shape_index]

        t = jnp.arange(self.static_env_params.num_thrusters)[state.thruster.object_index == shape_index]
        return jnp.concatenate([r_a, r_b], axis=0), t

    def _edit_thruster(self, pcg_state: PCGState, left_click: bool, right_click: bool):
        if not self.creating_shape and (1 - pcg_state.env_state.thruster.active.astype(int)).sum() == 0:
            if not right_click:
                return pcg_state

        thruster_pos = self._get_mouse_position_world_space()
        idx = -1
        for ri in self._get_polygons_on_mouse(pcg_state.env_state):
            r = jax.tree.map(lambda x: x[ri], pcg_state.env_state.polygon)
            thruster_pos = snap_to_polygon_center_line(r, thruster_pos)
            relative_pos = jnp.matmul(rmat(r.rotation).transpose((1, 0)), thruster_pos - r.position)
            idx = ri
            break
        if idx == -1:
            for ci in self._get_circles_on_mouse(pcg_state.env_state):
                c = jax.tree.map(lambda x: x[ci], pcg_state.env_state.circle)
                thruster_pos = snap_to_center(c, thruster_pos)
                thruster_pos = snap_to_circle_center_line(c, thruster_pos)
                relative_pos = thruster_pos - c.position
                idx = ci + self.static_env_params.num_polygons
                break
        if left_click:
            if self.creating_shape:
                self.creating_shape = False
            else:
                if idx >= 0:
                    self.creating_shape = True
                    self.creating_shape_position = thruster_pos
                    self.creating_shape_index = jnp.argmin(pcg_state.env_state.thruster.active)
                    shape = select_shape(pcg_state.env_state, idx, self.static_env_params)

                    def _add_thruster_to_state(state):
                        state = state.replace(
                            thruster=state.thruster.replace(
                                object_index=state.thruster.object_index.at[self.creating_shape_index].set(idx),
                                relative_position=state.thruster.relative_position.at[self.creating_shape_index].set(
                                    relative_pos
                                ),
                                power=state.thruster.power.at[self.creating_shape_index].set(
                                    1.0 / jax.lax.select(shape.inverse_mass == 0, 1.0, shape.inverse_mass)
                                ),
                                active=state.thruster.active.at[self.creating_shape_index].set(True),
                                global_position=state.thruster.global_position.at[self.creating_shape_index].set(
                                    thruster_pos
                                ),
                                rotation=state.thruster.rotation.at[self.creating_shape_index].set(0.0),
                            ),
                            thruster_bindings=state.thruster_bindings.at[self.creating_shape_index].set(0),
                        )
                        return state

                    pcg_state = pcg_state.replace(
                        env_state=_add_thruster_to_state(pcg_state.env_state),
                        env_state_max=_add_thruster_to_state(pcg_state.env_state_max),
                    )
        elif right_click:
            for ti in self._get_thrusters_on_mouse(pcg_state.env_state):

                def _remove_thruster_from_state(state):
                    return state.replace(
                        thruster=state.thruster.replace(active=state.thruster.active.at[ti].set(False))
                    )

                return pcg_state.replace(
                    env_state=_remove_thruster_from_state(pcg_state.env_state),
                    env_state_max=_remove_thruster_from_state(pcg_state.env_state_max),
                )
        else:
            if self.creating_shape:
                curr_pos = self._get_mouse_position_world_space()

                normal = pcg_state.env_state.thruster.relative_position[self.creating_shape_index]
                angle = jnp.arctan2(normal[1], normal[0])

                relative_pos = curr_pos - self.creating_shape_position
                # rotation = jnp.arctan2(relative_pos[1], relative_pos[0])
                rotation = jnp.pi + jnp.arctan2(relative_pos[1], relative_pos[0]) + angle
                angle_round = jnp.round(rotation / (jnp.pi / 2))
                angle_norm = rotation / (jnp.pi / 2)
                if jnp.abs(angle_round - angle_norm) < 0.3:
                    rotation = angle_round * (jnp.pi / 2)

                def _update_thruster_rotation(state):
                    return state.replace(
                        thruster=state.thruster.replace(
                            rotation=state.thruster.rotation.at[self.creating_shape_index].set(rotation - angle),
                        )
                    )

                pcg_state = pcg_state.replace(
                    env_state=_update_thruster_rotation(pcg_state.env_state),
                    env_state_max=_update_thruster_rotation(pcg_state.env_state_max),
                )
            else:
                pass

        return pcg_state

    def _edit_circle(self, pcg_state: PCGState, left_click: bool, right_click: bool):
        if right_click:
            for ci in self._get_circles_on_mouse(pcg_state.env_state):
                attached_j, attached_t = self._get_joints_attached_to_shape(
                    pcg_state.env_state, ci + self.static_env_params.num_polygons
                )

                def _remove_circle_from_state(state):
                    return state.replace(
                        circle=state.circle.replace(active=state.circle.active.at[ci].set(False)),
                        joint=state.joint.replace(active=state.joint.active.at[attached_j].set(False)),
                        thruster=state.thruster.replace(active=state.thruster.active.at[attached_t].set(False)),
                    )

                env_state = _remove_circle_from_state(pcg_state.env_state)
                env_state_pcg_mask = _remove_circle_from_state(pcg_state.env_state_pcg_mask)
                env_state_max = _remove_circle_from_state(pcg_state.env_state_max)

                env_state = env_state.replace(
                    collision_matrix=calculate_collision_matrix(self.static_env_params, env_state.joint)
                )

                return PCGState(
                    env_state=env_state,
                    env_state_pcg_mask=env_state_pcg_mask,
                    env_state_max=env_state_max,
                    tied_together=pcg_state.tied_together,
                )

        if not self.creating_shape and (1 - pcg_state.env_state.circle.active.astype(int)).sum() == 0:
            return pcg_state

        radius = jnp.linalg.norm(self._get_mouse_position_world_space() - self.create_shape_position)
        radius = jnp.clip(radius, 5.0 / self.env_params.pixels_per_unit, self.static_env_params.max_shape_size / 2)

        def _add_circle(state, highlight):
            state = state.replace(
                circle=state.circle.replace(
                    position=state.circle.position.at[self.creating_shape_index].set(self.create_shape_position),
                    velocity=state.circle.velocity.at[self.creating_shape_index].set(jnp.array([0.0, 0.0])),
                    radius=state.circle.radius.at[self.creating_shape_index].set(radius),
                    inverse_mass=state.circle.inverse_mass.at[self.creating_shape_index].set(1.0),
                    inverse_inertia=state.circle.inverse_inertia.at[self.creating_shape_index].set(1.0),
                    active=state.circle.active.at[self.creating_shape_index].set(True),
                    collision_mode=state.circle.collision_mode.at[self.creating_shape_index].set(1),
                ),
                circle_shape_roles=state.circle_shape_roles.at[self.creating_shape_index].set(0),
                circle_highlighted=state.circle_highlighted.at[self.creating_shape_index].set(highlight),
                circle_densities=state.circle_densities.at[self.creating_shape_index].set(1.0),
            )
            return state

        if left_click:
            if self.creating_shape:
                env_state = _add_circle(pcg_state.env_state, False)
                env_state_max = _add_circle(pcg_state.env_state_max, False)

                env_state = recalculate_mass_and_inertia(
                    env_state,
                    self.static_env_params,
                    env_state.polygon_densities,
                    env_state.circle_densities,
                )
                env_state_max = recalculate_mass_and_inertia(
                    env_state_max,
                    self.static_env_params,
                    env_state.polygon_densities,
                    env_state.circle_densities,
                )

                pcg_state = pcg_state.replace(env_state=env_state, env_state_max=env_state_max)

                self.creating_shape = False
            else:
                self.creating_shape_index = jnp.argmin(pcg_state.env_state.circle.active)
                self.create_shape_position = self._get_mouse_position_world_space()
                self.creating_shape = True

        else:
            if self.creating_shape:
                env_state = _add_circle(pcg_state.env_state, True)
                env_state_max = _add_circle(pcg_state.env_state_max, True)

                pcg_state = pcg_state.replace(env_state=env_state, env_state_max=env_state_max)
            else:
                pass

        return pcg_state

    def _get_polygons_on_mouse(self, state, n_vertices=None):
        # n_vertices=None selects both triangles and quads
        ris = []

        mouse_pos = self._get_mouse_position_world_space()

        for ri in jnp.arange(self.static_env_params.num_polygons)[::-1]:
            polygon = jax.tree.map(lambda x: x[ri], state.polygon)

            if (not polygon.active) or ((n_vertices is not None) and polygon.n_vertices != n_vertices):
                continue

            mpos = rmat(-polygon.rotation) @ (mouse_pos - polygon.position)

            def _signed_line_distance(a, b, c):
                return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

            inside = True
            for fi in range(polygon.n_vertices):
                v1 = polygon.vertices[fi]
                v2 = polygon.vertices[(fi + 1) % polygon.n_vertices]
                if _signed_line_distance(mpos, v1, v2) > 0:
                    inside = False

            if inside:
                ris.append(ri)

        return ris

    def _edit_rect(self, pcg_state: PCGState, left_click: bool, right_click: bool):
        if right_click:
            for ri in self._get_polygons_on_mouse(pcg_state.env_state, n_vertices=4):
                attached_j, attached_t = self._get_joints_attached_to_shape(pcg_state.env_state, ri)

                def _remove_rect_from_state(state):
                    state = state.replace(
                        polygon=state.polygon.replace(
                            active=state.polygon.active.at[ri].set(False),
                            rotation=state.polygon.rotation.at[ri].set(0.0),
                        ),
                        joint=state.joint.replace(active=state.joint.active.at[attached_j].set(False)),
                        thruster=state.thruster.replace(active=state.thruster.active.at[attached_t].set(False)),
                    )

                    return state

                env_state = _remove_rect_from_state(pcg_state.env_state)
                env_state_max = _remove_rect_from_state(pcg_state.env_state_max)
                env_state_pcg_mask = _remove_rect_from_state(pcg_state.env_state_pcg_mask)

                env_state = env_state.replace(
                    collision_matrix=calculate_collision_matrix(self.static_env_params, env_state.joint)
                )

                return PCGState(
                    env_state=env_state,
                    env_state_max=env_state_max,
                    env_state_pcg_mask=env_state_pcg_mask,
                    tied_together=pcg_state.tied_together,
                )

        if not self.creating_shape and (1 - pcg_state.env_state.polygon.active.astype(int)).sum() == 0:
            return pcg_state

        diff = (self._get_mouse_position_world_space() - self.create_shape_position) / 2
        diff = jnp.clip(
            diff,
            -(self.static_env_params.max_shape_size / 2) / jnp.sqrt(2),
            (self.static_env_params.max_shape_size / 2) / jnp.sqrt(2),
        )
        half_dim = jnp.abs(diff)
        half_dim = jnp.clip(half_dim, a_min=5.0 / self.env_params.pixels_per_unit)
        vertices = rectangle_vertices(half_dim)

        def _add_rect_to_state(state, highlight):
            state = state.replace(
                polygon=state.polygon.replace(
                    position=state.polygon.position.at[self.creating_shape_index].set(
                        self.create_shape_position + diff
                    ),
                    velocity=state.polygon.velocity.at[self.creating_shape_index].set(jnp.array([0.0, 0.0])),
                    vertices=state.polygon.vertices.at[self.creating_shape_index].set(vertices),
                    inverse_mass=state.polygon.inverse_mass.at[self.creating_shape_index].set(1.0),
                    inverse_inertia=state.polygon.inverse_inertia.at[self.creating_shape_index].set(1.0),
                    active=state.polygon.active.at[self.creating_shape_index].set(True),
                    collision_mode=state.polygon.collision_mode.at[self.creating_shape_index].set(1),
                    n_vertices=state.polygon.n_vertices.at[self.creating_shape_index].set(4),
                ),
                polygon_shape_roles=state.polygon_shape_roles.at[self.creating_shape_index].set(0),
                polygon_highlighted=state.polygon_highlighted.at[self.creating_shape_index].set(highlight),
                polygon_densities=state.polygon_densities.at[self.creating_shape_index].set(1.0),
            )

            return state

        if left_click:
            if self.creating_shape:
                env_state = _add_rect_to_state(pcg_state.env_state, False)
                env_state_max = _add_rect_to_state(pcg_state.env_state_max, False)

                env_state = recalculate_mass_and_inertia(
                    env_state,
                    self.static_env_params,
                    env_state.polygon_densities,
                    env_state.circle_densities,
                )
                env_state_max = recalculate_mass_and_inertia(
                    env_state_max,
                    self.static_env_params,
                    env_state.polygon_densities,
                    env_state.circle_densities,
                )

                pcg_state = pcg_state.replace(env_state=env_state, env_state_max=env_state_max)

                self.creating_shape = False
            else:
                self.creating_shape_index = jnp.argmin(pcg_state.env_state.polygon.active)
                self.create_shape_position = self._get_mouse_position_world_space()
                self.creating_shape = True
        else:
            if self.creating_shape:
                env_state = _add_rect_to_state(pcg_state.env_state, True)
                env_state_max = _add_rect_to_state(pcg_state.env_state_max, True)

                pcg_state = pcg_state.replace(env_state=env_state, env_state_max=env_state_max)
            else:
                pass

        return pcg_state

    def _reset_triangles(self):
        self.triangle_order = jnp.array([0, 1, 2])
        self.num_triangle_clicks = 0
        self.creating_shape = False

    def _edit_triangle(self, pcg_state: PCGState, left_click: bool, right_click: bool):
        if right_click:
            self.num_triangle_clicks = 0
            for ri in self._get_polygons_on_mouse(pcg_state.env_state, n_vertices=3):
                attached_r, attached_f = self._get_joints_attached_to_shape(pcg_state.env_state, ri)

                def _remove_triangle_from_state(state):
                    state = state.replace(
                        polygon=state.polygon.replace(
                            active=state.polygon.active.at[ri].set(False),
                            rotation=state.polygon.rotation.at[ri].set(0.0),
                        ),
                        joint=state.joint.replace(active=state.joint.active.at[attached_r].set(False)),
                    )

                    return state

                env_state = _remove_triangle_from_state(pcg_state.env_state)
                env_state_max = _remove_triangle_from_state(pcg_state.env_state_max)
                env_state_pcg_mask = _remove_triangle_from_state(pcg_state.env_state_pcg_mask)

                env_state = env_state.replace(
                    collision_matrix=calculate_collision_matrix(self.static_env_params, env_state.joint)
                )

                pcg_state = PCGState(
                    env_state=env_state,
                    env_state_max=env_state_max,
                    env_state_pcg_mask=env_state_pcg_mask,
                    tied_together=pcg_state.tied_together,
                )

                return pcg_state

        if not self.creating_shape and (1 - pcg_state.env_state.polygon.active.astype(int)).sum() == 0:
            return pcg_state

        def get_correct_center_two_verts(verts):
            return (jnp.max(verts, axis=0) + jnp.min(verts, axis=0)) / 2

        def order_clockwise(verts, loose_ordering=False):
            # verts has shape (3, 2), order them clockwise.
            # https://stackoverflow.com/questions/51074984/sort-vertices-in-clockwise-order
            # Calculate centroid
            centroid = jnp.mean(verts, axis=0)
            # Calculate angles
            angles = jnp.round(jnp.arctan2(verts[:, 1] - centroid[1], verts[:, 0] - centroid[0]), 2)
            # Order vertices
            order = jnp.argsort(-angles, stable=True)
            if loose_ordering:
                order = jnp.arange(len(order))
            ans = verts[order]
            # order is of shape (2, ) or (3, ). I want it to always be of shape 3
            if len(order) < 3:
                order = jnp.concatenate([order, jnp.array([2])])
            return ans, order

        def do_triangle_n_click(pcg_state, how_many_clicks, is_on_a_click=False):
            n = how_many_clicks
            # if we must keep them clockwise all the time, then the one we edit / move around may have varying indices.
            current_index_to_change = self.triangle_order[n]
            sign = 1
            idxs = jnp.arange(n + 1)
            idxs_to_allow = idxs[~(idxs == current_index_to_change)]

            # Get the new vertex and clip its position
            new_tentative_vert = (
                self._get_mouse_position_world_space() - pcg_state.env_state.polygon.position[self.creating_shape_index]
            )
            new_tentative_vert = jnp.clip(
                new_tentative_vert,
                jnp.max(pcg_state.env_state.polygon.vertices[self.creating_shape_index, idxs_to_allow], axis=0)
                - self.static_env_params.max_shape_size * 0.8,
                jnp.min(pcg_state.env_state.polygon.vertices[self.creating_shape_index, idxs_to_allow], axis=0)
                + self.static_env_params.max_shape_size * 0.8,
            )
            new_verts = pcg_state.env_state.polygon.vertices.at[self.creating_shape_index, current_index_to_change].set(
                new_tentative_vert
            )
            new_center_two = get_correct_center_two_verts(new_verts[self.creating_shape_index, : n + 1])

            _, new_center_three = calc_inverse_mass_polygon(
                new_verts[self.creating_shape_index],
                3,
                self.static_env_params,
                1.0,
            )
            new_center = jax.lax.select(n == 1, new_center_two, new_center_three)

            new_verts = new_verts.at[self.creating_shape_index].add(-sign * new_center)
            vvs = new_verts[self.creating_shape_index, : n + 1]

            ordered_vertices, new_permutation = order_clockwise(vvs, loose_ordering=not is_on_a_click)
            self.triangle_order = self.triangle_order[new_permutation]
            new_verts = new_verts.at[self.creating_shape_index, : n + 1].set(ordered_vertices)

            env_state = pcg_state.env_state.replace(
                polygon=pcg_state.env_state.polygon.replace(
                    vertices=new_verts,
                    position=pcg_state.env_state.polygon.position.at[self.creating_shape_index].add(sign * new_center),
                    n_vertices=pcg_state.env_state.polygon.n_vertices.at[self.creating_shape_index].set(n + 1),
                ),
            )
            env_state_max = pcg_state.env_state_max.replace(
                polygon=pcg_state.env_state_max.polygon.replace(
                    vertices=new_verts,
                    position=pcg_state.env_state_max.polygon.position.at[self.creating_shape_index].add(
                        sign * new_center
                    ),
                    n_vertices=pcg_state.env_state_max.polygon.n_vertices.at[self.creating_shape_index].set(n + 1),
                ),
            )

            pcg_state = pcg_state.replace(env_state=env_state, env_state_max=env_state_max)

            return pcg_state

        if left_click:
            if self.creating_shape:
                assert 3 > self.num_triangle_clicks > 0
                if self.num_triangle_clicks == 1:
                    pcg_state = do_triangle_n_click(pcg_state, 1, is_on_a_click=True)
                    self.num_triangle_clicks += 1
                else:  # this finishes it
                    pcg_state = do_triangle_n_click(pcg_state, 2, is_on_a_click=True)
                    self.creating_shape = False
                    self.num_triangle_clicks = 0
                    pcg_state = pcg_state.replace(
                        env_state=recalculate_mass_and_inertia(
                            pcg_state.env_state,
                            self.static_env_params,
                            pcg_state.env_state.polygon_densities,
                            pcg_state.env_state.circle_densities,
                        )
                    )

            else:
                self.triangle_order = jnp.array([0, 1, 2])
                self.creating_shape_index = jnp.argmin(pcg_state.env_state.polygon.active)
                self.create_shape_position = self._get_mouse_position_world_space()
                self.creating_shape = True
                self.num_triangle_clicks = 1
                vertices = jnp.zeros((self.static_env_params.max_polygon_vertices, 2), dtype=jnp.float32)

                def _add_triangle_to_state(state):
                    state = state.replace(
                        polygon=state.polygon.replace(
                            position=state.polygon.position.at[self.creating_shape_index].set(
                                self.create_shape_position
                            ),
                            velocity=state.polygon.velocity.at[self.creating_shape_index].set(jnp.array([0.0, 0.0])),
                            vertices=state.polygon.vertices.at[self.creating_shape_index].set(vertices),
                            inverse_mass=state.polygon.inverse_mass.at[self.creating_shape_index].set(1.0),
                            inverse_inertia=state.polygon.inverse_inertia.at[self.creating_shape_index].set(1.0),
                            active=state.polygon.active.at[self.creating_shape_index].set(True),
                            n_vertices=state.polygon.n_vertices.at[self.creating_shape_index].set(1),
                        ),
                        polygon_shape_roles=state.polygon_shape_roles.at[self.creating_shape_index].set(0),
                        polygon_highlighted=state.polygon_highlighted.at[self.creating_shape_index].set(False),
                        polygon_densities=state.polygon_densities.at[self.creating_shape_index].set(1.0),
                    )

                    return state

                pcg_state = pcg_state.replace(
                    env_state=_add_triangle_to_state(pcg_state.env_state),
                    env_state_max=_add_triangle_to_state(pcg_state.env_state_max),
                )

        elif self.creating_shape:
            assert 1 <= self.num_triangle_clicks <= 2
            pcg_state = do_triangle_n_click(
                pcg_state, self.num_triangle_clicks, is_on_a_click=self.num_triangle_clicks == 1
            )

        return pcg_state

    def _edit_joint(self, pcg_state: PCGState, left_click: bool, right_click: bool):
        if left_click and pcg_state.env_state.joint.active.all():
            return pcg_state

        if left_click:
            joint_index = jnp.argmin(pcg_state.env_state.joint.active)
            joint_position = self._get_mouse_position_world_space()
            # reverse them so that the joint order and rendering order remains the same.
            # We want the first shape to have a lower index than the second shape, with circles always having higher indices compared to rectangles.
            circles = self._get_circles_on_mouse(pcg_state.env_state)[::-1]
            rects = self._get_polygons_on_mouse(pcg_state.env_state)[::-1]

            if len(rects) + len(circles) >= 2:
                r1 = len(rects) >= 1
                r2 = len(rects) >= 2

                a_index = rects[0] if r1 else circles[0]  # + self.static_env_params.num_polygons
                b_index = rects[r1 * 1] if r2 else circles[1 - 1 * r1]  # + self.static_env_params.num_polygons

                a_shape = pcg_state.env_state.polygon if r1 else pcg_state.env_state.circle
                b_shape = pcg_state.env_state.polygon if r2 else pcg_state.env_state.circle

                a = jax.tree.map(lambda x: x[a_index], a_shape)
                b = jax.tree.map(lambda x: x[b_index], b_shape)

                a_index += (not r1) * self.static_env_params.num_polygons
                b_index += (not r2) * self.static_env_params.num_polygons

                joint_position = snap_to_center(a, joint_position)
                joint_position = snap_to_center(b, joint_position)

                a_relative_pos = jnp.matmul(rmat(a.rotation).transpose((1, 0)), joint_position - a.position)
                b_relative_pos = jnp.matmul(rmat(b.rotation).transpose((1, 0)), joint_position - b.position)

                def _add_joint_to_state(state):
                    state = state.replace(
                        joint=state.joint.replace(
                            a_index=state.joint.a_index.at[joint_index].set(a_index),
                            b_index=state.joint.b_index.at[joint_index].set(b_index),
                            a_relative_pos=state.joint.a_relative_pos.at[joint_index].set(a_relative_pos),
                            b_relative_pos=state.joint.b_relative_pos.at[joint_index].set(b_relative_pos),
                            active=state.joint.active.at[joint_index].set(True),
                            global_position=state.joint.global_position.at[joint_index].set(joint_position),
                            motor_on=state.joint.motor_on.at[joint_index].set(True),
                            motor_speed=state.joint.motor_speed.at[joint_index].set(1.0),
                            motor_power=state.joint.motor_power.at[joint_index].set(1.0),
                            rotation=state.joint.rotation.at[joint_index].set(b.rotation - a.rotation),
                        )
                    )

                    return state

                env_state = _add_joint_to_state(pcg_state.env_state)
                env_state_max = _add_joint_to_state(pcg_state.env_state_max)

                env_state = env_state.replace(
                    collision_matrix=calculate_collision_matrix(self.static_env_params, env_state.joint)
                )

                pcg_state = pcg_state.replace(env_state=env_state, env_state_max=env_state_max)

        return pcg_state

    def _reset_select_shape(self, pcg_state):
        pcg_state = pcg_state.replace(
            env_state=pcg_state.env_state.replace(
                polygon_highlighted=jnp.zeros_like(pcg_state.env_state.polygon_highlighted),
                circle_highlighted=jnp.zeros_like(pcg_state.env_state.circle_highlighted),
            )
        )
        self.selected_shape_index = -1
        self.selected_shape_type = ObjectType.POLYGON
        self._hide_all_widgets()
        return pcg_state

    def _hide_all_widgets(self):
        for widget in self.all_widgets.values():
            for w in widget.values():
                w.hide()

    def _show_correct_widgets(self, type: ObjectType | None, do_tie_ui: bool = False):
        for widget in self.all_widgets["GENERAL"].values():
            widget.show()
        if do_tie_ui:
            for widget in self.all_widgets["TIE_TOGETHER"].values():
                widget.show()

            n = len(self.all_selected_shapes)
            #  {[int(i) for (i, t) in self.all_selected_shapes]}
            self.all_widgets["GENERAL"]["lblGeneral"].setText(f"Selected {n} Objects")
            return

        for widget in self.all_widgets[type].values():
            widget.show()
        if type is None:
            self.all_widgets["GENERAL"]["lblGeneral"].setText(f"Global")
        else:
            self.all_widgets["GENERAL"]["lblGeneral"].setText(f"{type.name} (idx {self.selected_shape_index})")

    def _select_shape_keyboard_shortcuts(self, pcg_state: PCGState, left_click: bool, keys: list[int]):
        if left_click:
            return pcg_state
        if len(keys) != 0 and self.selected_shape_index != -1:
            s = 1.0
            ang_s = 0.1
            vel = jnp.array([0.0, 0.0])
            angular_vel = 0.0
            should_toggle_fixed = False
            should_toggle_collidable = False
            change_angle = 0

            def add_step(widget_name, direction, speed=10, overwrite_amount=None):
                widget = self.all_widgets[self.selected_shape_type][widget_name]
                val = widget.getValue()
                step = widget.step
                amount_to_add = overwrite_amount or step * direction * speed
                widget.setValue(jnp.clip(val + amount_to_add, widget.min, widget.max))

            if pygame.K_w in keys:
                add_step("sldPosition_Y", 1)
            if pygame.K_s in keys:
                add_step("sldPosition_Y", -1)
            if pygame.K_a in keys:
                add_step("sldPosition_X", -1)
            if pygame.K_d in keys:
                add_step("sldPosition_X", 1)

            if pygame.K_q in keys:
                add_step("sldRotation", 1)
            if pygame.K_e in keys:
                add_step("sldRotation", -1)

            if pygame.K_f in keys:
                self.all_widgets[self.selected_shape_type]["tglFixate"].toggle()

            if pygame.K_c in keys and not (pygame.key.get_mods() & pygame.KMOD_CTRL):
                widget = self.all_widgets[self.selected_shape_type]["sldCollidability"]
                curr_val = int(widget.getValue())
                widget.setValue((curr_val + 1) % (widget.max + 1))

            if pygame.K_r in keys and not (pygame.key.get_mods() & pygame.KMOD_CTRL):
                widget = self.all_widgets[self.selected_shape_type]["sldRole"]
                curr_val = int(widget.getValue())
                widget.setValue((curr_val + 1) % (widget.max + 1))

            if pygame.K_LEFTBRACKET in keys:
                add_step("sldRotation", 1, 10, jnp.pi / 4)

            if pygame.K_RIGHTBRACKET in keys:
                add_step("sldRotation", -1, 10, -jnp.pi / 4)

            if pygame.K_c in keys and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                # copy
                if self.selected_shape_type == ObjectType.POLYGON:  # rect
                    if not self.pcg_state.env_state.polygon.active.all():
                        where_to_add = jnp.argmin(pcg_state.env_state.polygon.active)
                        if where_to_add < self.static_env_params.num_polygons:

                            def _copy_polygon(state, shift):
                                state = state.replace(
                                    polygon=jax.tree.map(
                                        lambda x: x.at[where_to_add].set(x[self.selected_shape_index]), state.polygon
                                    )
                                )
                                if shift:
                                    state = state.replace(
                                        polygon=state.polygon.replace(
                                            position=state.polygon.position.at[where_to_add].add(0.1),
                                        ),
                                        polygon_highlighted=state.polygon_highlighted.at[where_to_add].set(False),
                                    )
                                return state

                            pcg_state = pcg_state.replace(
                                env_state=_copy_polygon(pcg_state.env_state, shift=True),
                                env_state_max=_copy_polygon(pcg_state.env_state_max, shift=True),
                                env_state_pcg_mask=_copy_polygon(pcg_state.env_state_pcg_mask, shift=False),
                            )

                elif self.selected_shape_type == ObjectType.CIRCLE:  # circle
                    if not self.pcg_state.env_state.circle.active.all():
                        where_to_add = jnp.argmin(pcg_state.env_state.circle.active)
                        if where_to_add < self.static_env_params.num_circles:

                            def _copy_circle(state, shift=True):
                                state = state.replace(
                                    circle=jax.tree.map(
                                        lambda x: x.at[where_to_add].set(x[self.selected_shape_index]), state.circle
                                    )
                                )
                                if shift:
                                    state = state.replace(
                                        circle=state.circle.replace(
                                            position=state.circle.position.at[where_to_add].add(0.1),
                                        ),
                                        circle_highlighted=state.circle_highlighted.at[where_to_add].set(False),
                                    )
                                return state

                            pcg_state = pcg_state.replace(
                                env_state=_copy_circle(pcg_state.env_state),
                                env_state_max=_copy_circle(pcg_state.env_state_max),
                                env_state_pcg_mask=_copy_circle(pcg_state.env_state_pcg_mask, shift=False),
                            )

            if self.selected_shape_index >= 0:
                num = get_numeric_key_pressed(self.pygame_events)
                if num is not None:
                    if self.selected_shape_type in [ObjectType.CIRCLE, ObjectType.POLYGON]:
                        self.all_widgets[self.selected_shape_type]["sldRole"].setValue(num % 4)
                    elif self.selected_shape_type == ObjectType.JOINT:
                        self.all_widgets[self.selected_shape_type]["sldColour"].setValue(
                            num % self.static_env_params.num_motor_bindings
                        )
                    elif self.selected_shape_type == ObjectType.THRUSTER:
                        self.all_widgets[self.selected_shape_type]["sldColour"].setValue(
                            num % self.static_env_params.num_thruster_bindings
                        )

        return pcg_state

    def _edit_select_shape(self, pcg_state: PCGState, left_click: bool, right_click: bool, keys: list[int]):
        def _find_shape(pcg_state):
            found_shape = False
            selected_shape_index, selected_shape_type = -1, ObjectType.POLYGON
            for ri in self._get_revolute_joints_on_mouse(pcg_state.env_state):
                selected_shape_index = ri
                selected_shape_type = ObjectType.JOINT
                found_shape = True
                break
            if not found_shape:
                for ti in self._get_thrusters_on_mouse(pcg_state.env_state):
                    selected_shape_index = ti
                    selected_shape_type = ObjectType.THRUSTER
                    found_shape = True
                    break
            if not found_shape:
                for ri in self._get_polygons_on_mouse(pcg_state.env_state):
                    pcg_state = pcg_state.replace(
                        env_state=pcg_state.env_state.replace(
                            polygon_highlighted=pcg_state.env_state.polygon_highlighted.at[ri].set(True),
                        )
                    )
                    selected_shape_index = ri
                    selected_shape_type = ObjectType.POLYGON
                    found_shape = True
                    break
            if not found_shape:
                for ci in self._get_circles_on_mouse(pcg_state.env_state):
                    pcg_state = pcg_state.replace(
                        env_state=pcg_state.env_state.replace(
                            circle_highlighted=pcg_state.env_state.circle_highlighted.at[ci].set(True),
                        )
                    )
                    selected_shape_index = ci
                    selected_shape_type = ObjectType.CIRCLE
                    found_shape = True
                    break
            return selected_shape_index, selected_shape_type, found_shape, pcg_state
            if found_shape and self.selected_shape_type in self.all_widgets:
                self._show_correct_widgets(self.selected_shape_type)

        # if left and shift
        if left_click and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            # This is trying to select multiple things.
            idx, type, found, pcg_state = _find_shape(pcg_state)
            if found:
                t = (idx, type)
                if t in self.all_selected_shapes:
                    self.all_selected_shapes.remove(t)
                else:
                    self.all_selected_shapes.append(t)

            self._hide_all_widgets()
            self._show_correct_widgets(None, do_tie_ui=True)
        elif left_click:
            self.all_selected_shapes = []
            self._hide_all_widgets()
            pcg_state = self._reset_select_shape(pcg_state)
            self.selected_shape_index, self.selected_shape_type, found_shape, pcg_state = _find_shape(pcg_state)
            if found_shape:
                self.all_selected_shapes = [(self.selected_shape_index, self.selected_shape_type)]
                if self.selected_shape_type in self.all_widgets:
                    self._show_correct_widgets(self.selected_shape_type)
        if self.selected_shape_index < 0:
            self._show_correct_widgets(None)

        return pcg_state

    def render(self, env_state):
        # Clear
        self.screen_surface.fill((0, 0, 0))

        if self.is_editing:
            pixels = self._render_fn_edit(env_state)
        else:
            pixels = self._render_fn(env_state)
        pixels = self._render_edit_overlay_fn(pixels, self.is_editing, self.edit_shape_mode.value)

        surface = pygame.surfarray.make_surface(np.array(pixels))
        self.screen_surface.blit(surface, (0, 0))

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False


@hydra.main(version_base=None, config_path="../configs", config_name="editor")
def main(config):
    config = normalise_config(OmegaConf.to_container(config), "EDITOR", editor_config=True)
    env_params, static_env_params = generate_params_from_config(config)
    static_env_params = static_env_params.replace(frame_skip=config["frame_skip"], downscale=config["downscale"])
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    env = make_kinetix_env_from_name("Kinetix-Entity-MultiDiscrete-v1", static_env_params=static_env_params)
    env = AutoResetWrapper(env, make_reset_function(static_env_params))

    seed = config["seed"]
    print("seed", seed)
    rng = jax.random.PRNGKey(seed)
    outer_timer = tmr()
    editor = Editor(env, env_params, config, upscale=config["upscale"])
    time_e = tmr()
    print("Took {:2f}s to create editor".format(time_e - outer_timer))

    clock = pygame.time.Clock()
    while not editor.is_quit_requested():
        rng, _rng = jax.random.split(rng)
        editor.update(_rng)
        clock.tick(config["fps"])


if __name__ == "__main__":
    main()
