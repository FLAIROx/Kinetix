from pprint import pprint
import unittest

import tree

from build.lib.kinetix.environment import env_state
from kinetix.environment import EnvState
import jax
import jax.numpy as jnp
import jax.random
from matplotlib import pyplot as plt

from kinetix.environment import EnvParams, make_kinetix_env
from kinetix.environment.env_state import StaticEnvParams
from kinetix.environment import ActionType, ObservationType
from kinetix.environment.ued.ued import make_reset_fn_sample_kinetix_level
from kinetix.render import make_render_pixels
from kinetix.render.renderer_pixels import make_render_pixels_rl
from kinetix.render.renderer_symbolic_entity import make_inverse_render_entity, make_render_entities
from kinetix.render.renderer_symbolic_flat import make_inverse_render_symbolic, make_render_symbolic


def compare_env_states(env_state_a: EnvState, env_state_b: EnvState):
    def tree_allclose(a, b):
        val = jax.tree.map(lambda x, y: jnp.allclose(x, y), a, b)

        ans = jax.tree.all(val)
        if not ans:
            print("---")
            print("not close")
            pprint(val)
            print("---")
        return ans

    def _mask_out_inactives(pytree):
        def _dummy(x):
            if x.dtype == jnp.bool_:
                return jnp.zeros_like(x)
            return jnp.ones_like(x) * -1

        active_mask = pytree.active

        @jax.vmap
        def _select(a, b, c):
            return jax.lax.select(a, b, c)

        return jax.tree.map(lambda x: _select(active_mask, x, _dummy(x)), pytree)

    def _delete_fields(pytree, type="polygon"):
        if type == "polygon":
            return pytree.replace(radius=jnp.zeros_like(pytree.radius))
        elif type == "circle":
            return pytree.replace(
                vertices=jnp.zeros_like(pytree.vertices), n_vertices=jnp.zeros_like(pytree.n_vertices)
            )
        else:
            raise ValueError("Unknown type")

    def _mask_out_all_inactives(env_state: EnvState) -> EnvState:
        env_state = env_state.replace(
            polygon=_delete_fields(
                _mask_out_inactives(
                    env_state.polygon,
                )
            ),
            circle=_delete_fields(
                _mask_out_inactives(
                    env_state.circle,
                ),
                "circle",
            ),
            joint=_mask_out_inactives(
                env_state.joint.replace(
                    rotation=jnp.where(
                        env_state.joint.rotation < 0, env_state.joint.rotation + 2 * jnp.pi, env_state.joint.rotation
                    )
                ),
            ),
            thruster=_mask_out_inactives(
                env_state.thruster.replace(
                    rotation=jnp.where(
                        env_state.thruster.rotation < 0,
                        env_state.thruster.rotation + 2 * jnp.pi,
                        env_state.thruster.rotation,
                    )
                ),
            ),
            # Set all the other fields to zero if inactive
            motor_bindings=jnp.where(
                env_state.joint.active & jnp.logical_not(env_state.joint.is_fixed_joint),
                env_state.motor_bindings,
                jnp.zeros_like(env_state.motor_bindings),
            ),
            thruster_bindings=jnp.where(
                env_state.thruster.active, env_state.thruster_bindings, jnp.zeros_like(env_state.thruster_bindings)
            ),
            # densities
            polygon_densities=jnp.where(
                env_state.polygon.active, env_state.polygon_densities, jnp.zeros_like(env_state.polygon_densities)
            ),
            circle_densities=jnp.where(
                env_state.circle.active, env_state.circle_densities, jnp.zeros_like(env_state.circle_densities)
            ),
            # shape roles
            polygon_shape_roles=jnp.where(
                env_state.polygon.active, env_state.polygon_shape_roles, jnp.zeros_like(env_state.polygon_shape_roles)
            ),
            circle_shape_roles=jnp.where(
                env_state.circle.active, env_state.circle_shape_roles, jnp.zeros_like(env_state.circle_shape_roles)
            ),
        )
        return env_state

    env_state_a = jax.vmap(_mask_out_all_inactives)(env_state_a)
    env_state_b = jax.vmap(_mask_out_all_inactives)(env_state_b)
    assert tree_allclose(env_state_a, env_state_b)


class TestInverseRender(unittest.TestCase):
    def test_basic_assertion(self):
        self.assertEqual(2, 2)

    def test_inverse_render_symbolic_flat(self):
        seed = 10
        num_samples = 128
        env_params = EnvParams()
        static_env_params = StaticEnvParams()

        # Create the environment
        env = make_kinetix_env(
            action_type=ActionType.CONTINUOUS,
            observation_type=ObservationType.PIXELS,
            reset_fn=make_reset_fn_sample_kinetix_level(env_params, static_env_params),
            env_params=env_params,
            static_env_params=static_env_params,
        )
        rng, _rng_reset, _rng_action, _rng_step = jax.random.split(jax.random.PRNGKey(seed), 4)

        obs, env_state = jax.vmap(env.reset, (0, None))(jax.random.split(_rng_reset, num_samples), env_params)

        render_fn = jax.vmap(make_render_symbolic(env_params, static_env_params, padded=False, clip=False))
        render_fn_pixels = jax.vmap(make_render_pixels_rl(env_params, static_env_params))
        inverse_render_fn = jax.vmap(
            make_inverse_render_symbolic(jax.tree.map(lambda x: x[0], env_state), env_params, static_env_params)
        )

        rendered = render_fn(env_state)
        inverse_rendered = inverse_render_fn(rendered)

        compare_env_states(env_state, inverse_rendered)

        rendered_pixels = render_fn_pixels(inverse_rendered)
        self.assertTrue(jnp.allclose(rendered_pixels.image, obs.image))

    def test_inverse_render_symbolic_entity(self):
        seed = 10
        num_samples = 128
        env_params = EnvParams()
        static_env_params = StaticEnvParams()

        # Create the environment
        env = make_kinetix_env(
            action_type=ActionType.CONTINUOUS,
            observation_type=ObservationType.PIXELS,
            reset_fn=make_reset_fn_sample_kinetix_level(env_params, static_env_params),
            env_params=env_params,
            static_env_params=static_env_params,
        )
        rng, _rng_reset, _rng_action, _rng_step = jax.random.split(jax.random.PRNGKey(seed), 4)

        obs, env_state = jax.vmap(env.reset, (0, None))(jax.random.split(_rng_reset, num_samples), env_params)

        render_fn = jax.vmap(make_render_entities(env_params, static_env_params))
        render_fn_pixels = jax.vmap(make_render_pixels_rl(env_params, static_env_params))
        inverse_render_fn = jax.vmap(
            make_inverse_render_entity(jax.tree.map(lambda x: x[0], env_state), env_params, static_env_params)
        )

        rendered = render_fn(env_state)
        inverse_rendered = inverse_render_fn(rendered)

        compare_env_states(env_state, inverse_rendered)

        rendered_pixels = render_fn_pixels(inverse_rendered)
        self.assertTrue(jnp.allclose(rendered_pixels.image, obs.image))


if __name__ == "__main__":
    unittest.main()
