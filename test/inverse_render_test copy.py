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
from kinetix.render.renderer_symbolic_flat import make_inverse_render_symbolic, make_render_symbolic


def compare_env_states(env_state_a: EnvState, env_state_b: EnvState):

    print("aaa")
    print(env_state_b.circle)
    # exit()

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
                env_state.joint,
            ),
            thruster=_mask_out_inactives(
                env_state.thruster,
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

    env_state_a = _mask_out_all_inactives(env_state_a)
    env_state_b = _mask_out_all_inactives(env_state_b)
    poly_a = _delete_fields(
        _mask_out_inactives(
            env_state_a.polygon,
        ),
        "polygon",
    )
    poly_b = _delete_fields(
        _mask_out_inactives(
            env_state_b.polygon,
        ),
        "polygon",
    )

    circle_a = _delete_fields(
        _mask_out_inactives(
            env_state_a.circle,
        ),
        "circle",
    )
    circle_b = _delete_fields(
        _mask_out_inactives(
            env_state_b.circle,
        ),
        "circle",
    )

    print("POLYS IS ALL CLOSE", tree_allclose(poly_a, poly_b))
    print("CIRCLES IS ALL CLOSE", tree_allclose(circle_a, circle_b))

    print("all close", tree_allclose(env_state_a, env_state_b))

    print("joint", env_state_a.joint.global_position, env_state_b.joint.global_position)

    print(
        "motor binding\n",
        env_state_a.motor_bindings,
        "\n",
        env_state_b.motor_bindings,
        "\n",
        env_state_a.joint.active,
        "\n",
        env_state_b.joint.active,
        "\n",
        env_state_a.joint.is_fixed_joint,
        "\n",
        env_state_b.joint.is_fixed_joint,
    )
    print("circle densities\n", env_state_a.circle_densities, env_state_b.circle_densities)


class TestInverseRender(unittest.TestCase):
    def test_basic_assertion(self):
        self.assertEqual(2, 2)

    def test_inverse_render(self):
        seed = 10
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

        obs, env_state = env.reset(_rng_reset, env_params)

        print("JOINT MOTOR BINFING", env_state.joint.active, env_state.motor_bindings)

        render_fn = make_render_symbolic(env_params, static_env_params, padded=False, clip=False)
        inverse_render_fn = make_inverse_render_symbolic(env_state, env_params, static_env_params)

        rendered = render_fn(env_state)
        inverse_rendered = inverse_render_fn(rendered)
        # print(obs.sum(), rendered.sum())
        ans = jax.tree.map(lambda x, y: jnp.allclose(x, y), env_state, inverse_rendered)

        # print("GOOD", env_state.circle)
        # print("BAD", inverse_rendered.circle)
        print(ans)

        compare_env_states(env_state, inverse_rendered)
        assert jnp.all(obs == rendered)
        assert jax.tree.all(ans)


if __name__ == "__main__":
    unittest.main()
