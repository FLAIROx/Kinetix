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
from kinetix.render.renderer_symbolic_flat import make_inverse_render_symbolic


# This applies normal gymnax auto-reset behaviour, resetting to a different randomly-generated level each time.
def main():
    env_params = EnvParams()
    static_env_params = StaticEnvParams()
    # Create the environment
    env = make_kinetix_env(
        action_type=ActionType.CONTINUOUS,
        observation_type=ObservationType.SYMBOLIC_FLAT,
        reset_fn=make_reset_fn_sample_kinetix_level(env_params, static_env_params),
        env_params=env_params,
        static_env_params=static_env_params,
    )
    rng, _rng_reset, _rng_action, _rng_step = jax.random.split(jax.random.PRNGKey(0), 4)
    # Reset the environment state (this resets to a random level)
    obs, env_state = env.reset(_rng_reset, env_params)

    # Take a step in the environment
    action = env.action_space(env_params).sample(_rng_action)
    obs, env_state, reward, done, info = env.step(_rng_step, env_state, action, env_params)

    inverse_render_fn = make_inverse_render_symbolic(env_state, env_params, static_env_params)

    # Render environment
    renderer = make_render_pixels(env_params, env.static_env_params.replace(downscale=1))

    pixels = renderer(env_state)

    env_state_from_symbolic = inverse_render_fn(obs)
    pixels_from_symbolic = renderer(env_state_from_symbolic)

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Original")
    axs[0].imshow(pixels.astype(jnp.uint8).transpose(1, 0, 2)[::-1])
    axs[1].set_title("From symbolic")
    axs[1].imshow(pixels_from_symbolic.astype(jnp.uint8).transpose(1, 0, 2)[::-1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
