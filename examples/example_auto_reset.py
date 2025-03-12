import jax
import jax.numpy as jnp
import jax.random
from matplotlib import pyplot as plt

from kinetix.environment import EnvParams, make_kinetix_env
from kinetix.environment.env_state import StaticEnvParams
from kinetix.environment.ued.ued import make_reset_fn_sample_kinetix_level
from kinetix.environment.utils import ActionType, ObservationType
from kinetix.render import make_render_pixels


# This applies normal gymnax auto-reset behaviour, resetting to a different randomly-generated level each time.
def main():
    env_params = EnvParams()
    static_env_params = StaticEnvParams()
    # Create the environment
    env = make_kinetix_env(
        observation_type=ObservationType.PIXELS,
        action_type=ActionType.CONTINUOUS,
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

    # Render environment
    renderer = make_render_pixels(env_params, env.static_env_params)

    pixels = renderer(env_state)

    plt.imshow(pixels.astype(jnp.uint8).transpose(1, 0, 2)[::-1])
    plt.show()


if __name__ == "__main__":
    main()
