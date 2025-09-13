import jax
import jax.numpy as jnp
import jax.random
from matplotlib import pyplot as plt

from kinetix.environment import EnvParams, UEDParams, make_kinetix_env, sample_kinetix_level, StaticEnvParams
from kinetix.environment import ActionType, ObservationType
from kinetix.render import make_render_pixels


def main():
    # Use default parameters
    env_params = EnvParams()
    ued_params = UEDParams()
    static_env_params = StaticEnvParams()

    # Create the environment
    env = make_kinetix_env(
        action_type=ActionType.CONTINUOUS,
        observation_type=ObservationType.PIXELS,
        reset_fn=None,
        env_params=env_params,
        static_env_params=static_env_params,
    )

    # Sample a random level
    rng, _rng_sample, _rng_reset, _rng_action, _rng_step = jax.random.split(jax.random.PRNGKey(0), 5)
    level = sample_kinetix_level(_rng_sample, env.physics_engine, env_params, env.static_env_params, ued_params)

    # Reset the environment state to this level
    # Different to normal gymnax, we give the level we want to reset to
    obs, env_state = env.reset(_rng_reset, env_params, level)

    # Take a step in the environment
    action = env.action_space(env_params).sample(_rng_action)
    # The final level argument is the level to be reset to if the environment is done
    obs, env_state, reward, done, info = env.step(_rng_step, env_state, action, env_params, level)

    # Render environment
    renderer = make_render_pixels(env_params, env.static_env_params)

    pixels = renderer(env_state)

    plt.imshow(pixels.astype(jnp.uint8).transpose(1, 0, 2)[::-1])
    plt.show()


if __name__ == "__main__":
    main()
