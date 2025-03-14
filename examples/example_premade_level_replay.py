import jax
import jax.numpy as jnp
import jax.random
from matplotlib import pyplot as plt

from kinetix.environment import make_kinetix_env
from kinetix.environment.utils import ActionType, ObservationType
from kinetix.render import make_render_pixels
from kinetix.util import load_from_json_file


def main():
    # Load a premade level
    level, static_env_params, env_params = load_from_json_file("./kinetix/levels/l/grasp_easy.json")

    # Create the environment
    env = make_kinetix_env(
        action_type=ActionType.CONTINUOUS,
        observation_type=ObservationType.PIXELS,
        reset_fn=lambda rng: level,
        env_params=env_params,
        static_env_params=static_env_params,
    )

    rng, _rng_reset, _rng_action, _rng_step = jax.random.split(jax.random.PRNGKey(0), 4)
    # Reset the environment state---this resets to the premade level based on the reset function
    obs, env_state = env.reset(_rng_reset, env_params)

    # Take a step in the environment
    action = env.action_space(env_params).sample(_rng_action)
    obs, env_state, reward, done, info = env.step(_rng_step, env_state, action, env_params)

    # Render environment
    renderer = make_render_pixels(env_params, static_env_params)

    pixels = renderer(env_state)
    plt.imshow(pixels.astype(jnp.uint8).transpose(1, 0, 2)[::-1])
    plt.show()


if __name__ == "__main__":
    main()
