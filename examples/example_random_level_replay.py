import jax
import jax.numpy as jnp
import jax.random
from jax2d.engine import PhysicsEngine
from matplotlib import pyplot as plt

from kinetix.environment.env import make_kinetix_env_from_args
from kinetix.environment.env_state import StaticEnvParams, EnvParams
from kinetix.environment.ued.distributions import sample_kinetix_level
from kinetix.environment.ued.ued_state import UEDParams
from kinetix.render.renderer_pixels import make_render_pixels

import imageio 
import os 

def main():

    file_path = "random_level"

    # Use default parameters
    env_params = EnvParams()
    static_env_params = StaticEnvParams()
    ued_params = UEDParams()

    # num steps 
    i = 0
    num_steps = 100
    rendered_images = []

    # Create the environment
    env = make_kinetix_env_from_args(
        obs_type="pixels", 
        action_type="discrete", 
        reset_type="replay", 
        static_env_params=static_env_params,
    )
    # Render environment
    renderer = make_render_pixels(env_params, static_env_params)

    # Sample a random level
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    level = sample_kinetix_level(_rng, env.physics_engine, env_params, static_env_params, ued_params)

    # Reset the environment state to this level
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset_to_level(_rng, level, env_params)
    rendered_images.append(
        renderer(env_state.env_state.env_state.env_state)
    )  

    while i < num_steps:

        # Take a step in the environment
        rng, _rng = jax.random.split(rng)
        action = env.action_space(env_params).sample(_rng)
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

        # There are a lot of wrappers
        pixels = renderer(env_state.env_state.env_state.env_state)
        rendered_images.append(pixels)
        i += 1 

    imageio.mimsave(f"{file_path}.mp4", rendered_images, fps=8, format="mp4")

    # There are a lot of wrappers
    # pixels = renderer(env_state.env_state.env_state.env_state)

    # plt.imshow(pixels.astype(jnp.uint8).transpose(1, 0, 2)[::-1])
    # plt.show()


if __name__ == "__main__":
    main()
