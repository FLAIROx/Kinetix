"""
KinetixEnv methods are jitted with the env as a static argument, so two envs that hash (and compare) equal share
compiled traces. Envs whose observations differ must therefore hash differently, otherwise the second env silently
returns observations computed with the first env's configuration.

Run with:  python test/test_env_hash.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import unittest

import jax

from kinetix.environment.env import KinetixEnv
from kinetix.environment.env_state import EnvParams, StaticEnvParams
from kinetix.environment.spaces import EntityObservations, MultiDiscreteActions
from kinetix.environment.utils import create_empty_env


def _make_env(ignore_mask: bool) -> KinetixEnv:
    env_params, static_env_params = EnvParams(), StaticEnvParams()
    return KinetixEnv(
        action_type=MultiDiscreteActions(env_params, static_env_params),
        observation_type=EntityObservations(env_params, static_env_params, ignore_mask=ignore_mask),
        static_env_params=static_env_params,
        reset_function=None,
    )


class TestEnvHash(unittest.TestCase):
    def test_observation_options_change_hash(self):
        self.assertEqual(hash(_make_env(False)), hash(_make_env(False)))
        self.assertNotEqual(hash(_make_env(False)), hash(_make_env(True)))
        self.assertNotEqual(_make_env(False), _make_env(True))

    def test_jitted_reset_uses_each_envs_observations(self):
        env_with_mask, env_without_mask = _make_env(False), _make_env(True)
        state = create_empty_env(StaticEnvParams())
        rng = jax.random.PRNGKey(0)

        obs_with_mask, _ = env_with_mask.reset(rng, EnvParams(), state)
        # Before the fix, this reused the trace from env_with_mask and returned an attention mask.
        obs_without_mask, _ = env_without_mask.reset(rng, EnvParams(), state)

        self.assertIsNotNone(obs_with_mask.attention_mask)
        self.assertIsNone(obs_without_mask.attention_mask)


if __name__ == "__main__":
    unittest.main()
