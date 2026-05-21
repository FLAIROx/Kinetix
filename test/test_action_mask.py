"""
Test that get_valid_action_mask_np matches get_valid_action_mask (JAX).

Run with:  CUDA_VISIBLE_DEVICES="" python tests/test_action_mask.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import unittest
import numpy as np
import jax
import jax.numpy as jnp

from kinetix.environment.env_state import StaticEnvParams
from kinetix.environment.utils import create_empty_env
from kinetix.data.bc_utils import get_valid_action_mask, get_valid_action_mask_np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_static(
    num_polygons=6, num_circles=3, num_joints=2, num_thrusters=2, num_motor_bindings=4, num_thruster_bindings=2
):
    return StaticEnvParams().replace(
        num_polygons=num_polygons,
        num_circles=num_circles,
        num_joints=num_joints,
        num_thrusters=num_thrusters,
        num_motor_bindings=num_motor_bindings,
        num_thruster_bindings=num_thruster_bindings,
    )


def _make_batch_of_env_states(static: StaticEnvParams, n: int, rng: np.random.Generator):
    """Return a batched EnvState (n copies of create_empty_env with randomised
    active flags and bindings) as a pytree of numpy arrays."""
    base = create_empty_env(static)

    action_dim = static.num_motor_bindings + static.num_thruster_bindings

    # Random joint active / is_fixed_joint
    joint_active = rng.integers(0, 2, size=(n, static.num_joints)).astype(bool)
    joint_fixed = rng.integers(0, 2, size=(n, static.num_joints)).astype(bool)
    # Random motor bindings in [0, num_motor_bindings)
    motor_bindings = rng.integers(0, static.num_motor_bindings, size=(n, static.num_joints)).astype(np.int32)
    # Random thruster active
    thruster_active = rng.integers(0, 2, size=(n, static.num_thrusters)).astype(bool)
    # Random thruster bindings in [0, num_thruster_bindings)
    thruster_bindings = rng.integers(0, static.num_thruster_bindings, size=(n, static.num_thrusters)).astype(np.int32)

    env_state = jax.tree.map(
        lambda x: np.tile(np.array(x)[None], (n,) + (1,) * len(np.array(x).shape)),
        base,
    )
    env_state = env_state.replace(
        joint=env_state.joint.replace(
            active=joint_active,
            is_fixed_joint=joint_fixed,
        ),
        motor_bindings=motor_bindings,
        thruster=env_state.thruster.replace(
            active=thruster_active,
        ),
        thruster_bindings=thruster_bindings,
    )
    action = np.zeros((n, action_dim), dtype=np.int32)
    return env_state, action


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestActionMaskNp(unittest.TestCase):
    def _compare(self, static, n, seed):
        rng = np.random.default_rng(seed)
        env_state, action = _make_batch_of_env_states(static, n, rng)

        mask_jax = np.asarray(get_valid_action_mask(env_state, static, action))
        mask_np = get_valid_action_mask_np(env_state, static, action)

        np.testing.assert_array_equal(
            mask_np,
            mask_jax,
            err_msg=f"Mismatch for static={static}, n={n}, seed={seed}",
        )

    def test_medium_small_batch(self):
        self._compare(_make_static(), n=64, seed=0)

    def test_medium_large_batch(self):
        self._compare(_make_static(), n=4096, seed=1)

    def test_single_transition(self):
        self._compare(_make_static(), n=1, seed=2)

    def test_different_sizes(self):
        for num_joints, num_thrusters, num_mb, num_tb in [
            (1, 1, 2, 1),
            (4, 2, 6, 2),
            (2, 1, 4, 1),
        ]:
            static = _make_static(
                num_joints=num_joints,
                num_thrusters=num_thrusters,
                num_motor_bindings=num_mb,
                num_thruster_bindings=num_tb,
            )
            with self.subTest(num_joints=num_joints, num_thrusters=num_thrusters):
                self._compare(static, n=256, seed=3)

    def test_all_active(self):
        """All joints active and non-fixed — every motor binding slot should be True."""
        static = _make_static(num_joints=2, num_thrusters=0, num_motor_bindings=4, num_thruster_bindings=0)
        rng = np.random.default_rng(42)
        env_state, action = _make_batch_of_env_states(static, 128, rng)

        # Force all joints active and non-fixed
        env_state = env_state.replace(
            joint=env_state.joint.replace(
                active=np.ones((128, static.num_joints), dtype=bool),
                is_fixed_joint=np.zeros((128, static.num_joints), dtype=bool),
            )
        )
        mask_jax = np.asarray(get_valid_action_mask(env_state, static, action))
        mask_np = get_valid_action_mask_np(env_state, static, action)
        np.testing.assert_array_equal(mask_np, mask_jax)

    def test_none_active(self):
        """No joints or thrusters active — mask should be all False."""
        static = _make_static()
        rng = np.random.default_rng(99)
        env_state, action = _make_batch_of_env_states(static, 64, rng)

        env_state = env_state.replace(
            joint=env_state.joint.replace(
                active=np.zeros((64, static.num_joints), dtype=bool),
            ),
            thruster=env_state.thruster.replace(
                active=np.zeros((64, static.num_thrusters), dtype=bool),
            ),
        )
        mask_jax = np.asarray(get_valid_action_mask(env_state, static, action))
        mask_np = get_valid_action_mask_np(env_state, static, action)
        np.testing.assert_array_equal(mask_np, mask_jax)
        self.assertTrue((mask_np == False).all())

    def test_fixed_joints_excluded(self):
        """Fixed joints should not set their binding slot to True."""
        static = _make_static(num_joints=2, num_thrusters=0, num_motor_bindings=4, num_thruster_bindings=0)
        rng = np.random.default_rng(7)
        env_state, action = _make_batch_of_env_states(static, 64, rng)

        env_state = env_state.replace(
            joint=env_state.joint.replace(
                active=np.ones((64, static.num_joints), dtype=bool),
                is_fixed_joint=np.ones((64, static.num_joints), dtype=bool),  # all fixed
            ),
        )
        mask_jax = np.asarray(get_valid_action_mask(env_state, static, action))
        mask_np = get_valid_action_mask_np(env_state, static, action)
        np.testing.assert_array_equal(mask_np, mask_jax)
        self.assertTrue((mask_np == False).all())


if __name__ == "__main__":
    unittest.main()
