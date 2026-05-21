"""
Tests for expand_env_state_numpy (single) and expand_env_state_numpy_batched.

Single:  compared against JAX expand_env_state.
Batched: compared against jax.vmap(expand_env_state, in_axes=(0, None)).

Run with:  CUDA_VISIBLE_DEVICES="" python tests/test_expand_env_state.py
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
from kinetix.util.saving import expand_env_state
from kinetix.data.bc_utils import expand_env_state_numpy, expand_env_state_numpy_batched


# ---------------------------------------------------------------------------
# Size presets (mirrors configs/env_size/{s,m,l}.yaml)
# ---------------------------------------------------------------------------


def _make_static(num_polygons, num_circles, num_joints, num_thrusters):
    return StaticEnvParams().replace(
        num_polygons=num_polygons,
        num_circles=num_circles,
        num_joints=num_joints,
        num_thrusters=num_thrusters,
    )


STATIC_S = _make_static(5, 2, 1, 1)
STATIC_M = _make_static(6, 3, 2, 2)
STATIC_L = _make_static(12, 4, 6, 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(static: StaticEnvParams):
    """Return a non-trivial EnvState at the given size.

    Activates one polygon, one circle, and one joint whose b_index references
    the circle (unified index = num_polygons), so _modify_index is exercised
    whenever polygons are added during expansion.
    """
    state = create_empty_env(static)

    state = state.replace(
        polygon=state.polygon.replace(
            active=state.polygon.active.at[4].set(True),
            position=state.polygon.position.at[4].set(jnp.array([1.0, 2.0])),
        ),
        polygon_shape_roles=state.polygon_shape_roles.at[4].set(1),
    )

    state = state.replace(
        circle=state.circle.replace(
            active=state.circle.active.at[0].set(True),
            position=state.circle.position.at[0].set(jnp.array([0.5, 0.5])),
            radius=state.circle.radius.at[0].set(0.3),
        ),
    )

    # Joint connecting polygon 4 → circle 0.  Circle 0 lives at unified index
    # num_polygons, which is >= num_rects, so _modify_index will shift it when
    # polygons are appended.
    b_unified = static.num_polygons
    state = state.replace(
        joint=state.joint.replace(
            active=state.joint.active.at[0].set(True),
            a_index=state.joint.a_index.at[0].set(4),
            b_index=state.joint.b_index.at[0].set(b_unified),
        ),
        thruster=state.thruster.replace(
            active=state.thruster.active.at[0].set(True),
            object_index=state.thruster.object_index.at[0].set(b_unified),
        ),
    )

    return state


def _compare(ref, got, label: str):
    """Assert every pytree leaf is numerically identical."""
    ref_leaves, ref_treedef = jax.tree.flatten(ref)
    got_leaves, got_treedef = jax.tree.flatten(got)
    assert ref_treedef == got_treedef, f"[{label}] pytree structures differ"
    for i, (r, g) in enumerate(zip(ref_leaves, got_leaves)):
        np.testing.assert_array_equal(
            np.asarray(r),
            np.asarray(g),
            err_msg=f"[{label}] leaf {i} mismatch",
        )


# ---------------------------------------------------------------------------
# Single-state tests (numpy vs JAX)
# ---------------------------------------------------------------------------


class TestExpandEnvStateNumpy(unittest.TestCase):
    def _run(self, src_static, dst_static, label):
        state = _make_state(src_static)
        jax_result = expand_env_state(state, dst_static)
        np_result = expand_env_state_numpy(state, dst_static)
        _compare(jax_result, np_result, label)

    def test_s_to_m(self):
        self._run(STATIC_S, STATIC_M, "s→m")

    def test_s_to_l(self):
        self._run(STATIC_S, STATIC_L, "s→l")

    def test_m_to_l(self):
        self._run(STATIC_M, STATIC_L, "m→l")

    def test_noop(self):
        self._run(STATIC_S, STATIC_S, "s→s noop")

    def test_index_shift(self):
        """b_index / object_index referencing a circle must shift by added_rects."""
        state = _make_state(STATIC_S)
        old_b = int(state.joint.b_index[0])
        old_obj = int(state.thruster.object_index[0])

        result = expand_env_state_numpy(state, STATIC_M)
        added = STATIC_M.num_polygons - STATIC_S.num_polygons

        self.assertEqual(int(result.joint.b_index[0]), old_b + added)
        self.assertEqual(int(result.thruster.object_index[0]), old_obj + added)


# ---------------------------------------------------------------------------
# Batched tests (numpy_batched vs jax.vmap)
# ---------------------------------------------------------------------------


class TestExpandEnvStateNumpyBatched(unittest.TestCase):
    def _make_batch(self, src_static, n=4):
        """Stack n single states into a batched EnvState (JAX arrays)."""
        states = [_make_state(src_static) for _ in range(n)]
        return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)

    def _run(self, src_static, dst_static, label, n=4):
        batched = self._make_batch(src_static, n)

        # Reference: vmap the JAX single-state function
        vmap_result = jax.vmap(expand_env_state, in_axes=(0, None))(batched, dst_static)

        # Ours: batched numpy version
        np_result = expand_env_state_numpy_batched(batched, dst_static)

        _compare(vmap_result, np_result, label)

    def test_s_to_m(self):
        self._run(STATIC_S, STATIC_M, "batched s→m")

    def test_s_to_l(self):
        self._run(STATIC_S, STATIC_L, "batched s→l")

    def test_m_to_l(self):
        self._run(STATIC_M, STATIC_L, "batched m→l")

    def test_noop(self):
        self._run(STATIC_S, STATIC_S, "batched s→s noop")

    def test_index_shift(self):
        """Batched b_index / object_index shift must match vmap reference."""
        batched = self._make_batch(STATIC_S)
        added = STATIC_M.num_polygons - STATIC_S.num_polygons

        ref = jax.vmap(expand_env_state, in_axes=(0, None))(batched, STATIC_M)
        got = expand_env_state_numpy_batched(batched, STATIC_M)

        np.testing.assert_array_equal(
            np.asarray(ref.joint.b_index),
            np.asarray(got.joint.b_index),
            err_msg="batched joint b_index mismatch",
        )
        np.testing.assert_array_equal(
            np.asarray(ref.thruster.object_index),
            np.asarray(got.thruster.object_index),
            err_msg="batched thruster object_index mismatch",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
