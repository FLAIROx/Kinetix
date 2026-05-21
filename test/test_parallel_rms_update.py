"""
Tests for parallel_rms_update.

The key invariant: splitting data across N devices and running parallel_rms_update
via pmap must give the same mean, var, and count as rms_update on the full data
on a single device.

Run with:
    CUDA_VISIBLE_DEVICES="" python test/test_parallel_rms_update.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
# 4 virtual CPU devices so psum actually aggregates across shards
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from kinetix.util.learning import RunningMeanStandard, rms_init, rms_update
from kinetix.util.learning_utils import parallel_rms_update

NUM_DEVICES = 4  # must match XLA_FLAGS above


def _make_rms(shape, epsilon=1e-4):
    return rms_init(jnp.zeros(shape, dtype=jnp.float32), epsilon=epsilon)


def _run_parallel(rms: RunningMeanStandard, full_data: jnp.ndarray) -> RunningMeanStandard:
    """Run parallel_rms_update via pmap, return the (identical) result from device 0."""
    n = full_data.shape[0]
    assert n % NUM_DEVICES == 0, "data length must be divisible by NUM_DEVICES"
    sharded = full_data.reshape(NUM_DEVICES, n // NUM_DEVICES, *full_data.shape[1:])
    replicated_rms = jax.tree.map(lambda x: jnp.stack([x] * NUM_DEVICES), rms)

    @partial(jax.pmap, axis_name="devices")
    def _step(r, d):
        return parallel_rms_update(r, d)

    result = _step(replicated_rms, sharded)
    # jax.device_get strips pmap sharding so the result can be reused in subsequent calls
    return jax.tree.map(lambda x: jnp.array(jax.device_get(x)[0]), result)


def _assert_rms_close(tc: unittest.TestCase, expected: RunningMeanStandard, actual: RunningMeanStandard, tag=""):
    np.testing.assert_allclose(
        np.array(expected.mean), np.array(actual.mean), rtol=1e-5, atol=1e-5, err_msg=f"{tag} mean mismatch"
    )
    np.testing.assert_allclose(
        np.array(expected.var), np.array(actual.var), rtol=1e-5, atol=1e-5, err_msg=f"{tag} var mismatch"
    )
    np.testing.assert_allclose(float(expected.count), float(actual.count), rtol=1e-5, err_msg=f"{tag} count mismatch")


class TestParallelRmsUpdateMatchesSerial(unittest.TestCase):
    """Parallel update must match serial rms_update on the full dataset."""

    def test_1d_mean_fresh_rms(self):
        rng = np.random.default_rng(0)
        N, D = NUM_DEVICES * 64, 8
        data = jnp.array(rng.normal(3.0, 2.0, (N, D)).astype(np.float32))
        rms = _make_rms((D,))

        expected = rms_update(rms, data, flatten="auto")
        actual = _run_parallel(rms, data)
        _assert_rms_close(self, expected, actual, "1d fresh")

    def test_entity_obs_shape(self):
        """(batch, entity, feature) input — entities count as extra samples for the feature-dim RMS."""
        rng = np.random.default_rng(1)
        N, E, D = NUM_DEVICES * 32, 10, 16
        data = jnp.array(rng.normal(0.0, 1.0, (N, E, D)).astype(np.float32))
        rms = _make_rms((D,))

        expected = rms_update(rms, data, flatten="auto")
        actual = _run_parallel(rms, data)
        _assert_rms_close(self, expected, actual, "entity obs")

    def test_sequential_updates(self):
        """Two rounds of parallel updates must match two rounds of serial updates."""
        rng = np.random.default_rng(2)
        N, D = NUM_DEVICES * 64, 6
        data1 = jnp.array(rng.normal(1.0, 1.0, (N, D)).astype(np.float32))
        data2 = jnp.array(rng.normal(-2.0, 3.0, (N, D)).astype(np.float32))
        rms0 = _make_rms((D,))

        rms_serial = rms_update(rms_update(rms0, data1, flatten="auto"), data2, flatten="auto")

        rms_par = _run_parallel(_run_parallel(rms0, data1), data2)

        _assert_rms_close(self, rms_serial, rms_par, "sequential")

    def test_known_statistics(self):
        """Constant data → var=0; ±1 data → mean=0, var=1."""
        N, D = NUM_DEVICES * 32, 4
        rms = _make_rms((D,))

        # All fives
        data = jnp.ones((N, D)) * 5.0
        result = _run_parallel(rms, data)
        np.testing.assert_allclose(np.array(result.mean), 5.0, atol=1e-5, err_msg="constant mean")
        np.testing.assert_allclose(np.array(result.var), 0.0, atol=1e-5, err_msg="constant var")

        # Alternating rows of +1 / -1: population mean=0, population var=1
        signs = jnp.where(jnp.arange(N)[:, None] % 2 == 0, 1.0, -1.0)
        vals = signs * jnp.ones((N, D), dtype=jnp.float32)
        result2 = _run_parallel(rms, vals)
        np.testing.assert_allclose(np.array(result2.mean), 0.0, atol=1e-5, err_msg="alt mean")
        np.testing.assert_allclose(np.array(result2.var), 1.0, atol=1e-4, err_msg="alt var")

    def test_naive_pmean_var_is_biased(self):
        """Document that naive pmean of per-shard variances underestimates when shard means differ."""
        rng = np.random.default_rng(3)
        D = 4
        # Two shards with very different means so the bias is large
        shard_a = jnp.zeros((64, D))  # mean=0, var=0
        shard_b = jnp.ones((64, D)) * 100.0  # mean=100, var=0
        full_data = jnp.concatenate([shard_a, shard_b], axis=0)

        rms = _make_rms((D,))
        correct = rms_update(rms, full_data, flatten="auto")  # mean=50, var=2500

        # Naive pmean: average the two per-shard variances (both 0) → var=0, which is wrong
        rms_a = rms_update(rms, shard_a, flatten="auto")
        rms_b = rms_update(rms, shard_b, flatten="auto")
        naive_var = (rms_a.var + rms_b.var) / 2  # = 0, not 2500

        self.assertFalse(
            np.allclose(np.array(naive_var), np.array(correct.var), atol=1.0), "naive pmean should give wrong variance"
        )
        # Our parallel update gives the correct answer
        actual = _run_parallel(rms, full_data)
        np.testing.assert_allclose(
            np.array(actual.var), np.array(correct.var), rtol=1e-4, err_msg="parallel var should match correct"
        )


if __name__ == "__main__":
    unittest.main()
