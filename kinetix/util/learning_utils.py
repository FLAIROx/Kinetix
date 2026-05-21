import jax
import jax.numpy as jnp
from kinetix.data.bc_types import Any
from kinetix.util.learning import RunningMeanStandard, rms_is_leaf, rms_normalise, rms_should_normalise, rms_update


def maybe_normalise(static_should_normalise: bool, rms: RunningMeanStandard, obs: Any):
    if static_should_normalise:
        return rms_normalise(rms, obs, flatten="auto")
    return obs


def parallel_rms_update(rms: RunningMeanStandard, data, axis_name="devices"):
    def _update_single(leaf_rms: RunningMeanStandard, arr):
        if not rms_should_normalise(leaf_rms.mean):
            return leaf_rms
        n_rms_dims = len(leaf_rms.mean.shape)
        arr = arr.reshape((-1, *arr.shape[-n_rms_dims:]))

        count_local = arr.shape[0]
        sum_local = jnp.sum(arr, axis=0)
        sum_sq_local = jnp.sum(arr**2, axis=0)

        count_global, sum_global, sum_sq_global = jax.lax.psum(
            (count_local, sum_local, sum_sq_local), axis_name=axis_name
        )

        batch_mean = sum_global / count_global
        batch_m2 = jnp.maximum(0.0, sum_sq_global - count_global * batch_mean**2)

        is_first_step = leaf_rms.count == leaf_rms.epsilon
        n_a = leaf_rms.count
        n_b = count_global
        n_ab = n_a + n_b
        delta = batch_mean - leaf_rms.mean

        mean_ab = jnp.where(is_first_step, batch_mean, leaf_rms.mean + delta * (n_b / n_ab))
        m2_ab = leaf_rms.var * n_a + batch_m2 + (delta**2) * (n_a * n_b / n_ab)
        var_ab = jnp.where(is_first_step, batch_m2 / n_b, m2_ab / n_ab)

        return RunningMeanStandard(mean_ab, var_ab, n_ab, leaf_rms.epsilon)

    return jax.tree.map(_update_single, rms, data, is_leaf=rms_is_leaf)
