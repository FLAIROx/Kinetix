from typing import Any, Sequence

import distrax
import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax import struct


@struct.dataclass
class HybridAction:
    discrete: int
    continuous: jnp.ndarray


class TemperatureCategorical(distrax.Categorical):
    def _sample_n(self, key: PRNGKey, n: int, temperature: float = 1.0) -> Any:
        if temperature == 1.0:
            return super()._sample_n(key, n)
        return jax.random.categorical(key, self.logits / temperature, shape=(n,) + self.logits.shape[:-1])


class HybridActionDistribution(distrax.Distribution):
    def __init__(self, discrete_logits, continuous_mu, continuous_sigma) -> None:
        self.discrete = TemperatureCategorical(logits=discrete_logits)
        self.continuous = distrax.MultivariateNormalDiag(continuous_mu, continuous_sigma)

    def _sample_n(self, rng: PRNGKey, n: int, temperature: float = 1.0) -> Any:
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        a = self.discrete._sample_n(_rng, n, temperature=temperature)
        b = self.continuous._sample_n(_rng2, n)
        return HybridAction(a, b)

    def log_prob(self, value: Any):
        a = self.discrete.log_prob(value.discrete)
        b = self.continuous.log_prob(value.continuous)
        return a + b  # log probs, we add.

    def entropy(self):
        return self.discrete.entropy() + self.continuous.entropy()

    def event_shape(self) -> Sequence[int]:
        return ()


class MultiDiscreteActionDistribution(distrax.Distribution):
    def __init__(self, flat_logits, number_of_dims_per_distribution) -> None:
        self.distributions = []
        total_dims = 0
        for dims in number_of_dims_per_distribution:
            self.distributions.append(TemperatureCategorical(logits=flat_logits[..., total_dims : total_dims + dims]))
            total_dims += dims

    def _sample_n(self, key: PRNGKey, n: int, temperature: float = 1.0) -> Any:
        rngs = jax.random.split(key, len(self.distributions))
        samples = [
            jnp.expand_dims(d._sample_n(rng, n, temperature=temperature), axis=-1)
            for rng, d in zip(rngs, self.distributions)
        ]
        return jnp.concatenate(samples, axis=-1)

    def log_prob(self, value: Any):
        return sum(d.log_prob(value[..., i]) for i, d in enumerate(self.distributions))

    def log_prob_disentangled(self, value: Any):
        return jnp.stack([d.log_prob(value[..., i]) for i, d in enumerate(self.distributions)], axis=-1)

    def entropy(self):
        return sum(d.entropy() for d in self.distributions)

    def entropy_disentangled(self):
        return jnp.stack([d.entropy() for i, d in enumerate(self.distributions)], axis=-1)

    def event_shape(self) -> Sequence[int]:
        return ()

    def __repr__(self):
        return f"MultiDiscrete()"
