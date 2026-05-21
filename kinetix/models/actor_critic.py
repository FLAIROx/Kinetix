import functools
from typing import Any, List, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from kinetix.environment.spaces import ActionType
from kinetix.models.action_spaces import (
    HybridActionDistribution,
    MultiDiscreteActionDistribution,
    TemperatureCategorical,
)


@jax.jit
def _calculate_dormancy(activations, threshold=0.0):
    # partially from alex rutherford
    h = jnp.abs(activations).reshape(-1, activations.shape[-1])  # make shape of (batch, num_neurons)
    per_neuron = jnp.mean(h, axis=0)  # Mean activation per neuron
    assert len(per_neuron.shape) == 1, f"Expected per_neuron to be 1D, got {per_neuron.shape}"
    num_neurons = per_neuron.shape[0]
    score = per_neuron / (per_neuron.sum() + 1e-8)  # this is the score per neuron.
    num_dormant_neurons = jnp.where(score <= threshold, 1, 0).sum()

    return jax.lax.stop_gradient(num_dormant_neurons * 1.0), jax.lax.stop_gradient(num_neurons * 1.0)


def _log_dormancy(self, activations, name, threshold=0.0):
    num_dormant_neurons, num_neurons = _calculate_dormancy(activations, threshold)
    self.sow("custom_intermediates", f"dormancy_{name}", (num_dormant_neurons, num_neurons))


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], 256),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=256)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size=256):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=256)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class DenseAndActivation(nn.Module):
    width: int
    activation: Any

    @nn.compact
    def __call__(self, x):
        ans = nn.Sequential(
            [
                nn.Dense(
                    self.width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                ),
                self.activation,
            ]
        )(x)
        _log_dormancy(self, ans, "dense_and_activation")
        return ans


class GeneralActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    actor_depth: int
    critic_depth: int
    actor_width: int
    critic_width: int
    action_type: ActionType
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    add_generator_embedding: bool = False
    generator_embedding_number_of_timesteps: int = 10
    recurrent: bool = False

    # Given an embedding, return the action/values, since this is shared across all models.
    @nn.compact
    def __call__(self, hidden, obs, dones, activation, embedding=None, actor_embedding=None, critic_embedding=None):

        if self.add_generator_embedding:
            raise NotImplementedError()

        if embedding is None:
            assert actor_embedding is not None and critic_embedding is not None
            if self.recurrent:
                hidden_actor, hidden_critic = hidden
                hidden_actor, actor_embedding = ScannedRNN()(hidden_actor, (actor_embedding, dones))
                hidden_critic, critic_embedding = ScannedRNN()(hidden_critic, (critic_embedding, dones))

                hidden = (hidden_actor, hidden_critic)
        else:
            if self.recurrent:
                new_hidden, embedding = ScannedRNN()(hidden[0], (embedding, dones))
                hidden = (new_hidden, hidden[1])
            actor_embedding = embedding
            critic_embedding = embedding

        def _get_model(width):
            return DenseAndActivation(width, activation)

        for _ in range(self.actor_depth):
            actor_embedding = _get_model(self.actor_width)(actor_embedding)
        for _ in range(self.critic_depth):
            critic_embedding = _get_model(self.critic_width)(critic_embedding)

        self.sow("custom_intermediates", "actor_feature_matrix", actor_embedding)
        self.sow("custom_intermediates", "critic_feature_matrix", critic_embedding)

        actor_mean_last = actor_embedding
        actor_embedding = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(
            actor_embedding
        )
        if self.action_type == ActionType.DISCRETE:
            pi = TemperatureCategorical(logits=actor_embedding)
        elif self.action_type == ActionType.CONTINUOUS:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_embedding, jnp.exp(actor_logtstd))
        elif self.action_type == ActionType.MULTI_DISCRETE:
            pi = MultiDiscreteActionDistribution(actor_embedding, self.multi_discrete_number_of_dims_per_distribution)
        else:
            actor_mean_continuous = nn.Dense(
                self.hybrid_action_continuous_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean_last)
            actor_mean_sigma = jnp.exp(
                nn.Dense(self.hybrid_action_continuous_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(
                    actor_mean_last
                )
            )
            pi = HybridActionDistribution(actor_embedding, actor_mean_continuous, actor_mean_sigma)

        critic_embedding = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic_embedding)
        return hidden, pi, jnp.squeeze(critic_embedding, axis=-1)

    @staticmethod
    def initialize_carry(batch_size, hidden_size=256):
        return (
            ScannedRNN.initialize_carry(batch_size, hidden_size),
            ScannedRNN.initialize_carry(batch_size, hidden_size),
        )


class ResNetBasicBlock(nn.Module):
    features: int
    strides: int = 1

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Conv(self.features, kernel_size=(3, 3), strides=(self.strides, self.strides), padding="SAME")(x)
        y = nn.GroupNorm(num_groups=min(32, self.features))(y)
        y = nn.relu(y)
        y = nn.Conv(self.features, kernel_size=(3, 3), padding="SAME")(y)
        y = nn.GroupNorm(num_groups=min(32, self.features))(y)
        if self.strides != 1 or residual.shape[-1] != self.features:
            residual = nn.Conv(self.features, kernel_size=(1, 1), strides=(self.strides, self.strides))(residual)
            residual = nn.GroupNorm(num_groups=min(32, self.features))(residual)
        return nn.relu(y + residual)


class ActorCriticPixelsRNN(nn.Module):

    action_dim: Sequence[int]
    actor_depth: int
    critic_depth: int
    actor_width: int
    critic_width: int
    action_mode: str
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    activation: str
    add_generator_embedding: bool = False
    generator_embedding_number_of_timesteps: int = 10
    recurrent: bool = True
    cnn_mode: str = "impala_fast"

    @nn.compact
    def __call__(self, hidden, x, **kwargs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        og_obs, dones = x

        if self.add_generator_embedding:
            obs = og_obs.obs
        else:
            obs = og_obs

        image = obs.image
        global_info = obs.global_info

        x = image
        if self.cnn_mode == "impala_fast":
            x = nn.Conv(features=16, kernel_size=(8, 8), strides=(4, 4))(x)
            x = nn.relu(x)
            x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
            x = nn.relu(x)
        elif self.cnn_mode == "resnet_18":
            x = nn.Conv(features=64, kernel_size=(7, 7), strides=(2, 2), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=32)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
            for stage_features, stage_stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
                x = ResNetBasicBlock(stage_features, strides=stage_stride)(x)
                x = ResNetBasicBlock(stage_features, strides=1)(x)
            x = x.mean(axis=(-3, -2), keepdims=True)
        else:
            raise ValueError(f"Unknown mode {self.cnn_mode}")
        embedding = x.reshape(x.shape[0], x.shape[1], -1)

        embedding = jnp.concatenate([embedding, global_info], axis=-1)

        return GeneralActorCriticRNN(
            action_dim=self.action_dim,
            actor_depth=self.actor_depth,
            critic_depth=self.critic_depth,
            actor_width=self.actor_width,
            critic_width=self.critic_width,
            action_type=self.action_mode,
            hybrid_action_continuous_dim=self.hybrid_action_continuous_dim,
            multi_discrete_number_of_dims_per_distribution=self.multi_discrete_number_of_dims_per_distribution,
            add_generator_embedding=self.add_generator_embedding,
            generator_embedding_number_of_timesteps=self.generator_embedding_number_of_timesteps,
            recurrent=self.recurrent,
        )(hidden, og_obs, dones, activation, embedding)

    @staticmethod
    def initialize_carry(batch_size, hidden_size=256):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)


class ActorCriticSymbolicRNN(nn.Module):
    action_dim: Sequence[int]
    actor_depth: int
    critic_depth: int
    actor_width: int
    critic_width: int
    action_mode: str
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    activation: str
    add_generator_embedding: bool = False
    generator_embedding_number_of_timesteps: int = 10
    recurrent: bool = True

    @nn.compact
    def __call__(self, hidden, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        og_obs, dones = x
        if self.add_generator_embedding:
            obs = og_obs.obs
        else:
            obs = og_obs

        embedding = nn.Dense(
            self.actor_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        return GeneralActorCriticRNN(
            action_dim=self.action_dim,
            actor_depth=self.actor_depth,
            critic_depth=self.critic_depth,
            actor_width=self.actor_width,
            critic_width=self.critic_width,
            action_type=self.action_mode,
            hybrid_action_continuous_dim=self.hybrid_action_continuous_dim,
            multi_discrete_number_of_dims_per_distribution=self.multi_discrete_number_of_dims_per_distribution,
            add_generator_embedding=self.add_generator_embedding,
            generator_embedding_number_of_timesteps=self.generator_embedding_number_of_timesteps,
            recurrent=self.recurrent,
        )(hidden, og_obs, dones, activation, embedding)

    @staticmethod
    def initialize_carry(batch_size, hidden_size=256):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)


class MultiHeadDense(nn.Module):
    num_heads: int  # Number of heads
    out_dim: int  # Output dimension for each head
    kernel_init: nn.initializers.Initializer
    bias_init: nn.initializers.Initializer

    @nn.compact
    def __call__(self, x):
        # x has shape (...., num_features, feature_dim)
        num_features, feature_dim = x.shape[-2:]

        # Initialize a Dense layer for each head, stacked as (num_heads, feature_dim, out_dim)
        dense_kernels = self.param("dense_kernels", self.kernel_init, (self.num_heads, feature_dim, self.out_dim))
        dense_biases = self.param("dense_biases", self.bias_init, (self.num_heads, self.out_dim))

        # Apply the dense layer to each head by broadcasting and matrix multiplying
        x_expanded = jnp.expand_dims(x, axis=-2)  # Shape: (..., num_features, 1, feature_dim)
        output = jnp.einsum("...fhd,hdo->...fho", x_expanded, dense_kernels) + dense_biases
        output = nn.relu(output)  # Shape: (..., num_features, num_heads, out_dim)
        output = output.sum(axis=-3)  # Shape=(..., num_heads, out_dim)

        output = output.reshape((*output.shape[:-2], self.num_heads * self.out_dim))  # Shape=(..., num_heads * out_dim)
        return output


class ActorCriticPermutationInvariantSymbolicRNN(nn.Module):
    action_dim: Sequence[int]
    symbolic_embedding_dim: int
    actor_depth: int
    critic_depth: int
    actor_width: int
    critic_width: int
    action_mode: str
    separate_actor_critic: bool
    multi_discrete_number_of_dims_per_distribution: List[int]
    hybrid_action_continuous_dim: int
    activation: str
    recurrent: bool
    add_generator_embedding: bool = False
    include_actions_and_rewards: bool = False
    permutation_invariant: bool = True
    num_heads: int = None
    preprocess_separately: bool = False
    encoder_size: int = 64

    @nn.compact
    def __call__(self, hidden, x):
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        og_obs, dones = x
        if self.add_generator_embedding:
            obs = og_obs.obs
        else:
            obs = og_obs

        if self.permutation_invariant:
            assert (
                self.symbolic_embedding_dim % self.num_heads == 0
            ), f"{self.symbolic_embedding_dim=} must be divisible by {self.num_heads=}"

            if self.preprocess_separately:

                def _single_encoder(features, entity_id):
                    num_to_remove = 4
                    embedding = activation_fn(
                        nn.Dense(
                            self.encoder_size - num_to_remove,
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0),
                        )(features)
                    )
                    _log_dormancy(self, embedding, f"single_encoder_{entity_id}")
                    id_1h = jnp.zeros((*embedding.shape[:3], 4)).at[:, :, :, entity_id].set(1)
                    return jnp.concatenate([embedding, id_1h], axis=-1)

                circle_encodings = _single_encoder(obs.circles, 0)
                polygon_encodings = _single_encoder(obs.polygons, 1)
                joint_encodings = _single_encoder(obs.joints, 2)
                thruster_encodings = _single_encoder(obs.thrusters, 3)

                all_encodings = jnp.concatenate(
                    [polygon_encodings, circle_encodings, joint_encodings, thruster_encodings], axis=2
                )
                all_mask = jnp.concatenate(
                    [obs.polygon_mask, obs.circle_mask, obs.joint_mask, obs.thruster_mask], axis=2
                )

                def mask(features, mask):
                    return jnp.where(mask[:, None], features, jnp.zeros_like(features))

                obs = jax.vmap(jax.vmap(mask))(all_encodings, all_mask)

            dim_per_head = self.symbolic_embedding_dim // self.num_heads

            def _get_embed():
                return activation_fn(
                    MultiHeadDense(
                        num_heads=self.num_heads,
                        out_dim=dim_per_head,
                        kernel_init=orthogonal(np.sqrt(2)),
                        bias_init=constant(0.0),
                    )(obs)
                )

            if self.separate_actor_critic:
                actor_embedding = _get_embed()
                critic_embedding = _get_embed()
                _log_dormancy(self, actor_embedding, "mh_actor_embed")
                _log_dormancy(self, critic_embedding, "mh_critic_embed")
                args = dict(actor_embedding=actor_embedding, critic_embedding=critic_embedding)
            else:
                embedding = _get_embed()
                _log_dormancy(self, embedding, "mh_joint_embed")
                args = dict(embedding=embedding)

        return GeneralActorCriticRNN(
            action_dim=self.action_dim,
            actor_depth=self.actor_depth,
            critic_depth=self.critic_depth,
            actor_width=self.actor_width,
            critic_width=self.critic_width,
            action_type=self.action_mode,
            hybrid_action_continuous_dim=self.hybrid_action_continuous_dim,
            multi_discrete_number_of_dims_per_distribution=self.multi_discrete_number_of_dims_per_distribution,
            recurrent=self.recurrent,
        )(hidden, og_obs, dones, activation_fn, **args)

    @staticmethod
    def initialize_carry(batch_size, hidden_dim):
        return ScannedRNN.initialize_carry(batch_size, hidden_dim)
