from typing import List, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.attention import MultiHeadDotProductAttention
from flax.linen.initializers import constant, orthogonal

from kinetix.environment.spaces import ActionType
from kinetix.models.action_spaces import (
    HybridActionDistribution,
    MultiDiscreteActionDistribution,
    TemperatureCategorical,
)
from kinetix.models.actor_critic import DenseAndActivation, GeneralActorCriticRNN, ScannedRNN
from kinetix.render.renderer_symbolic_entity import EntityObservation


class Gating(nn.Module):
    # code taken from https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    d_input: int
    bg: float = 0.0

    @nn.compact
    def __call__(self, x, y):
        r = jax.nn.sigmoid(nn.Dense(self.d_input, use_bias=False)(y) + nn.Dense(self.d_input, use_bias=False)(x))
        z = jax.nn.sigmoid(
            nn.Dense(self.d_input, use_bias=False)(y)
            + nn.Dense(self.d_input, use_bias=False)(x)
            - self.param("gating_bias", constant(self.bg), (self.d_input,))
        )
        h = jnp.tanh(nn.Dense(self.d_input, use_bias=False)(y) + nn.Dense(self.d_input, use_bias=False)(r * x))
        g = (1 - z) * x + (z * h)
        return g


class transformer_layer(nn.Module):
    num_heads: int
    out_features: int
    qkv_features: int
    transformer_ffn: bool
    gating: bool = False
    gating_bias: float = 0.0
    dropout_prob: float = 0.0

    def setup(self):
        self.attention1 = MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.qkv_features, out_features=self.out_features
        )

        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

        self.dense1 = nn.Dense(self.out_features * 4 if self.transformer_ffn else self.out_features)
        self.dense2 = nn.Dense(self.out_features)

        if self.gating:
            self.gate1 = Gating(self.out_features, self.gating_bias)
            self.gate2 = Gating(self.out_features, self.gating_bias)

        if self.dropout_prob > 0.0:
            assert self.transformer_ffn
            self.dropout_layer = nn.Dropout(self.dropout_prob)

    def __call__(self, queries: jnp.ndarray, mask: jnp.ndarray, deterministic: bool = True):
        # After reading the paper, this is what I think we should do:
        # First layernorm, then do attention
        queries_n = self.ln1(queries)
        y = self.attention1(queries_n, mask=mask)
        if self.gating:  # and gate
            y = self.gate1(queries, jax.nn.relu(y))
        else:
            y = queries + y
        # Dense after norming, crucially no relu.
        if self.transformer_ffn:
            temp = jax.nn.relu(self.dense1(self.ln2(y)))
            if self.dropout_prob > 0.0:
                temp = self.dropout_layer(temp, deterministic=deterministic)
            e = self.dense2(temp)
        else:
            e = self.dense1(self.ln2(y))

        if self.gating:  # and gate again
            # This may be the wrong way around
            e = self.gate2(y, jax.nn.relu(e))
        else:
            e = y + e

        return e


class ThrusterAndJointBlock(nn.Module):
    encoder_size: int
    hidden_size: int
    num_layers: int = 1
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        for l in range(self.num_layers - 1):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.relu(x)

        x = nn.Dense(self.encoder_size)(x)
        if self.activate_final:
            x = nn.relu(x)
        return x


class Transformer(nn.Module):
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers: int
    transformer_ffn: bool
    multilayer_joint_thruster_mixing: bool
    joint_thruster_block_num_layers: int
    joint_thruster_block_hidden_dim_multiplication: int
    gating: bool = False
    gating_bias: float = 0.0
    dropout_prob: float = 0.0

    def setup(self):
        # self.encoder = nn.Dense(self.encoder_size)

        # self.positional_encoding = PositionalEncoding(self.encoder_size, max_len=self.max_len)

        self.tf_layers = [
            transformer_layer(
                num_heads=self.num_heads,
                qkv_features=self.qkv_features,
                out_features=self.encoder_size,
                gating=self.gating,
                gating_bias=self.gating_bias,
                transformer_ffn=self.transformer_ffn,
                dropout_prob=self.dropout_prob,
            )
            for _ in range(self.num_layers)
        ]

        if self.multilayer_joint_thruster_mixing:
            self.joint_layers = [
                ThrusterAndJointBlock(
                    encoder_size=self.encoder_size,
                    hidden_size=self.encoder_size * self.joint_thruster_block_hidden_dim_multiplication,
                    num_layers=self.joint_thruster_block_num_layers,
                    activate_final=True,
                )
                for _ in range(self.num_layers)
            ]
            self.thruster_layers = [
                ThrusterAndJointBlock(
                    encoder_size=self.encoder_size,
                    hidden_size=self.encoder_size * self.joint_thruster_block_hidden_dim_multiplication,
                    num_layers=self.joint_thruster_block_num_layers,
                    activate_final=True,
                )
                for _ in range(self.num_layers)
            ]
        else:
            self.joint_layers = [nn.Dense(self.encoder_size) for _ in range(self.num_layers)]
            self.thruster_layers = [nn.Dense(self.encoder_size) for _ in range(self.num_layers)]

        # self.pos_emb=PositionalEmbedding(self.encoder_size)

    def __call__(
        self,
        shape_embeddings: jnp.ndarray,
        shape_attention_mask,
        joint_embeddings,
        joint_mask,
        joint_indexes,
        thruster_embeddings,
        thruster_mask,
        thruster_indexes,
        deterministic: bool = True,
    ):
        # forward eval so obs is only one timestep
        # encoded = self.encoder(shape_embeddings)
        # pos_embed=self.pos_emb(jnp.arange(1+memories.shape[-3],-1,-1))[:1+memories.shape[-3]]

        for tf_layer, joint_layer, thruster_layer in zip(self.tf_layers, self.joint_layers, self.thruster_layers):
            # Do attention
            shape_embeddings = tf_layer(shape_embeddings, shape_attention_mask, deterministic=deterministic)

            # Joints
            # T, B, 2J, (2SE + JE)

            @jax.vmap
            @jax.vmap
            def do_index2(to_ind, ind):
                return to_ind[ind]

            joint_shape_embeddings = jnp.concatenate(
                [
                    do_index2(shape_embeddings, joint_indexes[..., 0]),
                    do_index2(shape_embeddings, joint_indexes[..., 1]),
                    joint_embeddings,
                ],
                axis=-1,
            )

            shape_joint_entity_delta = joint_layer(joint_shape_embeddings) * joint_mask[..., None]

            @jax.vmap
            @jax.vmap
            def add2(addee, index, adder):
                return addee.at[index].add(adder)

            # Thrusters
            thruster_shape_embeddings = jnp.concatenate(
                [
                    do_index2(shape_embeddings, thruster_indexes),
                    thruster_embeddings,
                ],
                axis=-1,
            )

            shape_thruster_entity_delta = thruster_layer(thruster_shape_embeddings) * thruster_mask[..., None]

            shape_embeddings = add2(shape_embeddings, joint_indexes[..., 0], shape_joint_entity_delta)
            shape_embeddings = add2(shape_embeddings, thruster_indexes, shape_thruster_entity_delta)

        return shape_embeddings


class ActorCriticTransformer(nn.Module):
    action_dim: Sequence[int]
    actor_width: int
    critic_width: int
    action_mode: str
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    transformer_size: int
    transformer_encoder_size: int
    transformer_depth: int
    actor_depth: int
    critic_depth: int
    num_heads: int
    activation: str
    aggregate_mode: str  # "dummy" or "mean" or "dummy_and_mean"
    full_attention_mask: bool  # if true, only mask out inactives, and have everything attend to everything else
    transformer_ffn: bool
    multilayer_joint_thruster_mixing: bool
    joint_thruster_block_num_layers: int
    joint_thruster_block_hidden_dim_multiplication: int
    add_generator_embedding: bool = False
    generator_embedding_number_of_timesteps: int = 10
    recurrent: bool = True
    dropout_prob: float = 0.0

    @nn.compact
    def __call__(self, hidden, x, deterministic: bool = True):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        og_obs, dones = x
        if self.add_generator_embedding:
            obs = og_obs.obs
        else:
            obs = og_obs

        # obs._ is [T, B, N, L]
        # B - batch size
        # T - time
        # N - number of things
        # L - unembedded entity size
        obs: EntityObservation

        def _single_encoder(features, entity_id, concat=True):
            # assume two entity types
            num_to_remove = 1 if concat else 0
            embedding = activation(
                nn.Dense(
                    self.transformer_encoder_size - num_to_remove,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(features)
            )
            if concat:
                id_1h = jnp.zeros((*embedding.shape[:3], 1)).at[:, :, :, 0].set(entity_id)
                return jnp.concatenate([embedding, id_1h], axis=-1)
            else:
                return embedding

        circle_encodings = _single_encoder(obs.circles, 0)
        polygon_encodings = _single_encoder(obs.polygons, 1)
        joint_encodings = _single_encoder(obs.joints, -1, False)
        thruster_encodings = _single_encoder(obs.thrusters, -1, False)
        # Size of this is something like (T, B, N, K) (time, batch, num_entities, embedding_size)

        # T, B, M, K
        shape_encodings = jnp.concatenate([polygon_encodings, circle_encodings], axis=2)
        # T, B, M
        shape_mask = jnp.concatenate([obs.polygon_mask, obs.circle_mask], axis=2)

        def mask_out_inactives(flat_active_mask, matrix_attention_mask):
            matrix_attention_mask = matrix_attention_mask & (flat_active_mask[:, None]) & (flat_active_mask[None, :])
            return matrix_attention_mask

        joint_indexes = obs.joint_indexes
        thruster_indexes = obs.thruster_indexes

        if self.aggregate_mode == "dummy" or self.aggregate_mode == "dummy_and_mean":
            T, B, _, K = circle_encodings.shape
            dummy = jnp.ones((T, B, 1, K))
            shape_encodings = jnp.concatenate([dummy, shape_encodings], axis=2)
            shape_mask = jnp.concatenate(
                [jnp.ones((T, B, 1), dtype=bool), shape_mask],
                axis=2,
            )
            N = obs.attention_mask.shape[-1]
            overall_mask = (
                jnp.ones((T, B, obs.attention_mask.shape[2], N + 1, N + 1), dtype=bool)
                .at[:, :, :, 1:, 1:]
                .set(obs.attention_mask)
            )
            overall_mask = jax.vmap(jax.vmap(mask_out_inactives))(shape_mask, overall_mask)

            # To account for the dummy entity
            joint_indexes = joint_indexes + 1
            thruster_indexes = thruster_indexes + 1

        else:
            overall_mask = obs.attention_mask

        if self.full_attention_mask:
            overall_mask = jnp.ones(overall_mask.shape, dtype=bool)
            overall_mask = jax.vmap(jax.vmap(mask_out_inactives))(shape_mask, overall_mask)

        # Now do attention on these
        all_entities = Transformer(
            num_layers=self.transformer_depth,
            num_heads=self.num_heads,
            qkv_features=self.transformer_size,
            encoder_size=self.transformer_encoder_size,
            transformer_ffn=self.transformer_ffn,
            multilayer_joint_thruster_mixing=self.multilayer_joint_thruster_mixing,
            joint_thruster_block_num_layers=self.joint_thruster_block_num_layers,
            joint_thruster_block_hidden_dim_multiplication=self.joint_thruster_block_hidden_dim_multiplication,
            gating=True,
            gating_bias=0.0,
            dropout_prob=self.dropout_prob,
        )(
            shape_encodings,
            jnp.repeat(overall_mask, repeats=self.num_heads // overall_mask.shape[2], axis=2),
            joint_encodings,
            obs.joint_mask,
            joint_indexes,
            thruster_encodings,
            obs.thruster_mask,
            thruster_indexes,
            deterministic=deterministic,
        )  # add the extra dimension for the heads

        if self.aggregate_mode == "mean" or self.aggregate_mode == "dummy_and_mean":
            embedding = jnp.mean(all_entities, axis=2, where=shape_mask[..., None])
        else:
            embedding = all_entities[:, :, 0]  # Take the dummy entity as the embedding of the entire scene.

        hidden, pi, value = GeneralActorCriticRNN(
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

        return hidden, pi, value
