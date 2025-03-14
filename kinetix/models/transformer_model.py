from typing import List, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.attention import MultiHeadDotProductAttention
from flax.linen.initializers import constant, orthogonal

from kinetix.models.actor_critic import GeneralActorCriticRNN
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
    gating: bool = False
    gating_bias: float = 0.0

    def setup(self):
        self.attention1 = MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.qkv_features, out_features=self.out_features
        )

        self.ln1 = nn.LayerNorm()

        self.dense1 = nn.Dense(self.out_features)

        self.dense2 = nn.Dense(self.out_features)

        self.ln2 = nn.LayerNorm()
        if self.gating:
            self.gate1 = Gating(self.out_features, self.gating_bias)
            self.gate2 = Gating(self.out_features, self.gating_bias)

    def __call__(self, queries: jnp.ndarray, mask: jnp.ndarray):
        # After reading the paper, this is what I think we should do:
        # First layernorm, then do attention
        queries_n = self.ln1(queries)
        y = self.attention1(queries_n, mask=mask)
        if self.gating:  # and gate
            y = self.gate1(queries, jax.nn.relu(y))
        else:
            y = queries + y
        # Dense after norming, crucially no relu.
        e = self.dense1(self.ln2(y))
        if self.gating:  # and gate again
            # This may be the wrong way around
            e = self.gate2(y, jax.nn.relu(e))
        else:
            e = y + e

        return e


class Transformer(nn.Module):
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers: int
    gating: bool = False
    gating_bias: float = 0.0

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
            )
            for _ in range(self.num_layers)
        ]

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
    ):
        # forward eval so obs is only one timestep
        # encoded = self.encoder(shape_embeddings)
        # pos_embed=self.pos_emb(jnp.arange(1+memories.shape[-3],-1,-1))[:1+memories.shape[-3]]

        for tf_layer, joint_layer, thruster_layer in zip(self.tf_layers, self.joint_layers, self.thruster_layers):
            # Do attention
            shape_embeddings = tf_layer(shape_embeddings, shape_attention_mask)

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
    fc_layer_width: int
    action_mode: str
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    transformer_size: int
    transformer_encoder_size: int
    transformer_depth: int
    fc_layer_depth: int
    num_heads: int
    activation: str
    aggregate_mode: str  # "dummy" or "mean" or "dummy_and_mean"
    full_attention_mask: bool  # if true, only mask out inactives, and have everything attend to everything else
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
        embedding = Transformer(
            num_layers=self.transformer_depth,
            num_heads=self.num_heads,
            qkv_features=self.transformer_size,
            encoder_size=self.transformer_encoder_size,
            gating=True,
            gating_bias=0.0,
        )(
            shape_encodings,
            jnp.repeat(overall_mask, repeats=self.num_heads // overall_mask.shape[2], axis=2),
            joint_encodings,
            obs.joint_mask,
            joint_indexes,
            thruster_encodings,
            obs.thruster_mask,
            thruster_indexes,
        )  # add the extra dimension for the heads

        if self.aggregate_mode == "mean" or self.aggregate_mode == "dummy_and_mean":
            embedding = jnp.mean(embedding, axis=2, where=shape_mask[..., None])
        else:
            embedding = embedding[:, :, 0]  # Take the dummy entity as the embedding of the entire scene.

        return GeneralActorCriticRNN(
            action_dim=self.action_dim,
            fc_layer_depth=self.fc_layer_depth,
            fc_layer_width=self.fc_layer_width,
            action_type=self.action_mode,
            hybrid_action_continuous_dim=self.hybrid_action_continuous_dim,
            multi_discrete_number_of_dims_per_distribution=self.multi_discrete_number_of_dims_per_distribution,
            add_generator_embedding=self.add_generator_embedding,
            generator_embedding_number_of_timesteps=self.generator_embedding_number_of_timesteps,
            recurrent=self.recurrent,
        )(hidden, og_obs, embedding, dones, activation)
