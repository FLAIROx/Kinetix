from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.models.actor_critic import (
    ActorCriticPermutationInvariantSymbolicRNN,
    ActorCriticPixelsRNN,
    ActorCriticSymbolicRNN,
)
from kinetix.models.transformer_model import ActorCriticTransformer


def make_network_from_config(env, env_params, config, network_kws={}):

    action_mode = config["action_type"]
    action_dim = (
        env.action_space(env_params).shape[0]
        if action_mode == ActionType.CONTINUOUS
        else env.action_space(env_params).n
    )
    if "hybrid_action_continuous_dim" not in network_kws:
        network_kws["hybrid_action_continuous_dim"] = action_dim

    if "multi_discrete_number_of_dims_per_distribution" not in network_kws:
        num_joint_bindings = config["static_env_params"]["num_motor_bindings"]
        num_thruster_bindings = config["static_env_params"]["num_thruster_bindings"]
        network_kws["multi_discrete_number_of_dims_per_distribution"] = [3 for _ in range(num_joint_bindings)] + [
            2 for _ in range(num_thruster_bindings)
        ]
    network_kws["recurrent"] = config.get("recurrent_model", True)

    obs_type = config["observation_type"]
    assert obs_type in ObservationType and type(obs_type) != str
    if obs_type == ObservationType.PIXELS:
        cls_to_use = ActorCriticPixelsRNN
        network_kws["cnn_mode"] = config["cnn_mode"]
    elif obs_type == ObservationType.SYMBOLIC_FLAT:
        cls_to_use = ActorCriticSymbolicRNN

    if obs_type == ObservationType.SYMBOLIC_FLAT_PADDED or (
        obs_type == ObservationType.SYMBOLIC_ENTITY and config["permutation_invariant_mlp"]
    ):
        network = ActorCriticPermutationInvariantSymbolicRNN(
            action_dim=action_dim,
            actor_width=config["actor_width"],
            critic_width=config["critic_width"],
            actor_depth=config["actor_depth"],
            critic_depth=config["critic_depth"],
            action_mode=action_mode,
            activation=config["activation"],
            permutation_invariant=True,
            num_heads=config["num_heads"],
            symbolic_embedding_dim=config["symbolic_embedding_dim"],
            preprocess_separately=config["permutation_invariant_mlp"],
            encoder_size=config["encoder_size"],
            separate_actor_critic=config["separate_actor_critic"],
            **network_kws,
        )
    elif obs_type == ObservationType.SYMBOLIC_ENTITY:
        kwargs = dict(
            action_dim=action_dim,
            actor_width=config["actor_width"],
            critic_width=config["critic_width"],
            actor_depth=config["actor_depth"],
            critic_depth=config["critic_depth"],
            action_mode=action_mode,
            num_heads=config["num_heads"],
            transformer_depth=config["transformer_depth"],
            transformer_size=config["transformer_size"],
            transformer_encoder_size=config["transformer_encoder_size"],
            aggregate_mode=config["aggregate_mode"],
            full_attention_mask=config["full_attention_mask"],
            activation=config["activation"],
            transformer_ffn=config["transformer_ffn"],
            multilayer_joint_thruster_mixing=config["multilayer_joint_thruster_mixing"],
            **network_kws,
        )
        thruster_joint_kwargs = dict(
            joint_thruster_block_num_layers=config["joint_thruster_block_num_layers"],
            joint_thruster_block_hidden_dim_multiplication=config["joint_thruster_block_hidden_dim_multiplication"],
        )
        network = ActorCriticTransformer(
            **kwargs,
            **thruster_joint_kwargs,
            dropout_prob=config["dropout_prob"],
        )
    else:
        network = cls_to_use(
            action_dim,
            actor_width=config["actor_width"],
            critic_width=config["critic_width"],
            actor_depth=config["actor_depth"],
            critic_depth=config["critic_depth"],
            activation=config["activation"],
            action_mode=action_mode,
            **network_kws,
        )

    return network
