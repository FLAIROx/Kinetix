"""Evaluate a pretrained Kinetix agent on the hand-designed levels.

Download a checkpoint first, e.g.:
    hf download mbeukman/Kinetix-Checkpoints --local-dir ./checkpoints

Then run:
    python examples/example_pretrained.py --checkpoint_dir ./checkpoints/sfl-1m-envs --size m
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from kinetix.models import make_network_from_config
from kinetix.models.actor_critic import GeneralActorCriticRNN
from kinetix.util import (
    generate_params_from_config,
    load_evaluation_levels,
    load_pretrained_checkpoint,
    make_eval_fn,
    normalise_config,
)
from kinetix.util.eval_utils import EvalSpec
from kinetix.util.train_utils import make_env

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--size", default="m", choices=["s", "m", "l"], help="Which hand-designed levels to use")
    parser.add_argument("--num_attempts", type=int, default=20)
    args = parser.parse_args()

    params, pretrained_config = load_pretrained_checkpoint(args.checkpoint_dir)
    print(f"Loaded {pretrained_config['name']}: {pretrained_config['description']}")

    with initialize_config_dir(config_dir=os.path.abspath(CONFIG_DIR), version_base=None):
        config = compose("ppo", overrides=[f"env_size={args.size}", "eval=eval_all"])
    config = normalise_config(OmegaConf.to_container(config), "PPO", save_config=False)
    # The model options must match the ones the checkpoint was trained with.
    config |= pretrained_config["model"]

    level_names = [l for l in config["eval_levels"] if l.startswith(f"{args.size}/")]
    levels, static_env_params = load_evaluation_levels(level_names)
    env_params, _ = generate_params_from_config(config)
    config["static_env_params"] = to_state_dict(static_env_params)
    env = make_env(config, static_env_params, env_params)

    network = make_network_from_config(env, env_params, config)
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=optax.identity())  # no training

    # Check that the checkpoint matches the network that the config creates.
    obs, _ = env.reset(jax.random.PRNGKey(0), env_params, jax.tree.map(lambda x: x[0], levels))
    init_x = (jax.tree.map(lambda x: x[None, None], obs), jnp.zeros((1, 1), dtype=bool))
    init_params = network.init(jax.random.PRNGKey(0), GeneralActorCriticRNN.initialize_carry(1), init_x)
    assert jax.tree.structure({"params": init_params["params"]}) == jax.tree.structure(params)

    eval_spec = EvalSpec(
        levels_to_eval_on=levels, number_of_levels=len(level_names), level_names=level_names, plot_videos=False
    )
    eval_fn = make_eval_fn(env, env_params, args.num_attempts)
    metrics = eval_fn(jax.random.PRNGKey(1), train_state, {"hand_designed": eval_spec})["hand_designed"]
    solve_rates = np.asarray(metrics.episode_metrics.episode_solve_rates)

    for name, solve_rate in zip(level_names, solve_rates):
        print(f"{name:40s} {solve_rate:.2f}")
    print(f"Mean solve rate on the {args.size} hand-designed levels: {solve_rates.mean():.3f}")


if __name__ == "__main__":
    main()
