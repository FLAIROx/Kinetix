import datetime
import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
from kinetix.environment.ued.ued import make_reset_fn_from_config
from kinetix.environment.env import make_kinetix_env
from kinetix.environment.ued.ued import make_vmapped_filtered_level_sampler
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.environment.wrappers import LogWrapper
from kinetix.util.saving import load_params

import wandb


class EMA(NamedTuple):
    value: float
    count: float  # unnormalized weight accumulator, not a step counter
    beta: float = 0.97

    def update_ema(self, current_value):
        # Unnormalized form: count tracks the denominator so value stays in original units.
        # Equivalent to standard EMA but avoids a separate bias-correction step.
        new_count = self.count * self.beta + 1
        new_val = 1 / new_count * current_value + self.beta * self.count / new_count * self.value
        return EMA(new_val, new_count, self.beta)


def get_logger():

    # ANSI escape codes for colors
    COLOR_CODES = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[1;91m",  # Bold Red
    }
    RESET = "\033[0m"

    class ColorFormatter(logging.Formatter):
        def format(self, record):
            log_color = COLOR_CODES.get(record.levelname, "")
            log_time = datetime.datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"{log_time} | {record.levelname}: {record.getMessage()}"
            return f"{log_color}{formatted_message}{RESET}"

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = ColorFormatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False  # ✅ Prevent messages from being passed to the root logger
    return logger


def get_randomly_sampled_eval_levels(config, env_params, static_env_params, env, num_dr_eval_levels):
    sample_random_level = make_reset_fn_from_config(
        config | {"train_level_mode": "random"},
        env_params,
        static_env_params,
        physics_engine=env.physics_engine,
    )
    sample_random_levels = make_vmapped_filtered_level_sampler(
        sample_random_level, env_params, static_env_params, config, env=env
    )
    key_to_sample_dr_eval_set = jax.random.PRNGKey(100)
    return sample_random_levels(key_to_sample_dr_eval_set, num_dr_eval_levels)


def make_env(config, static_env_params, env_params, reset_fn=None):
    env = LogWrapper(
        make_kinetix_env(
            config["action_type"],
            config["observation_type"],
            reset_fn,
            env_params,
            static_env_params,
            create_dummy_env=config["dummy_env"],
        )
    )
    return env


def get_config_from_checkpoint_path(bc_artifact_path):
    config = wandb.Api().artifact(bc_artifact_path).logged_by().config
    config["observation_type"] = ObservationType.from_string(config["observation_type_str"])
    config["action_type"] = ActionType.from_string(config["action_type_str"])
    return config


@jax.jit
def weight_norm(tree):
    """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
    leaves, _ = jax.tree.flatten(tree)
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def compute_gns_metrics(grads, per_device_batch_size, axis_name="devices"):
    """Compute Gradient Noise Scale (GNS) metrics across devices.
        See https://arxiv.org/abs/1812.06162

    GNS = g_squared / s, where s is the simple gradient norm and g_squared
    measures inter-device gradient variance. A high GNS means the batch size
    is below the noise scale and increasing it would help; a low GNS means
    the batch is already large enough relative to gradient noise.

    Must be called inside a pmap/shmap (uses jax.lax.pmean over axis_name).
    Returns (-1, -1, grad_norm, avg_grads) when running on a single device.

    Args:
        grads: Per-device gradient pytree.
        per_device_batch_size: Number of samples on this device.
        axis_name: pmap axis name used for pmean reduction.

    Returns:
        (g_squared, s, grad_norm, averaged_grads)
    """
    b_big = per_device_batch_size * jax.device_count()
    b_small = per_device_batch_size

    local_grad_sq_norm = sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))

    G_avg_grads, local_grad_sq_norm = jax.lax.pmean((grads, local_grad_sq_norm), axis_name=axis_name)

    global_grad_sq_norm = sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(G_avg_grads))
    if jax.device_count() == 1:
        return -1, -1, jnp.sqrt(global_grad_sq_norm), G_avg_grads

    g_squared = 1 / (b_big - b_small) * (b_big * global_grad_sq_norm - b_small * local_grad_sq_norm)
    s = 1 / (1 / b_small - 1 / b_big) * (local_grad_sq_norm - global_grad_sq_norm)

    return g_squared, s, jnp.sqrt(global_grad_sq_norm), G_avg_grads
