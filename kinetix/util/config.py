import copy
import datetime
import gzip
import json
import os
from hashlib import md5

import jax
import jax.numpy as jnp
import numpy as np
from numpy import isin
from kinetix.environment.ued.ued_state import UEDParams
from omegaconf import OmegaConf
from pandas import isna
from typing import List, Tuple
import wandb
from kinetix.environment.env_state import EnvParams, StaticEnvParams
from collections import defaultdict

from kinetix.util.saving import load_from_json_file


def get_hash_without_seed(config):
    old_seed = config["seed"]
    config["seed"] = 0
    ans = md5(OmegaConf.to_yaml(config, sort_keys=True).encode()).hexdigest()
    config["seed"] = old_seed
    return ans


def get_date() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def generate_params_from_config(config):
    if config.get("env_size_type", "predefined") == "custom":
        # must load env params from a file
        _, static_env_params, env_params = load_from_json_file(os.path.join("worlds", config["custom_path"]))
        return env_params, static_env_params.replace(
            frame_skip=config["frame_skip"],
        )
    env_params = EnvParams()

    static_env_params = StaticEnvParams().replace(
        num_polygons=config["num_polygons"],
        num_circles=config["num_circles"],
        num_joints=config["num_joints"],
        num_thrusters=config["num_thrusters"],
        frame_skip=config["frame_skip"],
        num_motor_bindings=config["num_motor_bindings"],
        num_thruster_bindings=config["num_thruster_bindings"],
    )

    return env_params, static_env_params


def generate_ued_params_from_config(config) -> UEDParams:
    ans = UEDParams()

    if config["env_size_name"] == "s":
        ans = ans.replace(add_shape_n_proposals=1)  # otherwise we get a very weird XLA bug.
    if "fixate_chance_max" in config:
        print("Changing fixate chance max to", config["fixate_chance_max"])
        ans = ans.replace(fixate_chance_max=config["fixate_chance_max"])
    return ans


def get_eval_level_groups(eval_levels: List[str]) -> List[Tuple[str, str]]:
    def get_groups(s):
        # This is the size group
        group_one = s.split("/")[0]
        group_two = s.split("/")[1].split("_")[0]
        group_two = "".join([i for i in group_two if not i.isdigit()])
        if group_two == "h":
            group_two = "handmade"
        if group_two == "r":
            group_two = "random"
        return f"{group_one}_all", f"{group_one}_{group_two}"

    indices = defaultdict(list)

    for idx, s in enumerate(eval_levels):
        groups = get_groups(s)
        for group in groups:
            indices[group].append(idx)

    indices2 = {}
    for g in indices:
        indices2[g] = np.array(indices[g])

    return indices2


def normalise_config(config, name, editor_config=False):
    old_config = copy.deepcopy(config)
    keys = ["env", "learning", "model", "misc", "eval", "ued", "env_size", "train_levels"]
    for k in keys:
        if k not in config:
            config[k] = {}
        small_d = config[k]
        del config[k]
        for kk, vv in small_d.items():
            assert kk not in config, kk
            config[kk] = vv

    if not editor_config:
        config["eval_env_size_true"] = config["eval_env_size"]
        if config["num_train_envs"] == 2048 and "Pixels" in config["env_name"]:
            config["num_train_envs"] = 512
        if "SFL" in name and config["env_size_name"] in ["m", "l"]:
            config["eval_num_attempts"] = 6  # to avoid a very weird XLA bug.
        config["hash"] = get_hash_without_seed(config)

        config["random_hash"] = np.random.randint(2**31)

        config["log_save_path"] = f"logs/{config['hash']}/{config['seed']}-{get_date()}"
        os.makedirs(config["log_save_path"], exist_ok=True)
        with open(f"{config['log_save_path']}/config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(old_config))
        if config["group"] == "auto":
            config["group"] = f"{name}-" + config["group_auto_prefix"] + config["env_name"].replace("Kinetix-", "")
            config["group"] += "-" + str(config["env_size_name"])

        if config["eval_levels"] == ["auto"] or config["eval_levels"] == "auto":
            config["eval_levels"] = config["train_levels_list"]
            print("Using Auto eval levels:", config["eval_levels"])
        config["num_eval_levels"] = len(config["eval_levels"])

        steps = (
            config["num_steps"]
            * config.get("outer_rollout_steps", 1)
            * config["num_train_envs"]
            * (2 if name == "PAIRED" else 1)
        )
        config["num_updates"] = int(config["total_timesteps"]) // steps

        nsteps = int(config["total_timesteps"] // 1e6)
        letter = "M"
        if nsteps >= 1000:
            nsteps = nsteps // 1000
            letter = "B"
        config["run_name"] = (
            config["env_name"] + f"-{name}-" + str(nsteps) + letter + "-" + str(config["num_train_envs"])
        )

        if config["checkpoint_save_freq"] >= config["num_updates"]:
            config["checkpoint_save_freq"] = config["num_updates"]
    return config


def get_tags(config, name):
    return [name]
    tags = [name]
    if name in ["PLR", "ACCEL", "DR"]:
        if config["use_accel"]:
            tags.append("ACCEL")
        else:
            tags.append("PLR")
    return tags


def init_wandb(config, name) -> wandb.run:
    run = wandb.init(
        config=config,
        project=config["wandb_project"],
        group=config["group"],
        name=config["run_name"],
        entity=config["wandb_entity"],
        mode=config["wandb_mode"],
        tags=get_tags(config, name),
    )
    wandb.define_metric("timing/num_updates")
    wandb.define_metric("timing/num_env_steps")
    wandb.define_metric("*", step_metric="timing/num_env_steps")
    wandb.define_metric("timing/sps", step_metric="timing/num_env_steps")
    return run


def save_data_to_local_file(data_to_save, config):
    if not config.get("save_local_data", False):
        return

    def reverse_in(li, value):
        for i, v in enumerate(li):
            if v in value:
                return True
        return False

    clean_data = {k: v for k, v in data_to_save.items() if not reverse_in(["media/", "images/"], k)}

    def _clean(x):
        if isinstance(x, jnp.ndarray):
            return x.tolist()
        elif isinstance(x, jnp.float32):
            if jnp.isnan(x):
                return -float("inf")
            return round(float(x) * 1000) / 1000
        elif isinstance(x, jnp.int32):
            return int(x)
        return x

    clean_data = jax.tree_map(lambda x: _clean(x), clean_data)
    print("Saving this data:", clean_data)
    with open(f"{config['log_save_path']}/data.jsonl", "a+") as f:
        f.write(json.dumps(clean_data) + "\n")


def compress_log_files_after_run(config):
    fpath = f"{config['log_save_path']}/data.jsonl"
    with open(fpath, "rb") as f_in, gzip.open(fpath + ".gz", "wb") as f_out:
        f_out.writelines(f_in)


def get_video_frequency(config, update_step):
    frac_through_training = update_step / config["num_updates"]
    vid_frequency = (
        config["eval_freq"]
        * config["video_frequency"]
        * jax.lax.select(
            (0.1 <= frac_through_training) & (frac_through_training < 0.3),
            1,
            jax.lax.select(
                (0.3 <= frac_through_training) & (frac_through_training < 0.6),
                2,
                4,
            ),
        )
    )
    return vid_frequency
