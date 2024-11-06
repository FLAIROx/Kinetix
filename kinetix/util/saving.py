import json
import os
import pickle
from typing import Any, Dict, Union

import flax.serialization
import flax.serialization
import flax.serialization
import flax.serialization
import flax.serialization
import flax.serialization
import flax.serialization
import jax
import jax.numpy as jnp
import flax
import wandb
from jax2d.engine import (
    calculate_collision_matrix,
    get_empty_collision_manifolds,
    get_pairwise_interaction_indices,
    recalculate_mass_and_inertia,
)
from jax2d.sim_state import RigidBody, SimState
from kinetix.environment.env_state import EnvState, StaticEnvParams, EnvParams

from flax.traverse_util import flatten_dict, unflatten_dict

from safetensors.flax import save_file, load_file

from kinetix.pcg.pcg import env_state_to_pcg_state
from kinetix.pcg.pcg_state import PCGState
import bz2


def check_if_mass_and_inertia_are_correct(state: SimState, env_params: EnvParams, static_params):
    new = recalculate_mass_and_inertia(state, static_params, state.polygon_densities, state.circle_densities)

    def _check(a, b, shape, name):
        a = jnp.where(shape.active, a, jnp.zeros_like(a))
        b = jnp.where(shape.active, b, jnp.zeros_like(b))

        if not jnp.allclose(a, b):
            idxs = jnp.arange(len(shape.active))[(a != b) & shape.active]
            new_one = a[idxs]
            old_one = b[idxs]
            raise ValueError(
                f"Error: {name} is not the same after loading. Indexes {idxs} are incorrect. New = {new_one} | Before = {old_one}"
            )

    _check(new.polygon.inverse_mass, state.polygon.inverse_mass, state.polygon, "Polygon inverse mass")
    _check(new.circle.inverse_mass, state.circle.inverse_mass, state.circle, "Circle inverse mass")
    _check(new.polygon.inverse_inertia, state.polygon.inverse_inertia, state.polygon, "Polygon inverse inertia")
    _check(new.circle.inverse_inertia, state.circle.inverse_inertia, state.circle, "Circle inverse inertia")
    return True


def save_pickle(filename, state):
    with open(filename, "wb") as f:
        pickle.dump(state, f)


def load_pcg_state_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def expand_env_state(env_state: EnvState, static_env_params: StaticEnvParams, ignore_collision_matrix=False):

    num_rects = len(env_state.polygon.position)
    num_circles = len(env_state.circle.position)
    num_joints = len(env_state.joint.a_index)
    num_thrusters = len(env_state.thruster.object_index)

    def _add_dummy(num_to_add, obj):
        return jax.tree_map(
            lambda current: jnp.concatenate(
                [current, jnp.zeros((num_to_add, *current.shape[1:]), dtype=current.dtype)], axis=0
            ),
            obj,
        )

    does_need_to_change = False
    added_rects = 0

    if (
        num_rects > static_env_params.num_polygons
        or num_circles > static_env_params.num_circles
        or num_joints > static_env_params.num_joints
    ):
        raise Exception(
            f"The current static_env_params is too small to accommodate the loaded env_state (needs num_rects={num_rects}, num_circles={num_circles}, num_joints={num_joints} but current is {static_env_params.num_polygons}, {static_env_params.num_circles}, {static_env_params.num_joints})."
        )

    if num_rects < static_env_params.num_polygons:
        added_rects = static_env_params.num_polygons - num_rects
        does_need_to_change = True
        env_state = env_state.replace(
            polygon=_add_dummy(added_rects, env_state.polygon),
            polygon_shape_roles=_add_dummy(added_rects, env_state.polygon_shape_roles),
            polygon_highlighted=_add_dummy(added_rects, env_state.polygon_highlighted),
            polygon_densities=_add_dummy(added_rects, env_state.polygon_densities),
        )

    if num_circles < static_env_params.num_circles:
        does_need_to_change = True
        n_to_add = static_env_params.num_circles - num_circles
        env_state = env_state.replace(
            circle=_add_dummy(n_to_add, env_state.circle),
            circle_shape_roles=_add_dummy(n_to_add, env_state.circle_shape_roles),
            circle_highlighted=_add_dummy(n_to_add, env_state.circle_highlighted),
            circle_densities=_add_dummy(n_to_add, env_state.circle_densities),
        )

    if num_joints < static_env_params.num_joints:
        does_need_to_change = True
        n_to_add = static_env_params.num_joints - num_joints
        env_state = env_state.replace(
            joint=_add_dummy(n_to_add, env_state.joint),
            motor_bindings=_add_dummy(n_to_add, env_state.motor_bindings),
            motor_auto=_add_dummy(n_to_add, env_state.motor_auto),
        )

    if num_thrusters < static_env_params.num_thrusters:
        does_need_to_change = True
        n_to_add = static_env_params.num_thrusters - num_thrusters
        env_state = env_state.replace(
            thruster=_add_dummy(n_to_add, env_state.thruster),
            thruster_bindings=_add_dummy(n_to_add, env_state.thruster_bindings),
        )

    # This fixes the indices
    def _modify_index(old_indices):
        return jnp.where(old_indices >= num_rects, old_indices + added_rects, old_indices)

    if added_rects > 0:
        env_state = env_state.replace(
            joint=env_state.joint.replace(
                a_index=_modify_index(env_state.joint.a_index),
                b_index=_modify_index(env_state.joint.b_index),
            ),
            thruster=env_state.thruster.replace(
                object_index=_modify_index(env_state.thruster.object_index),
            ),
        )
    # Double check the collision manifolds are fine
    if does_need_to_change or 1:
        # print("Loading but changing the shapes to match the current static params.")
        acc_rr_manifolds, acc_cr_manifolds, acc_cc_manifolds = get_empty_collision_manifolds(static_env_params)
        env_state = env_state.replace(
            collision_matrix=(
                env_state.collision_matrix
                if ignore_collision_matrix
                else calculate_collision_matrix(static_env_params, env_state.joint)
            ),
            acc_rr_manifolds=acc_rr_manifolds,
            acc_cr_manifolds=acc_cr_manifolds,
            acc_cc_manifolds=acc_cc_manifolds,
        )
    return env_state


def expand_pcg_state(pcg_state: PCGState, static_env_params):
    new_pcg_state = pcg_state.replace(
        env_state=expand_env_state(pcg_state.env_state, static_env_params),
        env_state_max=expand_env_state(pcg_state.env_state_max, static_env_params),
        env_state_pcg_mask=expand_env_state(
            pcg_state.env_state_pcg_mask, static_env_params, ignore_collision_matrix=True
        ),
    )
    new_pcg_state = new_pcg_state.replace(
        env_state_pcg_mask=new_pcg_state.env_state_pcg_mask.replace(
            collision_matrix=jnp.zeros_like(new_pcg_state.env_state.collision_matrix, dtype=bool),
        )
    )
    num_shapes = new_pcg_state.env_state.polygon.active.shape[0] + new_pcg_state.env_state.circle.active.shape[0]

    return new_pcg_state.replace(
        tied_together=jnp.zeros((num_shapes, num_shapes), dtype=bool)
        .at[
            : pcg_state.tied_together.shape[0],
            : pcg_state.tied_together.shape[1],
        ]
        .set(pcg_state.tied_together)
    )


def load_world_state_pickle(filename, params=None, static_env_params=None):
    static_params = static_env_params or StaticEnvParams()
    with open(filename, "rb") as f:
        state: SimState = pickle.load(f)
        state = jax.tree.map(lambda x: jnp.nan_to_num(x), state)
        # Check if the mass and inertia are reasonable.
        check_if_mass_and_inertia_are_correct(state, params or EnvParams(), static_params)

    # Now check if the shapes are correct
    return expand_env_state(state, static_params)


def stack_list_of_pytrees(list_of_pytrees):
    v = jax.tree_map(lambda x: jnp.expand_dims(x, 0), list_of_pytrees[0])
    for l in list_of_pytrees[1:]:
        v = jax.tree_map(lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 0)], axis=0), v, l)
    return v

def get_pcg_state_from_json(json_filename) -> PCGState:
    env_state, _, _ = load_from_json_file(json_filename)
    return env_state_to_pcg_state(env_state)

def my_load_file(filename):
    data = bz2.BZ2File(filename, "rb")
    data = pickle.load(data)
    return data


def my_save_file(obj, filename):
    with bz2.BZ2File(filename, "w") as f:
        pickle.dump(obj, f)


def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    my_save_file(params, filename)


def load_params(filename: Union[str, os.PathLike], legacy=False) -> Dict:
    if legacy:
        filename = filename.replace("full_model.pbz2", "model.safetensors")
        filename = filename.replace(".pbz2", ".safetensors")
        return unflatten_dict(load_file(filename), sep=",")
    return my_load_file(filename)


def load_params_from_wandb_artifact_path(checkpoint_name, legacy=False):
    api = wandb.Api()
    name = api.artifact(checkpoint_name).download()
    network_params = load_params(name + "/model.pbz2", legacy=legacy)
    return network_params


def save_params_to_wandb(params, timesteps, config):
    if config["checkpoint_human_numbers"]:
        timesteps = str(round(timesteps / 1e9)) + "B"

    run_name = config["run_name"] + "-" + str(config["random_hash"]) + "-" + str(timesteps)
    save_dir = os.path.join(config["save_path"], run_name)
    os.makedirs(save_dir, exist_ok=True)
    save_params(params, f"{save_dir}/model.pbz2")

    # upload this to wandb as an artifact
    artifact = wandb.Artifact(f"{run_name}-checkpoint", type="checkpoint")
    artifact.add_file(f"{save_dir}/model.pbz2")
    artifact.save()
    print(f"Parameters of model saved in {save_dir}/model.pbz2")


def load_params_wandb_artifact_path_full_model(checkpoint_name):
    api = wandb.Api()
    name = api.artifact(checkpoint_name).download()
    all_dict = load_params(name + "/full_model.pbz2")
    return all_dict["params"]


def load_train_state_from_wandb_artifact_path(train_state, checkpoint_name, load_only_params=False, legacy=False):
    api = wandb.Api()
    name = api.artifact(checkpoint_name).download()
    all_dict = load_params(name + "/full_model.pbz2", legacy=legacy)
    if legacy:
        return train_state.replace(params=all_dict)
    train_state = train_state.replace(params=all_dict["params"])
    if not load_only_params:
        train_state = train_state.replace(
            # step=all_dict["step"],
            opt_state=all_dict["opt_state"]
        )
    return train_state


def save_params_to_wandb(params, timesteps, config):
    return save_dict_to_wandb(params, timesteps, config, "params")


def save_dict_to_wandb(dict, timesteps, config, name):
    timesteps = str(round(timesteps / 1e9)) + "B"
    run_name = config["run_name"] + "-" + str(config["random_hash"]) + "-" + str(timesteps)
    save_dir = os.path.join(config["save_path"], run_name)
    os.makedirs(save_dir, exist_ok=True)
    save_params(dict, f"{save_dir}/{name}.pbz2")

    # upload this to wandb as an artifact
    artifact = wandb.Artifact(f"{run_name}-checkpoint", type="checkpoint")
    artifact.add_file(f"{save_dir}/{name}.pbz2")
    artifact.save()
    print(f"Parameters of model saved in {save_dir}/{name}.pbz2")


def save_model_to_wandb(train_state, timesteps, config, is_final=False):
    dict_to_use = {"step": train_state.step, "params": train_state.params, "opt_state": train_state.opt_state}
    step = int(train_state.step)
    if config["economical_saving"]:
        if step in [2048, 10240, 40960, 81920] or is_final:
            save_dict_to_wandb(dict_to_use, timesteps, config, "full_model")
        else:
            print("Not saving model because step is", step)
    else:
        save_dict_to_wandb(dict_to_use, timesteps, config, "full_model")


def import_env_state_from_json(json_file: dict[str, Any]) -> tuple[EnvState, StaticEnvParams, EnvParams]:
    from kinetix.environment.env import create_empty_env

    def normalise(k, v):
        if k == "screen_dim":
            return v
        if type(v) == dict and "0" in v:
            return jnp.array([normalise(k, v[str(i)]) for i in range(len(v))])
        return v

    env_state = json_file["env_state"]
    env_params = json_file["env_params"]
    static_env_params = json_file["static_env_params"]
    env_params_target = EnvParams()
    static_env_params_target = StaticEnvParams()
    new_env_params = flax.serialization.from_state_dict(
        env_params_target, {k: normalise(k, v) for k, v in env_params.items()}
    )
    norm_static = {k: normalise(k, v) for k, v in static_env_params.items()}
    # norm_static["screen_dim"] = tuple(static_env_params_target.screen_dim)
    norm_static["downscale"] = static_env_params_target.downscale
    # print(
    #     static_env_params_target,
    # )
    new_static_env_params = flax.serialization.from_state_dict(static_env_params_target, norm_static)
    new_static_env_params = new_static_env_params.replace(screen_dim=static_env_params_target.screen_dim)

    env_state_target = create_empty_env(new_static_env_params)

    def astype(x, all):
        return jnp.astype(x, all.dtype)

    def _load_rigidbody(env_state_target, i, is_poly):

        to_load_from: dict[str, Any] = env_state["circle" if not is_poly else "polygon"][i]
        role = to_load_from.pop("role")
        density = to_load_from.pop("density")
        if "highlighted" in to_load_from:
            _ = to_load_from.pop("highlighted")
        new_obj = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.circle if not is_poly else env_state_target.polygon),
            {k: normalise(k, v) for k, v in to_load_from.items()},
        )

        if is_poly:
            env_state_target = env_state_target.replace(
                polygon_shape_roles=env_state_target.polygon_shape_roles.at[i].set(role),
                polygon_densities=env_state_target.polygon_densities.at[i].set(density),
                polygon=jax.tree.map(
                    lambda all, new: all.at[i].set(astype(new, all)), env_state_target.polygon, new_obj
                ),
            )
        else:
            env_state_target = env_state_target.replace(
                circle_shape_roles=env_state_target.circle_shape_roles.at[i].set(role),
                circle_densities=env_state_target.circle_densities.at[i].set(density),
                circle=jax.tree.map(lambda all, new: all.at[i].set(astype(new, all)), env_state_target.circle, new_obj),
            )
        return env_state_target

    # Now load the env state:
    for i in range(new_static_env_params.num_circles):
        env_state_target = _load_rigidbody(env_state_target, i, False)
    for i in range(new_static_env_params.num_polygons):
        env_state_target = _load_rigidbody(env_state_target, i, True)

    for i in range(new_static_env_params.num_joints):
        to_load_from = env_state["joint"][i]
        motor_binding = to_load_from.pop("motor_binding")
        new_obj = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.joint), {k: normalise(k, v) for k, v in to_load_from.items()}
        )
        env_state_target = env_state_target.replace(
            joint=jax.tree.map(lambda all, new: all.at[i].set(astype(new, all)), env_state_target.joint, new_obj),
            motor_bindings=env_state_target.motor_bindings.at[i].set(motor_binding),
        )

    for i in range(new_static_env_params.num_thrusters):
        to_load_from = env_state["thruster"][i]
        thruster_binding = to_load_from.pop("thruster_binding")
        new_obj = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.thruster),
            {k: normalise(k, v) for k, v in to_load_from.items()},
        )

        env_state_target = env_state_target.replace(
            thruster=jax.tree.map(lambda all, new: all.at[i].set(astype(new, all)), env_state_target.thruster, new_obj),
            thruster_bindings=env_state_target.thruster_bindings.at[i].set(thruster_binding),
        )

    env_state_target = env_state_target.replace(
        collision_matrix=flax.serialization.from_state_dict(
            env_state_target.collision_matrix, normalise("collision_matrix", env_state["collision_matrix"])
        )
    )

    for i in range(env_state_target.acc_rr_manifolds.active.shape[0]):
        a = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.acc_rr_manifolds),
            {k: normalise(k, v) for k, v in env_state["acc_rr_manifolds"][i].items()},
        )
        b = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.acc_rr_manifolds),
            {k: normalise(k, v) for k, v in env_state["acc_rr_manifolds"][i + 1].items()},
        )
        env_state_target = env_state_target.replace(
            acc_rr_manifolds=jax.tree.map(
                lambda all, new: all.at[i].set(astype(new, all)), env_state_target.acc_rr_manifolds, a
            ),
        )
        env_state_target.replace(
            acc_rr_manifolds=jax.tree.map(
                lambda all, new: all.at[i + 1].set(astype(new, all)), env_state_target.acc_rr_manifolds, b
            )
        )
    for i in range(env_state_target.acc_cr_manifolds.active.shape[0]):
        a = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.acc_cr_manifolds),
            {k: normalise(k, v) for k, v in env_state["acc_cr_manifolds"][i].items()},
        )
        env_state_target = env_state_target.replace(
            acc_cr_manifolds=jax.tree.map(
                lambda all, new: all.at[i].set(astype(new, all)), env_state_target.acc_cr_manifolds, a
            ),
        )
    for i in range(env_state_target.acc_cc_manifolds.active.shape[0]):
        a = flax.serialization.from_state_dict(
            jax.tree.map(lambda x: x[i], env_state_target.acc_cc_manifolds),
            {k: normalise(k, v) for k, v in env_state["acc_cc_manifolds"][i].items()},
        )
        env_state_target = env_state_target.replace(
            acc_cc_manifolds=jax.tree.map(
                lambda all, new: all.at[i].set(astype(new, all)), env_state_target.acc_cc_manifolds, a
            ),
        )

    env_state_target = env_state_target.replace(
        collision_matrix=calculate_collision_matrix(new_static_env_params, env_state_target.joint)
    )

    return (
        env_state_target,
        new_static_env_params,
        new_env_params.replace(max_timesteps=env_params_target.max_timesteps),
    )


def export_env_state_to_json(
    filename: str, env_state: EnvState, static_env_params: StaticEnvParams, env_params: EnvParams
):
    json_to_save = {
        "polygon": [],
        "circle": [],
        "joint": [],
        "thruster": [],
        "collision_matrix": flax.serialization.to_state_dict(env_state.collision_matrix.tolist()),
        "acc_rr_manifolds": [],
        "acc_cr_manifolds": [],
        "acc_cc_manifolds": [],
        "gravity": flax.serialization.to_state_dict(env_state.gravity.tolist()),
    }

    def _rigidbody_to_json(index: int, is_poly):
        main_arr = env_state.polygon if is_poly else env_state.circle
        c = jax.tree.map(lambda x: x[index].tolist(), main_arr)
        roles = env_state.polygon_shape_roles if is_poly else env_state.circle_shape_roles
        densities = env_state.polygon_densities if is_poly else env_state.circle_densities
        highlighted = env_state.polygon_highlighted if is_poly else env_state.circle_highlighted

        d = flax.serialization.to_state_dict(c)
        d["role"] = roles[index].tolist()
        d["density"] = densities[index].tolist()
        d["highlighted"] = highlighted[index].tolist()
        return d

    def _joint_to_json(i):
        joint = jax.tree.map(lambda x: x[i].tolist(), env_state.joint)
        d = flax.serialization.to_state_dict(joint)
        d["motor_binding"] = env_state.motor_bindings[i].tolist()
        return d

    def _thruster_to_json(i):
        thruster = jax.tree.map(lambda x: x[i].tolist(), env_state.thruster)
        d = flax.serialization.to_state_dict(thruster)
        d["thruster_binding"] = env_state.thruster_bindings[i].tolist()
        return d

    for i in range(static_env_params.num_circles):
        json_to_save["circle"].append(_rigidbody_to_json(i, False))
    for i in range(static_env_params.num_polygons):
        json_to_save["polygon"].append(_rigidbody_to_json(i, True))
    for i in range(static_env_params.num_joints):
        json_to_save["joint"].append(_joint_to_json(i))
    for i in range(static_env_params.num_thrusters):
        json_to_save["thruster"].append(_thruster_to_json(i))

    ncc, ncr, nrr, circle_circle_pairs, circle_rect_pairs, rect_rect_pairs = get_pairwise_interaction_indices(
        static_env_params
    )
    for i in range(nrr):
        a = jax.tree.map(lambda x: x[i, 0].tolist(), env_state.acc_rr_manifolds)
        b = jax.tree.map(lambda x: x[i, 1].tolist(), env_state.acc_rr_manifolds)
        json_to_save["acc_rr_manifolds"].append(flax.serialization.to_state_dict(a))
        json_to_save["acc_rr_manifolds"].append(flax.serialization.to_state_dict(b))
    for i in range(ncr):
        a = jax.tree.map(lambda x: x[i].tolist(), env_state.acc_cr_manifolds)
        json_to_save["acc_cr_manifolds"].append(flax.serialization.to_state_dict(a))

    for i in range(ncc):
        a = jax.tree.map(lambda x: x[i].tolist(), env_state.acc_cc_manifolds)
        json_to_save["acc_cc_manifolds"].append(flax.serialization.to_state_dict(a))

    to_save = {
        "env_state": json_to_save,
        "env_params": flax.serialization.to_state_dict(
            jax.tree.map(lambda x: x.tolist() if type(x) == jnp.ndarray else x, env_params)
        ),
        "static_env_params": flax.serialization.to_state_dict(
            jax.tree.map(lambda x: x.tolist() if type(x) == jnp.ndarray else x, static_env_params)
        ),
    }
    with open(filename, "w+") as f:
        json.dump(to_save, f)

    return to_save


def load_from_json_file(filename):
    with open(filename, "r") as f:
        return import_env_state_from_json(json.load(f))


if __name__ == "__main__":
    pass
