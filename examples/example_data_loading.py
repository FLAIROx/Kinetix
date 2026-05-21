"""
Minimal example showing how to use TrajectoryDatasetManager.

Usage:
    python experiments/data_loading_example.py --dataset_dir /path/to/traj_data

    # Save a GIF of the first trajectory:
    python experiments/data_loading_example.py --dataset_dir /path/to/traj_data --gif_out out.gif
"""

import argparse
from pathlib import Path

import jax
import numpy as np
import imageio

from kinetix.data import TrajectoryDatasetManager
from kinetix.environment import EnvParams, StaticEnvParams
from kinetix.render import make_render_pixels


def static_env_params_from_batch(env_state, downscale: int = 1) -> StaticEnvParams:
    """Derive StaticEnvParams from the array shapes in a loaded batch's env_state.

    Structural counts (polygons, circles, joints, thrusters, vertices) are read
    from array dimensions. Params that have no shape encoding (num_static_fixated_polys,
    num_motor_bindings, num_thruster_bindings) keep their defaults.
    """
    return StaticEnvParams(
        num_polygons=env_state.polygon.position.shape[-2],
        num_circles=env_state.circle.position.shape[-2],
        num_joints=env_state.joint.active.shape[-1],
        num_thrusters=env_state.thruster.active.shape[-1],
        max_polygon_vertices=env_state.polygon.vertices.shape[-2],
        downscale=downscale,
    )


def render_and_save_gifs(batch, gif_out: str, n_gifs: int) -> None:
    n = min(n_gifs, batch.action.shape[0])
    static_env_params = static_env_params_from_batch(batch.env_state, downscale=2)
    pixel_renderer = jax.jit(make_render_pixels(EnvParams(), static_env_params))
    base = Path(gif_out)
    for i in range(n):
        path = str(base) if n == 1 else str(base.with_stem(f"{base.stem}_{i:03d}"))
        save_gif(render_trajectory(batch, i, pixel_renderer), path)
        print(f"GIF saved: {path}")


def render_trajectory(batch, traj_idx: int, pixel_renderer) -> np.ndarray:
    traj_state = jax.tree.map(lambda x: x[traj_idx], batch.env_state)
    frames_f32 = jax.vmap(pixel_renderer)(traj_state)
    frames = np.clip(np.array(frames_f32), 0, 255).astype(np.uint8)
    return frames.transpose(0, 2, 1, 3)[:, ::-1]  # (T, H, W, C), upright


def save_gif(frames: np.ndarray, path: str, fps: int = 10) -> None:
    imageio.mimsave(path, frames, fps=fps, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maximum_number_of_shards", type=int, default=-1)
    parser.add_argument("--n_val_shards", type=int, default=1)
    parser.add_argument("--dataset_proportions", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--gif_out",
        default=None,
        help="Path for saved GIF(s). With --n_gifs > 1, index is inserted before the extension.",
    )
    parser.add_argument("--n_gifs", type=int, default=1, help="Number of trajectories to render as GIFs.")
    args = parser.parse_args()
    config = vars(args)

    _dataset_common = dict(
        dataset_dir=config["dataset_dir"],
        seed=config["seed"],
        maximum_number_of_shards=config["maximum_number_of_shards"],
        n_val_shards=config["n_val_shards"],
    )

    # TrajectoryDatasetManager: batch_size is in trajectories, not timesteps.
    # Each batch has shape (batch_size, T, *dims).
    dataset_manager = TrajectoryDatasetManager(
        batch_size=config["batch_size"],
        val_batch_size=config["batch_size"],
        **_dataset_common,
    )

    print(f"Dataset length (estimated minibatches): {dataset_manager.length}")

    # ── Load a training batch ─────────────────────────────────────────────────
    batch = dataset_manager.load_next_batch()
    print(f"Training batch — action shape:  {batch.action.shape}")
    print(f"Training batch — mask  shape:   {batch.mask.shape}")

    # ── Optional GIF rendering ────────────────────────────────────────────────
    if config["gif_out"]:
        render_and_save_gifs(batch, config["gif_out"], config["n_gifs"])

    # ── Inspect the validation batch ─────────────────────────────────────────
    if dataset_manager.validation_batch is not None:
        val = dataset_manager.validation_batch
        print(f"Validation batch — action shape: {val.action.shape}")
        print(f"Validation batch — mask  shape:  {val.mask.shape}")
    else:
        print("No validation batch (set n_val_shards >= 1 to enable)")

    # ── Iterate a few batches ─────────────────────────────────────────────────
    for i in range(3):
        b = dataset_manager.load_next_batch()
        n_transitions = b.action.shape[0] * b.action.shape[1]
        print(f"Batch {i + 1}: {n_transitions} transitions, mask fraction = {b.mask.mean():.3f}")


if __name__ == "__main__":
    main()
