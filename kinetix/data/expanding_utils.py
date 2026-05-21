from functools import partial
import jax
import jax.numpy as jnp
from jax2d.engine import CollisionManifold, Joint, RigidBody, Thruster
import numpy as np

from kinetix.data.bc_types import ActionEnvStateMask
from kinetix.environment.env_state import EnvState, StaticEnvParams


def _np_calc_nrr(static_env_params: StaticEnvParams) -> int:
    nrr_all = static_env_params.num_polygons * (static_env_params.num_polygons - 1) // 2
    nrr_sf = static_env_params.num_static_fixated_polys * (static_env_params.num_static_fixated_polys - 1) // 2
    return nrr_all - nrr_sf


def _np_get_empty_collision_manifolds(static_env_params: StaticEnvParams):
    nrr = _np_calc_nrr(static_env_params)
    acc_rr_manifolds = CollisionManifold(
        normal=np.zeros((nrr, 2, 2), dtype=np.float32),
        penetration=np.zeros((nrr, 2), dtype=np.float32),
        collision_point=np.zeros((nrr, 2, 2), dtype=np.float32),
        acc_impulse_normal=np.zeros((nrr, 2), dtype=np.float32),
        acc_impulse_tangent=np.zeros((nrr, 2), dtype=np.float32),
        active=np.zeros((nrr, 2), dtype=bool),
        restitution_velocity_target=np.zeros((nrr, 2), dtype=np.float32),
    )

    ncr = static_env_params.num_polygons * static_env_params.num_circles
    acc_cr_manifolds = CollisionManifold(
        normal=np.zeros((ncr, 2), dtype=np.float32),
        penetration=np.zeros(ncr, dtype=np.float32),
        collision_point=np.zeros((ncr, 2), dtype=np.float32),
        acc_impulse_normal=np.zeros(ncr, dtype=np.float32),
        acc_impulse_tangent=np.zeros(ncr, dtype=np.float32),
        active=np.zeros(ncr, dtype=bool),
        restitution_velocity_target=np.zeros(ncr, dtype=np.float32),
    )

    ncc = (static_env_params.num_circles * (static_env_params.num_circles - 1)) // 2
    acc_cc_manifolds = CollisionManifold(
        normal=np.zeros((ncc, 2), dtype=np.float32),
        penetration=np.zeros(ncc, dtype=np.float32),
        collision_point=np.zeros((ncc, 2), dtype=np.float32),
        acc_impulse_normal=np.zeros(ncc, dtype=np.float32),
        acc_impulse_tangent=np.zeros(ncc, dtype=np.float32),
        active=np.zeros(ncc, dtype=bool),
        restitution_velocity_target=np.zeros(ncc, dtype=np.float32),
    )

    return acc_rr_manifolds, acc_cr_manifolds, acc_cc_manifolds


def _np_calculate_collision_matrix(static_env_params: StaticEnvParams, joints) -> np.ndarray:
    """Numpy equivalent of jax2d's calculate_collision_matrix.

    Replicates the scan over tile(arange(N), N): for each of N rounds, iterate
    over all N joints and propagate the no-collision constraint for active ones.
    """
    matrix_size = static_env_params.num_polygons + static_env_params.num_circles
    collision_matrix = ~np.eye(matrix_size, dtype=bool)

    for _ in range(static_env_params.num_joints):
        for j_idx in range(static_env_params.num_joints):
            if not bool(joints.active[j_idx]):
                continue
            a = int(joints.a_index[j_idx])
            b = int(joints.b_index[j_idx])
            row = collision_matrix[a] & collision_matrix[b]
            col = collision_matrix[:, b] & collision_matrix[:, a]
            joint_collisions = row & col
            new_cm = collision_matrix.copy()
            new_cm[a] = joint_collisions
            new_cm[:, b] = joint_collisions
            new_cm[b] = joint_collisions
            new_cm[:, a] = joint_collisions
            collision_matrix = new_cm

    return collision_matrix


def expand_env_state_numpy(env_state: EnvState, static_env_params: StaticEnvParams) -> EnvState:
    """CPU/numpy equivalent of kinetix.util.saving.expand_env_state.

    Pads all per-shape arrays with zeros to match static_env_params, fixes
    joint/thruster indices after polygon padding, and resets collision
    manifolds and the collision matrix.  Input arrays may be JAX or numpy;
    all outputs are numpy arrays.
    """
    num_rects = len(env_state.polygon.position)
    num_circles = len(env_state.circle.position)
    num_joints = len(env_state.joint.a_index)
    num_thrusters = len(env_state.thruster.object_index)

    if (
        num_rects > static_env_params.num_polygons
        or num_circles > static_env_params.num_circles
        or num_joints > static_env_params.num_joints
    ):
        raise Exception(
            f"The current static_env_params is too small to accommodate the loaded env_state "
            f"(needs num_rects={num_rects}, num_circles={num_circles}, num_joints={num_joints} "
            f"but current is {static_env_params.num_polygons}, {static_env_params.num_circles}, "
            f"{static_env_params.num_joints})."
        )

    def _add_dummy(num_to_add, obj):
        return jax.tree.map(
            lambda cur: np.concatenate(
                [np.asarray(cur), np.zeros((num_to_add, *cur.shape[1:]), dtype=cur.dtype)], axis=0
            ),
            obj,
        )

    added_rects = 0

    if num_rects < static_env_params.num_polygons:
        added_rects = static_env_params.num_polygons - num_rects
        env_state = env_state.replace(
            polygon=_add_dummy(added_rects, env_state.polygon),
            polygon_shape_roles=_add_dummy(added_rects, env_state.polygon_shape_roles),
            polygon_highlighted=_add_dummy(added_rects, env_state.polygon_highlighted),
            polygon_densities=_add_dummy(added_rects, env_state.polygon_densities),
        )

    if num_circles < static_env_params.num_circles:
        n_to_add = static_env_params.num_circles - num_circles
        env_state = env_state.replace(
            circle=_add_dummy(n_to_add, env_state.circle),
            circle_shape_roles=_add_dummy(n_to_add, env_state.circle_shape_roles),
            circle_highlighted=_add_dummy(n_to_add, env_state.circle_highlighted),
            circle_densities=_add_dummy(n_to_add, env_state.circle_densities),
        )

    if num_joints < static_env_params.num_joints:
        n_to_add = static_env_params.num_joints - num_joints
        env_state = env_state.replace(
            joint=_add_dummy(n_to_add, env_state.joint),
            motor_bindings=_add_dummy(n_to_add, env_state.motor_bindings),
            motor_auto=_add_dummy(n_to_add, env_state.motor_auto),
        )

    if num_thrusters < static_env_params.num_thrusters:
        n_to_add = static_env_params.num_thrusters - num_thrusters
        env_state = env_state.replace(
            thruster=_add_dummy(n_to_add, env_state.thruster),
            thruster_bindings=_add_dummy(n_to_add, env_state.thruster_bindings),
        )

    if added_rects > 0:

        def _modify_index(old_indices):
            arr = np.asarray(old_indices)
            return np.where(arr >= num_rects, arr + added_rects, arr)

        env_state = env_state.replace(
            joint=env_state.joint.replace(
                a_index=_modify_index(env_state.joint.a_index),
                b_index=_modify_index(env_state.joint.b_index),
            ),
            thruster=env_state.thruster.replace(
                object_index=_modify_index(env_state.thruster.object_index),
            ),
        )

    acc_rr_manifolds, acc_cr_manifolds, acc_cc_manifolds = _np_get_empty_collision_manifolds(static_env_params)
    env_state = env_state.replace(
        collision_matrix=_np_calculate_collision_matrix(static_env_params, env_state.joint),
        acc_rr_manifolds=acc_rr_manifolds,
        acc_cr_manifolds=acc_cr_manifolds,
        acc_cc_manifolds=acc_cc_manifolds,
    )

    return env_state


# ---------------------------------------------------------------------------
# Batched numpy expansion (for mixed-size multi-file data loading)
# ---------------------------------------------------------------------------


def _np_get_empty_collision_manifolds_batched(static_env_params: StaticEnvParams, batch_size: int):
    nrr = _np_calc_nrr(static_env_params)
    acc_rr = CollisionManifold(
        normal=np.zeros((batch_size, nrr, 2, 2), dtype=np.float32),
        penetration=np.zeros((batch_size, nrr, 2), dtype=np.float32),
        collision_point=np.zeros((batch_size, nrr, 2, 2), dtype=np.float32),
        acc_impulse_normal=np.zeros((batch_size, nrr, 2), dtype=np.float32),
        acc_impulse_tangent=np.zeros((batch_size, nrr, 2), dtype=np.float32),
        active=np.zeros((batch_size, nrr, 2), dtype=bool),
        restitution_velocity_target=np.zeros((batch_size, nrr, 2), dtype=np.float32),
    )
    ncr = static_env_params.num_polygons * static_env_params.num_circles
    acc_cr = CollisionManifold(
        normal=np.zeros((batch_size, ncr, 2), dtype=np.float32),
        penetration=np.zeros((batch_size, ncr), dtype=np.float32),
        collision_point=np.zeros((batch_size, ncr, 2), dtype=np.float32),
        acc_impulse_normal=np.zeros((batch_size, ncr), dtype=np.float32),
        acc_impulse_tangent=np.zeros((batch_size, ncr), dtype=np.float32),
        active=np.zeros((batch_size, ncr), dtype=bool),
        restitution_velocity_target=np.zeros((batch_size, ncr), dtype=np.float32),
    )
    ncc = (static_env_params.num_circles * (static_env_params.num_circles - 1)) // 2
    acc_cc = CollisionManifold(
        normal=np.zeros((batch_size, ncc, 2), dtype=np.float32),
        penetration=np.zeros((batch_size, ncc), dtype=np.float32),
        collision_point=np.zeros((batch_size, ncc, 2), dtype=np.float32),
        acc_impulse_normal=np.zeros((batch_size, ncc), dtype=np.float32),
        acc_impulse_tangent=np.zeros((batch_size, ncc), dtype=np.float32),
        active=np.zeros((batch_size, ncc), dtype=bool),
        restitution_velocity_target=np.zeros((batch_size, ncc), dtype=np.float32),
    )
    return acc_rr, acc_cr, acc_cc


def _np_calculate_collision_matrix_batched(static_env_params: StaticEnvParams, joints) -> np.ndarray:
    """Vectorised batched version of _np_calculate_collision_matrix.

    joints.active / a_index / b_index all have shape (batch_size, num_joints).
    Returns (batch_size, matrix_size, matrix_size) bool array.

    The inner joint loop is vectorised over the batch axis using advanced
    numpy indexing, so there are no Python loops over batch elements.
    """
    batch_size = np.asarray(joints.active).shape[0]
    matrix_size = static_env_params.num_polygons + static_env_params.num_circles

    cm = np.tile(~np.eye(matrix_size, dtype=bool), (batch_size, 1, 1))

    B = np.arange(batch_size)
    M = np.arange(matrix_size)

    for _ in range(static_env_params.num_joints):
        for j_idx in range(static_env_params.num_joints):
            active = np.asarray(joints.active[:, j_idx])  # (batch_size,)
            if not np.any(active):
                continue
            a = np.asarray(joints.a_index[:, j_idx]).astype(int)  # (batch_size,)
            b = np.asarray(joints.b_index[:, j_idx]).astype(int)  # (batch_size,)

            # Read rows/cols from the current cm (copies, so unaffected by writes below).
            # cm[B, a, :][i] == cm[i, a[i], :]
            row_a = cm[B, a, :]  # (batch_size, matrix_size)
            row_b = cm[B, b, :]  # (batch_size, matrix_size)
            # cm[B[:,None], M[None,:], x[:,None]][i,j] == cm[i, j, x[i]]
            col_b = cm[B[:, None], M[None, :], b[:, None]]  # (batch_size, matrix_size)
            col_a = cm[B[:, None], M[None, :], a[:, None]]  # (batch_size, matrix_size)

            jc = (row_a & row_b) & (col_b & col_a)  # (batch_size, matrix_size)

            # Apply only where this joint is active.
            act = active[:, None]
            new_row_a = np.where(act, jc, row_a)
            new_row_b = np.where(act, jc, row_b)

            # Set row a, row b, col a, col b (matrix stays symmetric).
            cm[B, a, :] = new_row_a
            cm[B, b, :] = new_row_b
            cm[B[:, None], M[None, :], a[:, None]] = new_row_a
            cm[B[:, None], M[None, :], b[:, None]] = new_row_b

    return cm


def expand_env_state_numpy_batched(env_state: EnvState, dst_static: StaticEnvParams) -> EnvState:
    """Batched CPU/numpy version of expand_env_state.

    All per-shape arrays have a leading batch dimension:
      polygon.position.shape == (batch_size, num_polygons, 2)
    Source sizes are inferred from array shapes; only dst_static is required.
    """
    num_rects = env_state.polygon.position.shape[1]
    num_circles = env_state.circle.position.shape[1]
    num_joints = env_state.joint.a_index.shape[1]
    num_thrusters = env_state.thruster.object_index.shape[1]
    batch_size = env_state.polygon.position.shape[0]

    if (
        num_rects > dst_static.num_polygons
        or num_circles > dst_static.num_circles
        or num_joints > dst_static.num_joints
    ):
        raise Exception(
            f"dst_static is too small: needs polys={num_rects}, circles={num_circles}, "
            f"joints={num_joints} but got {dst_static.num_polygons}, "
            f"{dst_static.num_circles}, {dst_static.num_joints}."
        )

    def _add_dummy(num_to_add, obj):
        # cur.shape == (batch_size, count, ...) — pad count axis (axis=1)
        return jax.tree.map(
            lambda cur: np.concatenate(
                [np.asarray(cur), np.zeros((cur.shape[0], num_to_add, *cur.shape[2:]), dtype=cur.dtype)],
                axis=1,
            ),
            obj,
        )

    added_rects = 0

    if num_rects < dst_static.num_polygons:
        added_rects = dst_static.num_polygons - num_rects
        env_state = env_state.replace(
            polygon=_add_dummy(added_rects, env_state.polygon),
            polygon_shape_roles=_add_dummy(added_rects, env_state.polygon_shape_roles),
            polygon_highlighted=_add_dummy(added_rects, env_state.polygon_highlighted),
            polygon_densities=_add_dummy(added_rects, env_state.polygon_densities),
        )

    if num_circles < dst_static.num_circles:
        n_to_add = dst_static.num_circles - num_circles
        env_state = env_state.replace(
            circle=_add_dummy(n_to_add, env_state.circle),
            circle_shape_roles=_add_dummy(n_to_add, env_state.circle_shape_roles),
            circle_highlighted=_add_dummy(n_to_add, env_state.circle_highlighted),
            circle_densities=_add_dummy(n_to_add, env_state.circle_densities),
        )

    if num_joints < dst_static.num_joints:
        n_to_add = dst_static.num_joints - num_joints
        env_state = env_state.replace(
            joint=_add_dummy(n_to_add, env_state.joint),
            motor_bindings=_add_dummy(n_to_add, env_state.motor_bindings),
            motor_auto=_add_dummy(n_to_add, env_state.motor_auto),
        )

    if num_thrusters < dst_static.num_thrusters:
        n_to_add = dst_static.num_thrusters - num_thrusters
        env_state = env_state.replace(
            thruster=_add_dummy(n_to_add, env_state.thruster),
            thruster_bindings=_add_dummy(n_to_add, env_state.thruster_bindings),
        )

    if added_rects > 0:

        def _modify_index(old_indices):
            arr = np.asarray(old_indices)
            return np.where(arr >= num_rects, arr + added_rects, arr)

        env_state = env_state.replace(
            joint=env_state.joint.replace(
                a_index=_modify_index(env_state.joint.a_index),
                b_index=_modify_index(env_state.joint.b_index),
            ),
            thruster=env_state.thruster.replace(
                object_index=_modify_index(env_state.thruster.object_index),
            ),
        )

    acc_rr, acc_cr, acc_cc = _np_get_empty_collision_manifolds_batched(dst_static, batch_size)
    env_state = env_state.replace(
        collision_matrix=_np_calculate_collision_matrix_batched(dst_static, env_state.joint),
        acc_rr_manifolds=acc_rr,
        acc_cr_manifolds=acc_cr,
        acc_cc_manifolds=acc_cc,
    )

    return env_state
