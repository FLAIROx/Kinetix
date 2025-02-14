import chex
import jax
from jax2d.engine import calculate_collision_matrix
from kinetix.environment.env_state import EnvState, StaticEnvParams
import jax.numpy as jnp

from kinetix.pcg.pcg_state import PCGState


def permute_state(rng: chex.PRNGKey, env_state: EnvState, static_env_params: StaticEnvParams) -> EnvState:
    idxs_circles = jnp.arange(static_env_params.num_circles)
    idxs_polygons = jnp.arange(static_env_params.num_polygons)
    idxs_joints = jnp.arange(static_env_params.num_joints)
    idxs_thrusters = jnp.arange(static_env_params.num_thrusters)

    rng, *_rngs = jax.random.split(rng, 5)
    idxs_circles_permuted = jax.random.permutation(_rngs[0], idxs_circles, independent=True)
    idxs_polygons_permuted = idxs_polygons.at[static_env_params.num_static_fixated_polys :].set(
        jax.random.permutation(_rngs[1], idxs_polygons[static_env_params.num_static_fixated_polys :], independent=True)
    )

    idxs_joints_permuted = jax.random.permutation(_rngs[2], idxs_joints, independent=True)
    idxs_thrusters_permuted = jax.random.permutation(_rngs[3], idxs_thrusters, independent=True)

    combined = jnp.concatenate([idxs_polygons_permuted, idxs_circles_permuted + static_env_params.num_polygons])
    # Change the ordering of the shapes, and also remember to change the indices associated with the joints

    inverse_permutation = jnp.argsort(combined)

    env_state = env_state.replace(
        polygon_shape_roles=env_state.polygon_shape_roles[idxs_polygons_permuted],
        circle_shape_roles=env_state.circle_shape_roles[idxs_circles_permuted],
        polygon_highlighted=env_state.polygon_highlighted[idxs_polygons_permuted],
        circle_highlighted=env_state.circle_highlighted[idxs_circles_permuted],
        polygon_densities=env_state.polygon_densities[idxs_polygons_permuted],
        circle_densities=env_state.circle_densities[idxs_circles_permuted],
        polygon=jax.tree.map(lambda x: x[idxs_polygons_permuted], env_state.polygon),
        circle=jax.tree.map(lambda x: x[idxs_circles_permuted], env_state.circle),
        joint=env_state.joint.replace(
            a_index=inverse_permutation[env_state.joint.a_index],
            b_index=inverse_permutation[env_state.joint.b_index],
        ),
        thruster=env_state.thruster.replace(
            object_index=inverse_permutation[env_state.thruster.object_index],
        ),
    )

    # And now permute the thrusters and joints
    env_state = env_state.replace(
        thruster_bindings=env_state.thruster_bindings[idxs_thrusters_permuted],
        motor_bindings=env_state.motor_bindings[idxs_joints_permuted],
        motor_auto=env_state.motor_auto[idxs_joints_permuted],
        joint=jax.tree.map(lambda x: x[idxs_joints_permuted], env_state.joint),
        thruster=jax.tree.map(lambda x: x[idxs_thrusters_permuted], env_state.thruster),
    )
    # and collision matrix
    env_state = env_state.replace(collision_matrix=calculate_collision_matrix(static_env_params, env_state.joint))
    return env_state


def permute_pcg_state(rng: chex.PRNGKey, pcg_state: PCGState, static_env_params: StaticEnvParams):
    return pcg_state.replace(
        env_state=permute_state(rng, pcg_state.env_state, static_env_params),
        env_state_max=permute_state(rng, pcg_state.env_state_max, static_env_params),
        env_state_pcg_mask=jax.tree.map(lambda x: jnp.zeros_like(x, dtype=bool), pcg_state.env_state_pcg_mask),
    )
