import dataclasses
import math

from flax import struct


@struct.dataclass
class UEDParams:
    max_shape_size: float = 1.0
    goal_body_opposide_side_chance: float = 0.5
    goal_body_size_factor: float = 1.0
    min_rjoints_bias: int = 2

    large_rect_dim_chance: float = 0.3
    large_rect_dim_scale: float = 2.0

    generate_triangles: bool = False
    thruster_power_multiplier: float = 2.0

    thruster_align_com_prob: float = 0.8

    motor_on_chance: float = 0.8
    motor_min_speed: float = 0.4
    motor_max_speed: float = 3.0
    motor_min_power: float = 1.0
    motor_max_power: float = 3.0
    wheel_max_power: float = 1.0

    joint_limit_chance: float = 0.4
    joint_limit_max: float = math.pi
    joint_fixed_chance: float = 0.1

    fixate_chance_min: float = 0.02
    fixate_chance_max: float = 1.0
    fixate_chance_scale: float = 4.0  # Fixation probability scales with size
    fixate_shape_bottom_bias: float = 0.0
    fixate_shape_bottom_bias_special_role: float = 0.6

    circle_max_size_coeff: float = 0.8

    connect_to_fixated_prob_coeff: float = 0.05
    connect_visibility_min: float = 0.05
    connect_no_visibility_bias: float = 10.0

    add_shape_chance: float = 0.35
    add_connected_shape_chance: float = 0.35
    add_no_shape_chance: float = 0.3
    add_thruster_chance: float = 0.3
    add_shape_n_proposals: int = 8

    floor_prob_normal: float = 0.9
    floor_prob_green: float = 0.0
    floor_prob_blue: float = 0.02
    floor_prob_red: float = 0.08

    def __hash__(self):
        field_values = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return hash(tuple(sorted(field_values.items())))

    def __eq__(self, value):
        return super().__eq__(value)
