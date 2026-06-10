# utils/__init__.py
# 工具模块统一入口

from .carla_utils import (
    connect_to_carla, enable_sync_mode, disable_sync_mode,
    spawn_ego_vehicle, spawn_adv_vehicle,
    spawn_npc_vehicles, spawn_pedestrians, spawn_pedestrian_at, walk_to_location,
    set_vehicle_speed, apply_brake,
    get_spawn_points, get_random_spawn_point,
    destroy_actors, cleanup_all,
)

from .sensor_utils import (
    CollisionSensor, RGBCamera, LidarSensor,
    DistanceMonitor, LaneInvasionSensor,
)

from .data_saver import (
    ensure_dir, init_result_dir,
    save_vehicle_log, save_training_log, load_training_log,
    save_metrics, load_metrics, save_model, load_model,
    save_xosc, save_scenario_config, save_experiment_results,
)

from .metrics import (
    compute_ttc, compute_thw, compute_collision_probability,
    danger_level, classify_danger,
    EpisodeStats, aggregate_episodes, MovingAverage,
    composite_danger_score, SCENARIO_CATEGORIES, CATEGORY_LABELS_CN,
)

from .geometry_utils import (
    distance_2d, distance_3d, distance_between_vehicles,
    speed_ms, speed_kmh, relative_speed,
    heading_vector, heading_angle, angle_between, is_facing,
    get_lane_offset, get_current_lane_id, is_same_lane,
    trajectory_deviation, get_vehicle_bbox, bbox_overlap,
    to_local_coordinates, to_world_coordinates,
)
