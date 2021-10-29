'''
CARLA Labels API
'''
from collections import OrderedDict
from object_types import (
    WEATHERS,
    weather_names,
    pedestrian_types,
    vehicle_types,
    static_types,
    vehicle_colors,
    car_types,
    motorcycle_types,
    cyclist_types,
)
from customized_utils import emptyobject


keywords_dict = {
    "num_of_weathers": len(weather_names),
    "num_of_vehicle_colors": len(vehicle_colors),
    "num_of_pedestrian_types": len(pedestrian_types),
    "num_of_vehicle_types": len(vehicle_types),
}

general_labels = [
    "friction",
    "num_of_weathers",
    "num_of_static",
    "num_of_pedestrians",
    "num_of_vehicles",
]

weather_labels = [
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wind_intensity",
    "sun_azimuth_angle",
    "sun_altitude_angle",
    "fog_density",
    "fog_distance",
    "wetness",
    "fog_falloff",
]

# number of waypoints to perturb
waypoints_num_limit = 0

waypoint_labels = [
    "perturbation_x",
    "perturbation_y"
]

static_general_labels = [
    "num_of_static_types",
    "static_x",
    "static_y",
    "static_yaw"
]

pedestrian_general_labels = [
    "num_of_pedestrian_types",
    "pedestrian_x",
    "pedestrian_y",
    "pedestrian_yaw",
    "pedestrian_trigger_distance",
    "pedestrian_speed",
    "pedestrian_dist_to_travel",
]

vehicle_general_labels = [
    "num_of_vehicle_types",
    "vehicle_x",
    "vehicle_y",
    "vehicle_yaw",
    "vehicle_initial_speed",
    "vehicle_trigger_distance",
    "vehicle_targeted_speed",
    "vehicle_waypoint_follower",
    "vehicle_targeted_x",
    "vehicle_targeted_y",
    "vehicle_avoid_collision",
    "vehicle_dist_to_travel",
    "vehicle_targeted_yaw",
    "num_of_vehicle_colors",
]


def setup_bounds_mask_labels_distributions_stage1(use_fine_grained_weather=False):

    parameters_min_bounds = OrderedDict()
    parameters_max_bounds = OrderedDict()
    mask = []
    labels = []

    fixed_hyperparameters = {
        "num_of_weathers": len(WEATHERS),
        "num_of_static_types": len(static_types),
        "num_of_pedestrian_types": len(pedestrian_types),
        "num_of_vehicle_types": len(vehicle_types),
        "num_of_vehicle_colors": len(vehicle_colors),
        "waypoints_num_limit": waypoints_num_limit,
    }

    general_min = [0.5, 0, 0, 0, 0]
    general_max = [0.9, fixed_hyperparameters["num_of_weathers"] - 1, 2, 2, 2]
    general_mask = ["real", "int", "int", "int", "int"]

    if use_fine_grained_weather:
        general_min[1] = -1
        general_max[1] = -1

    # general
    mask.extend(general_mask)
    for j in range(len(general_labels)):
        general_label = general_labels[j]
        k_min = "_".join([general_label, "min"])
        k_max = "_".join([general_label, "max"])
        k = "_".join([general_label])

        labels.append(k)
        parameters_min_bounds[k_min] = general_min[j]
        parameters_max_bounds[k_max] = general_max[j]

    if use_fine_grained_weather:
        weather_min = [0, 0, 0, 0, 0, -90, 0, 0, 0, 0]
        weather_max = [100, 80, 80, 50, 360, 90, 15, 100, 40, 2]
        # [100, 100, 100, 100, 360, 90, 100, 100, inf, 5]
        weather_mask = ["real"] * 10

        mask.extend(weather_mask)
        for j in range(len(weather_labels)):
            weather_label = weather_labels[j]
            k_min = "_".join([weather_label, "min"])
            k_max = "_".join([weather_label, "max"])
            k = "_".join([weather_label])

            labels.append(k)
            parameters_min_bounds[k_min] = weather_min[j]
            parameters_max_bounds[k_max] = weather_max[j]

    return (
        fixed_hyperparameters,
        parameters_min_bounds,
        parameters_max_bounds,
        mask,
        labels,
    )


# Set up default bounds, mask, labels, and distributions for a Problem object
def setup_bounds_mask_labels_distributions_stage2(
    fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels
):

    waypoint_min = [-0.5, 0.5]
    waypoint_max = [0.5, 0.5]
    waypoint_mask = ["real", "real"]

    static_general_min = [0, -20, -20, 0]
    static_general_max = [fixed_hyperparameters["num_of_static_types"] - 1, 20, 20, 360]
    static_mask = ["int"] + ["real"] * 3

    # pedestrian activation threshold: 2->8
    pedestrian_general_min = [0, -20, -20, 0, 10, 0, 0]
    pedestrian_general_max = [
        fixed_hyperparameters["num_of_pedestrian_types"] - 1,
        20,
        20,
        360,
        50,
        4,
        50,
    ]
    pedestrian_mask = ["int"] + ["real"] * 6

    # vehicle activation threshold: 0->10
    vehicle_general_min = [0, -20, -20, 0, 0, 10, 0, 0, -20, -20, 0, 0, 0, 0]
    vehicle_general_max = [
        fixed_hyperparameters["num_of_vehicle_types"] - 1,
        20,
        20,
        360,
        10,
        50,
        10,
        1,
        20,
        20,
        1,
        50,
        360,
        fixed_hyperparameters["num_of_vehicle_colors"] - 1,
    ]
    vehicle_mask = (
        ["int"]
        + ["real"] * 6
        + ["int"]
        + ["real"] * 2
        + ["int"]
        + ["real"] * 2
        + ["int"]
    )

    assert len(waypoint_min) == len(waypoint_max)
    assert len(waypoint_min) == len(waypoint_mask)
    assert len(waypoint_mask) == len(waypoint_labels)

    assert len(static_general_min) == len(static_general_max)
    assert len(static_general_min) == len(static_mask)
    assert len(static_mask) == len(static_general_labels)

    assert len(pedestrian_general_min) == len(pedestrian_general_max)
    assert len(pedestrian_general_min) == len(pedestrian_mask)
    assert len(pedestrian_mask) == len(pedestrian_general_labels)

    assert len(vehicle_general_min) == len(vehicle_general_max)
    assert len(vehicle_general_min) == len(vehicle_mask)
    assert len(vehicle_mask) == len(vehicle_general_labels)

    # ego_car waypoint
    for i in range(fixed_hyperparameters["waypoints_num_limit"]):
        mask.extend(waypoint_mask)

        for j in range(len(waypoint_labels)):
            waypoint_label = waypoint_labels[j]
            k_min = "_".join(["ego_car", waypoint_label, "min", str(i)])
            k_max = "_".join(["ego_car", waypoint_label, "max", str(i)])
            k = "_".join(["ego_car", waypoint_label, str(i)])

            labels.append(k)
            parameters_min_bounds[k_min] = waypoint_min[j]
            parameters_max_bounds[k_max] = waypoint_max[j]

    # static
    for i in range(parameters_max_bounds["num_of_static_max"]):
        mask.extend(static_mask)

        for j in range(len(static_general_labels)):
            static_general_label = static_general_labels[j]
            k_min = "_".join([static_general_label, "min", str(i)])
            k_max = "_".join([static_general_label, "max", str(i)])
            k = "_".join([static_general_label, str(i)])

            labels.append(k)
            parameters_min_bounds[k_min] = static_general_min[j]
            parameters_max_bounds[k_max] = static_general_max[j]

    # pedestrians
    for i in range(parameters_max_bounds["num_of_pedestrians_max"]):
        mask.extend(pedestrian_mask)

        for j in range(len(pedestrian_general_labels)):
            pedestrian_general_label = pedestrian_general_labels[j]
            k_min = "_".join([pedestrian_general_label, "min", str(i)])
            k_max = "_".join([pedestrian_general_label, "max", str(i)])
            k = "_".join([pedestrian_general_label, str(i)])

            labels.append(k)
            parameters_min_bounds[k_min] = pedestrian_general_min[j]
            parameters_max_bounds[k_max] = pedestrian_general_max[j]

    # vehicles
    for i in range(parameters_max_bounds["num_of_vehicles_max"]):
        mask.extend(vehicle_mask)

        for j in range(len(vehicle_general_labels)):
            vehicle_general_label = vehicle_general_labels[j]
            k_min = "_".join([vehicle_general_label, "min", str(i)])
            k_max = "_".join([vehicle_general_label, "max", str(i)])
            k = "_".join([vehicle_general_label, str(i)])

            labels.append(k)
            parameters_min_bounds[k_min] = vehicle_general_min[j]
            parameters_max_bounds[k_max] = vehicle_general_max[j]

        for p in range(fixed_hyperparameters["waypoints_num_limit"]):
            mask.extend(waypoint_mask)

            for q in range(len(waypoint_labels)):
                waypoint_label = waypoint_labels[q]
                k_min = "_".join(["vehicle", str(i), waypoint_label, "min", str(p)])
                k_max = "_".join(["vehicle", str(i), waypoint_label, "max", str(p)])
                k = "_".join(["vehicle", str(i), waypoint_label, str(p)])

                labels.append(k)
                parameters_min_bounds[k_min] = waypoint_min[q]
                parameters_max_bounds[k_max] = waypoint_max[q]

    parameters_distributions = OrderedDict()
    for label in labels:
        if "perturbation" in label:
            parameters_distributions[label] = ("normal", 0, 0.25)
        else:
            parameters_distributions[label] = "uniform"

    n_var = (
        5
        + fixed_hyperparameters["waypoints_num_limit"] * 2
        + parameters_max_bounds["num_of_static_max"] * 4
        + parameters_max_bounds["num_of_pedestrians_max"] * 7
        + parameters_max_bounds["num_of_vehicles_max"]
        * (14 + fixed_hyperparameters["waypoints_num_limit"] * 2)
    )

    return (
        fixed_hyperparameters,
        parameters_min_bounds,
        parameters_max_bounds,
        mask,
        labels,
        parameters_distributions,
        n_var,
    )


def assign_key_value_pairs(search_space_info, fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds):
    for d in [fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds]:
        for k, v in d.items():
            assert not hasattr(search_space_info, k), k+'should not appear twice.'
            setattr(search_space_info, k, v)

def generate_fuzzing_content(customized_config):
    customized_parameters_bounds = customized_config['customized_parameters_bounds']

    customized_parameters_distributions = customized_config['customized_parameters_distributions']

    customized_center_transforms = customized_config['customized_center_transforms']

    customized_constraints = customized_config['customized_constraints']

    fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels = setup_bounds_mask_labels_distributions_stage1()
    customize_parameters(parameters_min_bounds, customized_parameters_bounds)
    customize_parameters(parameters_max_bounds, customized_parameters_bounds)

    fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels, parameters_distributions, n_var = setup_bounds_mask_labels_distributions_stage2(fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels)

    customize_parameters(parameters_min_bounds, customized_parameters_bounds)
    customize_parameters(parameters_max_bounds, customized_parameters_bounds)
    customize_parameters(parameters_distributions, customized_parameters_distributions)

    search_space_info = emptyobject()
    assign_key_value_pairs(search_space_info, fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds)


    fuzzing_content = emptyobject(labels=labels, mask=mask, parameters_min_bounds=parameters_min_bounds, parameters_max_bounds=parameters_max_bounds, parameters_distributions=parameters_distributions, customized_constraints=customized_constraints, customized_center_transforms=customized_center_transforms, n_var=n_var, fixed_hyperparameters=fixed_hyperparameters,
    search_space_info=search_space_info, keywords_dict=keywords_dict)

    return fuzzing_content

# Customize parameters
def customize_parameters(parameters, customized_parameters):
    for k, v in customized_parameters.items():
        if k in parameters:
            parameters[k] = v
        else:
            # print(k, 'is not defined in the parameters.')
            pass
