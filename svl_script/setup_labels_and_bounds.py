'''
SVL Labels API
'''
from collections import OrderedDict
from .object_types import (
    static_types,
    pedestrian_types,
    vehicle_types,
)
from customized_utils import emptyobject


# class emptyobject():
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)

keywords_dict = {
    "num_of_static_types": len(static_types),
    "num_of_pedestrian_types": len(pedestrian_types),
    "num_of_vehicle_types": len(vehicle_types),
}



general_fields = [
    ("num_of_static", "int", 0, 0),
    ("num_of_pedestrians", "int", 1, 1),
    ("num_of_vehicles", "int", 1, 1),
]

road_fields = [
    ("damage", "real", 0, 1),
]

weather_fields = [
    ("rain", "real", 0, 1),
    ("fog", "real", 0, 1),
    ("wetness", "real", 0, 1),
    ("cloudiness", "real", 0, 1),
]

time_fields = [
    ("hour", "real", 0, 24),
]

# number of waypoints to perturb
waypoints_num_limit = 2

waypoint_fields = [
    ("idle", "real", 0, 20),
    ("trigger_distance", "real", 0, 20),
    ("waypoint_x", "real", -20, 20),
    ("waypoint_y", "real", -20, 20),
]

static_general_fields = [
    ("num_of_static_types", "int", 0, len(static_types)-1),
    ("static_x", "real", -20, 20),
    ("static_y", "real", -20, 20),
]

pedestrian_general_fields = [
    ("num_of_pedestrian_types", "int", 0, len(pedestrian_types)-1),
    ("pedestrian_x", "real", -20, 20),
    ("pedestrian_y", "real", -20, 20),
    ("pedestrian_speed", "real", 1, 5),
]

vehicle_general_fields = [
    ("num_of_vehicle_types", "int", 0, len(vehicle_types)-1),
    ("vehicle_x", "real", -20, 20),
    ("vehicle_y", "real", -20, 20),
    ("vehicle_speed", "real", 1, 10),
]



def setup_bounds_mask_labels_distributions_stage1(use_fine_grained_weather=False):

    fixed_hyperparameters = {
        "num_of_static_types": len(static_types),
        "num_of_pedestrian_types": len(pedestrian_types),
        "num_of_vehicle_types": len(vehicle_types),
        "waypoints_num_limit": waypoints_num_limit,
    }
    parameters_min_bounds = OrderedDict()
    parameters_max_bounds = OrderedDict()
    mask = []
    labels = []
    fields = general_fields + road_fields + weather_fields + time_fields

    for label, data_type, low, high in fields:
        labels.append(label)
        mask.append(data_type)
        parameters_min_bounds

        k_min = "_".join([label, "min"])
        k_max = "_".join([label, "max"])
        k = "_".join([label])

        parameters_min_bounds[k_min] = low
        parameters_max_bounds[k_max] = high


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

    # static
    for i in range(parameters_max_bounds["num_of_static_max"]):
        for label, data_type, low, high in static_general_fields:
            k_min = "_".join([label, "min", str(i)])
            k_max = "_".join([label, "max", str(i)])
            k = "_".join([label, str(i)])

            labels.append(k)
            mask.append(data_type)
            parameters_min_bounds[k_min] = low
            parameters_max_bounds[k_max] = high


    # pedestrians
    for i in range(parameters_max_bounds["num_of_pedestrians_max"]):
        for label, data_type, low, high in pedestrian_general_fields:
            k_min = "_".join([label, "min", str(i)])
            k_max = "_".join([label, "max", str(i)])
            k = "_".join([label, str(i)])

            labels.append(k)
            mask.append(data_type)
            parameters_min_bounds[k_min] = low
            parameters_max_bounds[k_max] = high

        for p in range(fixed_hyperparameters["waypoints_num_limit"]):
            for label, data_type, low, high in waypoint_fields:
                k_min = "_".join(["pedestrian", str(i), label, "min", str(p)])
                k_max = "_".join(["pedestrian", str(i), label, "max", str(p)])
                k = "_".join(["pedestrian", str(i), label, str(p)])

                labels.append(k)
                mask.append(data_type)
                parameters_min_bounds[k_min] = low
                parameters_max_bounds[k_max] = high


    # vehicles
    for i in range(parameters_max_bounds["num_of_vehicles_max"]):
        for label, data_type, low, high in vehicle_general_fields:
            k_min = "_".join([label, "min", str(i)])
            k_max = "_".join([label, "max", str(i)])
            k = "_".join([label, str(i)])

            labels.append(k)
            mask.append(data_type)
            parameters_min_bounds[k_min] = low
            parameters_max_bounds[k_max] = high

        for p in range(fixed_hyperparameters["waypoints_num_limit"]):
            for label, data_type, low, high in waypoint_fields:
                k_min = "_".join(["vehicle", str(i), label, "min", str(p)])
                k_max = "_".join(["vehicle", str(i), label, "max", str(p)])
                k = "_".join(["vehicle", str(i), label, str(p)])

                labels.append(k)
                mask.append(data_type)
                parameters_min_bounds[k_min] = low
                parameters_max_bounds[k_max] = high




    parameters_distributions = OrderedDict()
    for label in labels:
        parameters_distributions[label] = "uniform"
    # 9 + 1 * 3 + 2 * (4+3) + 2 * (4+3)
    n_var = (
        len(general_fields)+len(road_fields)+len(weather_fields)+len(time_fields)+
        + parameters_max_bounds["num_of_static_max"] * len(static_general_fields)
        + parameters_max_bounds["num_of_pedestrians_max"] * (len(pedestrian_general_fields) + fixed_hyperparameters["waypoints_num_limit"] * len(waypoint_fields))
        + parameters_max_bounds["num_of_vehicles_max"] * (len(vehicle_general_fields) + fixed_hyperparameters["waypoints_num_limit"] * len(waypoint_fields))

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

# Customize parameters
def customize_parameters(parameters, customized_parameters):
    for k, v in customized_parameters.items():
        if k in parameters:
            parameters[k] = v
        else:
            # print(k, 'is not defined in the parameters.')
            continue

def sanity_check(parameters, customized_parameters):
    for k, v in customized_parameters.items():
        if k not in parameters:
            print(k, 'is not defined in the parameters.')
            print('parameters', parameters)
            raise


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

    sanity_check([k for k in parameters_min_bounds]+[k for k in parameters_max_bounds]+[k for k in parameters_distributions], customized_parameters_bounds)

    search_space_info = emptyobject()
    assign_key_value_pairs(search_space_info, fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds)


    fuzzing_content = emptyobject(labels=labels, mask=mask, parameters_min_bounds=parameters_min_bounds, parameters_max_bounds=parameters_max_bounds, parameters_distributions=parameters_distributions, customized_constraints=customized_constraints, customized_center_transforms=customized_center_transforms, n_var=n_var, fixed_hyperparameters=fixed_hyperparameters,
    search_space_info=search_space_info, keywords_dict=keywords_dict)

    return fuzzing_content
