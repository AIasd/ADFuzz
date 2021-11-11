customized_bounds_and_distributions = {
    "default": {
        "customized_parameters_bounds": {},
        "customized_parameters_distributions": {},
        "customized_center_transforms": {},
        "customized_constraints": [],
    },

    "turn_left_one_ped_and_one_vehicle": {
        "customized_parameters_bounds": {

            "hour_min": 10,
            "hour_max": 16,

            "pedestrian_x_min_0": -3,
            "pedestrian_x_max_0": 3,
            "pedestrian_y_min_0": -3,
            "pedestrian_y_max_0": 3,
            "pedestrian_speed_min_0": 1,
            "pedestrian_speed_max_0": 5,
            "pedestrian_0_trigger_distance_min_0": 20,
            "pedestrian_0_trigger_distance_max_0": 100,
            "pedestrian_0_waypoint_x_min_0": 0,
            "pedestrian_0_waypoint_x_max_0": 0,
            "pedestrian_0_waypoint_y_min_0": 20,
            "pedestrian_0_waypoint_y_max_0": 50,

            "vehicle_x_min_0": -2,
            "vehicle_x_max_0": 2,
            "vehicle_y_min_0": -5,
            "vehicle_y_max_0": 5,
            "vehicle_speed_min_0": 2,
            "vehicle_speed_max_0": 8,
            "vehicle_0_trigger_distance_min_0": 20,
            "vehicle_0_trigger_distance_max_0": 100,
            "vehicle_0_waypoint_x_min_0": 0,
            "vehicle_0_waypoint_x_max_0": 0,
            "vehicle_0_waypoint_y_min_0": 30,
            "vehicle_0_waypoint_y_max_0": 60,

        },
        "customized_parameters_distributions": {},
        "customized_center_transforms": {
            "vehicle_center_transform_0": ("absolute_location", 19.4,-2.4,-24.3,0,287,0),
            "pedestrian_center_transform_0": ("absolute_location", -11.4,-2,0.4,0,108,0),
        },
        "customized_constraints": [],
    }
}
customized_routes = {
    "BorregasAve_forward": {
        "town_name": "BorregasAve",
        "location_list": [(-50.3399963378906, -1.03600025177002, -9.03000640869141, 0, 104.823371887207, 0), (24.970, -2.615, -29.956, 0.731, 104.547, 358.660)],
    },
    "BorregasAve_left": {
        "town_name": "BorregasAve",
        "location_list": [(-50.3399963378906, -1.03600025177002, -9.03000640869141, 0, 104.823371887207, 0), (7.53597784042358, -2.04524087905884, 5.36189842224121, 0.126350626349449, 14.6359872817993, 358.324432373047)],
    },
    "SanFrancisco_right": {
        "town_name": "SanFrancisco",
        "location_list": [(-201.879058837891, 10.2880001068115, 217.720001220703,0,180,0), (-201.59309387207, 10.13, 151.04377746582,0,179.230941772461,0)],
    },
}
