import os
import math
import pathlib
import traceback
import json
import pickle
import re
from distutils.dir_util import copy_tree
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime

from leaderboard.fuzzing import LeaderboardEvaluator
from leaderboard.utils.statistics_manager import StatisticsManager
from customized_utils import arguments_info, make_hierarchical_dir, emptyobject, is_distinct_vectorized


from carla_specific_utils.scene_configs import customized_bounds_and_distributions, customized_routes
from carla_specific_utils.object_params import Static, Pedestrian, Vehicle


from carla_specific_utils.setup_labels_and_bounds import generate_fuzzing_content, static_general_labels, pedestrian_general_labels, vehicle_general_labels, waypoint_labels, waypoints_num_limit

from carla_specific_utils.carla_specific_tools import perturb_route, add_transform, create_transform, copy_transform, estimate_objectives

from carla_specific_utils.object_types import static_types, pedestrian_types, vehicle_types, vehicle_colors, car_types, motorcycle_types, cyclist_types

import carla

def parse_route_file(route_filename, route_length_lower_bound=50):
    def l2_dist(x, y, prev_x, prev_y):
        return np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

    config_list = []
    tree = ET.parse(route_filename)

    for route in tree.iter("route"):
        route_id = int(route.attrib["id"])
        town_name = route.attrib["town"]

        transform_list = []
        first_waypoint = True
        d = 0
        for waypoint in route.iter("waypoint"):
            x, y, z = (
                float(waypoint.attrib["x"]),
                float(waypoint.attrib["y"]),
                float(waypoint.attrib["z"]),
            )
            pitch, yaw, roll = (
                float(waypoint.attrib["pitch"]),
                float(waypoint.attrib["yaw"]),
                float(waypoint.attrib["roll"]),
            )

            if first_waypoint:
                first_waypoint = False
            else:
                d += l2_dist(x, y, prev_x, prev_y)

            transform_list.append((x, y, z, pitch, yaw, roll))
            if d > route_length_lower_bound:
                first_waypoint = True
                d = 0

                config_list.append([route_id, town_name, transform_list])
                transform_list = []

            prev_x, prev_y = x, y

    return config_list



def correct_spawn_locations_all(x_data, labels):
    with open('tmp_folder/total.pickle', 'rb') as f_in:
        all_final_generated_transforms_list = pickle.load(f_in)

    label_to_id = {label:i for i, label in enumerate(labels)}
    print('-'*10, 'correct_spawn_locations', '-'*10)
    for i, all_final_generated_transforms_list_i in enumerate(all_final_generated_transforms_list):
        if all_final_generated_transforms_list_i:
            correct_spawn_locations(x_data, label_to_id, all_final_generated_transforms_list_i, 'static', static_general_labels)
            correct_spawn_locations(x_data, label_to_id, all_final_generated_transforms_list_i, 'pedestrian', pedestrian_general_labels)
            correct_spawn_locations(x_data, label_to_id, all_final_generated_transforms_list_i, 'vehicle', vehicle_general_labels)
            print('\n'*3)

def correct_spawn_locations(x_data, label_to_id, all_final_generated_transforms_list_i, object_type, keys):
    object_type_plural = object_type
    if object_type in ['pedestrian', 'vehicle']:
        object_type_plural += 's'

    num_of_objects_ind = label_to_id['num_of_'+object_type_plural]
    x_data[num_of_objects_ind] = 0

    empty_slots = deque()
    for j, (x, y, yaw) in enumerate(all_final_generated_transforms_list_i[object_type]):
        if x == None:
            empty_slots.append(j)
        else:
            x_data[num_of_objects_ind] += 1
            if object_type+'_x_'+str(j) not in label_to_id:
                print(label_to_id)
                print(object_type+'_x_'+str(j))
                print(all_final_generated_transforms_list_i[object_type])
                raise
            x_j_ind = label_to_id[object_type+'_x_'+str(j)]
            y_j_ind = label_to_id[object_type+'_y_'+str(j)]
            yaw_j_ind = label_to_id[object_type+'_yaw_'+str(j)]


            # print(object_type, j)
            # print('x', x_data[x_j_ind], '->', x)
            # print('y', x_data[y_j_ind], '->', y)
            # print('yaw', x_data[yaw_j_ind], '->', yaw)
            x_data[x_j_ind] = x
            x_data[y_j_ind] = y
            x_data[yaw_j_ind] = yaw

            if len(empty_slots) > 0:
                q = empty_slots.popleft()
                print('shift', j, 'to', q)
                for k in keys:
                    # print(k)
                    ind_to = label_to_id[k+'_'+str(q)]
                    ind_from = label_to_id[k+'_'+str(j)]
                    x_data[ind_to] = x_data[ind_from]
                if object_type == 'vehicle':
                    for p in range(waypoints_num_limit):
                        for waypoint_label in waypoint_labels:
                            ind_to = label_to_id['_'.join(['vehicle', str(q), waypoint_label, str(p)])]
                            ind_from = label_to_id['_'.join(['vehicle', str(j), waypoint_label, str(p)])]
                            x_data[ind_to] = x_data[ind_from]

                empty_slots.append(j)






def convert_x_to_customized_data(
    x,
    fuzzing_content,
    port
):

    waypoints_num_limit = fuzzing_content.search_space_info.waypoints_num_limit
    num_of_static_max = fuzzing_content.search_space_info.num_of_static_max
    num_of_pedestrians_max = fuzzing_content.search_space_info.num_of_pedestrians_max
    num_of_vehicles_max = fuzzing_content.search_space_info.num_of_vehicles_max

    customized_center_transforms = fuzzing_content.customized_center_transforms
    parameters_min_bounds = fuzzing_content.parameters_min_bounds
    parameters_max_bounds = fuzzing_content.parameters_max_bounds

    # parameters
    # global
    friction = x[0]
    weather_index = int(x[1])
    num_of_static = int(x[2])
    num_of_pedestrians = int(x[3])
    num_of_vehicles = int(x[4])

    ind = 5

    # if use_fine_grained_weather:
    if weather_index == -1:
        fine_grained_weather = carla.WeatherParameters(
            x[ind],
            x[ind + 1],
            x[ind + 2],
            x[ind + 3],
            x[ind + 4],
            x[ind + 5],
            x[ind + 6],
            x[ind + 7],
            x[ind + 8],
            x[ind + 9],
        )
        print(
            "weather params:",
            x[ind],
            x[ind + 1],
            x[ind + 2],
            x[ind + 3],
            x[ind + 4],
            x[ind + 5],
            x[ind + 6],
            x[ind + 7],
            x[ind + 8],
            x[ind + 9],
        )
        ind += 10
    else:
        fine_grained_weather = None

    # ego car
    ego_car_waypoints_perturbation = []
    for _ in range(waypoints_num_limit):
        dx = x[ind]
        dy = x[ind + 1]
        ego_car_waypoints_perturbation.append([dx, dy])
        ind += 2

    # static
    static_list = []
    for i in range(num_of_static_max):
        if i < num_of_static:
            static_type_i = static_types[int(x[ind])]
            static_transform_i = create_transform(
                x[ind + 1], x[ind + 2], 0, 0, x[ind + 3], 0
            )
            static_i = Static(model=static_type_i, spawn_transform=static_transform_i)
            static_list.append(static_i)
        ind += 4

    # pedestrians
    pedestrian_list = []
    for i in range(num_of_pedestrians_max):
        if i < num_of_pedestrians:
            pedestrian_type_i = pedestrian_types[int(x[ind])]
            pedestrian_transform_i = create_transform(
                x[ind + 1], x[ind + 2], 0, 0, x[ind + 3], 0
            )
            pedestrian_i = Pedestrian(
                model=pedestrian_type_i,
                spawn_transform=pedestrian_transform_i,
                trigger_distance=x[ind + 4],
                speed=x[ind + 5],
                dist_to_travel=x[ind + 6],
                after_trigger_behavior="stop",
            )
            pedestrian_list.append(pedestrian_i)
        ind += 7

    # vehicles
    vehicle_list = []
    for i in range(num_of_vehicles_max):
        if i < num_of_vehicles:
            vehicle_type_i = vehicle_types[int(x[ind])]

            vehicle_transform_i = create_transform(
                x[ind + 1], x[ind + 2], 0, 0, x[ind + 3], 0
            )

            vehicle_initial_speed_i = x[ind + 4]
            vehicle_trigger_distance_i = x[ind + 5]

            vehicle_targeted_speed_i = x[ind + 6]
            vehicle_waypoint_follower_i = bool(x[ind + 7])

            vehicle_targeted_waypoint_i = create_transform(
                x[ind + 8], x[ind + 9], 0, 0, 0, 0
            )

            vehicle_avoid_collision_i = bool(x[ind + 10])
            vehicle_dist_to_travel_i = x[ind + 11]
            vehicle_target_yaw_i = x[ind + 12]
            x_dir = np.cos(np.deg2rad(vehicle_target_yaw_i))
            y_dir = np.sin(np.deg2rad(vehicle_target_yaw_i))
            vehicle_target_direction_i = carla.Vector3D(x_dir, y_dir, 0)

            vehicle_color_i = vehicle_colors[int(x[ind + 13])]

            ind += 14

            vehicle_waypoints_perturbation_i = []
            for _ in range(waypoints_num_limit):
                dx = x[ind]
                dy = x[ind + 1]
                vehicle_waypoints_perturbation_i.append([dx, dy])
                ind += 2

            vehicle_i = Vehicle(
                model=vehicle_type_i,
                spawn_transform=vehicle_transform_i,
                avoid_collision=vehicle_avoid_collision_i,
                initial_speed=vehicle_initial_speed_i,
                trigger_distance=vehicle_trigger_distance_i,
                waypoint_follower=vehicle_waypoint_follower_i,
                targeted_waypoint=vehicle_targeted_waypoint_i,
                dist_to_travel=vehicle_dist_to_travel_i,
                target_direction=vehicle_target_direction_i,
                targeted_speed=vehicle_targeted_speed_i,
                after_trigger_behavior="stop",
                color=vehicle_color_i,
                waypoints_perturbation=vehicle_waypoints_perturbation_i,
            )
            # print('\n'*3, 'vehicle', i, vehicle_transform_i, vehicle_avoid_collision_i, vehicle_initial_speed_i, vehicle_trigger_distance_i, vehicle_waypoint_follower_i, vehicle_targeted_waypoint_i, vehicle_dist_to_travel_i, vehicle_target_direction_i, vehicle_targeted_speed_i, '\n'*3)
            vehicle_list.append(vehicle_i)
        else:
            ind += 14 + waypoints_num_limit * 2


    customized_data = {
        "friction": friction,
        "weather_index": weather_index,
        "num_of_static": num_of_static,
        "num_of_pedestrians": num_of_pedestrians,
        "num_of_vehicles": num_of_vehicles,
        "static_list": static_list,
        "pedestrian_list": pedestrian_list,
        "vehicle_list": vehicle_list,
        "using_customized_route_and_scenario": True,
        "ego_car_waypoints_perturbation": ego_car_waypoints_perturbation,
        "add_center": True,
        "port": port,
        "customized_center_transforms": customized_center_transforms,
        # "parameters_min_bounds": parameters_min_bounds,
        # "parameters_max_bounds": parameters_max_bounds,
        "fine_grained_weather": fine_grained_weather,
        "tmp_travel_dist_file": None
        # "tmp_travel_dist_file_" + str(port) + ".txt",
    }

    return customized_data





def norm_2d(loc_1, loc_2):
    return np.sqrt((loc_1.x - loc_2.x) ** 2 + (loc_1.y - loc_2.y) ** 2)

def get_angle(x1, y1, x2, y2):
    angle = np.arctan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)

    return angle

def get_bbox(vehicle):
    current_tra = vehicle.get_transform()
    current_loc = current_tra.location

    heading_vec = current_tra.get_forward_vector()
    heading_vec.z = 0
    heading_vec = heading_vec / math.sqrt(
        math.pow(heading_vec.x, 2) + math.pow(heading_vec.y, 2)
    )
    perpendicular_vec = carla.Vector3D(-heading_vec.y, heading_vec.x, 0)

    extent = vehicle.bounding_box.extent
    x_boundary_vector = heading_vec * extent.x
    y_boundary_vector = perpendicular_vec * extent.y

    bbox = [
        current_loc + carla.Location(x_boundary_vector - y_boundary_vector),
        current_loc + carla.Location(x_boundary_vector + y_boundary_vector),
        current_loc + carla.Location(-1 * x_boundary_vector - y_boundary_vector),
        current_loc + carla.Location(-1 * x_boundary_vector + y_boundary_vector),
    ]

    return bbox


def correct_travel_dist(data, labels, tmp_travel_dist_file):
    from collections import OrderedDict

    if os.path.exists(tmp_travel_dist_file):
        label_to_id = {label: i for i, label in enumerate(labels)}
        # add label and value of resulting variables to x
        id_to_label = {}
        id_to_dist = {}
        with open(tmp_travel_dist_file, "r") as f_in:
            for line in f_in:
                tokens = line.strip().split(",")
                if len(tokens) == 3:
                    actor_id, general_actor_type, index = tokens
                    id_to_label[actor_id] = "_".join(
                        [general_actor_type, "dist_to_travel", index]
                    )
                elif len(tokens) == 2:
                    actor_id = tokens[0]
                    dist = float(tokens[1])
                    if actor_id not in id_to_dist or (
                        actor_id in id_to_dist and dist > id_to_dist[actor_id]
                    ):
                        id_to_dist[actor_id] = dist

        for actor_id in id_to_label:
            label = id_to_label[actor_id]
            if actor_id in id_to_dist:
                dist = id_to_dist[actor_id]
            else:
                dist = 0
            entry_i = labels.index(label)
            data[entry_i] = dist
    else:
        pass


def angle_from_center_view_fov(target, ego, fov=90):
    target_location = target.get_location()
    ego_location = ego.get_location()
    ego_orientation = ego.get_transform().rotation.yaw

    # hack: adjust to the front central camera's location
    # this needs to be changed when the camera's location / fov change
    dx = 1.3 * np.cos(np.deg2rad(ego_orientation - 90))

    ego_location = ego.get_location()
    ego_x = ego_location.x + dx
    ego_y = ego_location.y

    target_vector = np.array([target_location.x - ego_x, target_location.y - ego_y])
    norm_target = np.linalg.norm(target_vector)

    if norm_target < 0.001:
        return 0

    forward_vector = np.array(
        [
            math.cos(math.radians(ego_orientation)),
            math.sin(math.radians(ego_orientation)),
        ]
    )

    try:
        d_angle = np.abs(
            math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))
        )
    except:
        print(
            "\n" * 3,
            "np.dot(forward_vector, target_vector)",
            np.dot(forward_vector, target_vector),
            norm_target,
            "\n" * 3,
        )
        d_angle = 0
    # d_angle_norm == 0 when target within fov
    d_angle_norm = np.clip((d_angle - fov / 2) / (180 - fov / 2), 0, 1)

    return d_angle_norm






def run_carla_simulation(x, fuzzing_content, fuzzing_arguments, sim_specific_arguments, dt_arguments, launch_server, counter, port):

    customized_data = convert_x_to_customized_data(x, fuzzing_content, port)
    episode_max_time = fuzzing_arguments.episode_max_time
    ego_car_model = fuzzing_arguments.ego_car_model
    record_every_n_step = fuzzing_arguments.record_every_n_step
    debug = fuzzing_arguments.debug
    parent_folder = fuzzing_arguments.parent_folder
    route_type = fuzzing_arguments.route_type
    mean_objectives_across_generations_path = fuzzing_arguments.mean_objectives_across_generations_path
    town_name = sim_specific_arguments.town_name
    scenario = sim_specific_arguments.scenario
    direction = sim_specific_arguments.direction
    route_str = sim_specific_arguments.route_str
    scenario_file = sim_specific_arguments.scenario_file
    call_from_dt = dt_arguments.call_from_dt
    customized_data['terminate_on_collision'] = fuzzing_arguments.terminate_on_collision

    return run_carla_simulation_helper(customized_data,
    launch_server, episode_max_time, call_from_dt,
    town_name, scenario, direction, route_str, route_type, scenario_file, ego_car_model,
    record_every_n_step=record_every_n_step, debug=debug, counter=counter, parent_folder=parent_folder, mean_objectives_across_generations_path=mean_objectives_across_generations_path, fuzzing_arguments=fuzzing_arguments, dt_arguments=dt_arguments, sim_specific_arguments=sim_specific_arguments, fuzzing_content=fuzzing_content, x=x)


def run_carla_simulation_helper(customized_data, launch_server, episode_max_time, call_from_dt, town_name, scenario, direction, route_str, route_type, scenario_file, ego_car_model, ego_car_model_path=None, rerun=False, record_every_n_step=2000, debug=0, counter=0, parent_folder='', mean_objectives_across_generations_path='', fuzzing_arguments=None, dt_arguments=None, sim_specific_arguments=None, fuzzing_content=None, x=None):

    arguments = arguments_info()
    arguments.record_every_n_step = record_every_n_step
    arguments.port = customized_data['port']
    arguments.debug = debug



    if ego_car_model == 'lbc':
        arguments.agent = 'carla_lbc/scenario_runner/team_code/image_agent.py'
        arguments.agent_config = 'carla_lbc/models/epoch=24.ckpt'
        base_save_folder = 'carla_lbc/collected_data_customized'
    elif ego_car_model == 'lbc_augment':
        arguments.agent = 'carla_lbc/scenario_runner/team_code/image_agent.py'

        arguments.agent_config = 'carla_lbc/checkpoints/stage2_pretrained/town05_left_non_bug_train_non_debug/epoch=0.ckpt'

        base_save_folder = 'collected_data_lbc_augment'
    elif ego_car_model == 'auto_pilot':
        arguments.agent = 'carla_lbc/leaderboard/team_code/auto_pilot.py'
        arguments.agent_config = ''
        base_save_folder = 'collected_data_autopilot'
    elif ego_car_model == 'pid_agent':
        arguments.agent = 'carla_lbc/scenario_runner/team_code/pid_agent.py'
        arguments.agent_config = ''
        base_save_folder = 'collected_data_pid_agent'

    else:
        print('unknown ego_car_model:', ego_car_model)

    if ego_car_model_path:
        arguments.agent_config = ego_car_model_path


    if rerun:
        os.environ['SAVE_FOLDER'] = make_hierarchical_dir([base_save_folder, '/rerun', str(int(arguments.port)), str(call_from_dt)])
    else:
        os.environ['SAVE_FOLDER'] = make_hierarchical_dir([base_save_folder, str(int(arguments.port)), str(call_from_dt)])

    arguments.scenarios = scenario_file




    statistics_manager = StatisticsManager()



    # sample_factor is an integer between [1, 8]
    sample_factor = 5
    weather_index = customized_data['weather_index']


    # Laundry Stuff-------------------------------------------------------------
    arguments.weather_index = weather_index
    os.environ['WEATHER_INDEX'] = str(weather_index)



    os.environ['SAVE_FOLDER'] = make_hierarchical_dir([os.environ['SAVE_FOLDER'] + '/' + town_name, scenario, direction])
    arguments.save_folder = os.environ['SAVE_FOLDER']


    arguments.routes = 'carla_lbc/leaderboard/data/customized_routes/' + '/'.join([town_name, scenario, direction]) + '/route_' + route_str + '.xml'
    os.environ['ROUTES'] = arguments.routes

    tmp_save_path = os.path.join(arguments.save_folder, 'route_'+route_str)

    # TBD: for convenience
    arguments.deviations_folder = tmp_save_path


    # extract waypoints along route
    tree = ET.parse(arguments.routes)
    route_waypoints = []



    # this iteration should only go once since we only keep one route per file
    for route in tree.iter("route"):
        route_id = route.attrib['id']
        route_town = route.attrib['town']

        for waypoint in route.iter('waypoint'):
            route_waypoints.append(create_transform(float(waypoint.attrib['x']), float(waypoint.attrib['y']), float(waypoint.attrib['z']), float(waypoint.attrib['pitch']), float(waypoint.attrib['yaw']), float(waypoint.attrib['roll'])))


    # --------------------------------------------------------------------------

    customized_data['using_customized_route_and_scenario'] = True
    customized_data['destination'] = route_waypoints[-1].location
    customized_data['sample_factor'] = sample_factor
    customized_data['number_of_attempts_to_request_actor'] = 10




    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager, launch_server, episode_max_time)
        leaderboard_evaluator.run(arguments, customized_data)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator
        # collect signals for estimating objectives


        objectives, loc, object_type, route_completion = estimate_objectives(tmp_save_path, default_objectives=fuzzing_arguments.default_objectives)

    # hack for correcting spawn locations:
    filename = 'carla_lbc/tmp_folder/'+str(arguments.port)+'.pickle'
    with open(filename, 'rb') as f_in:
        all_final_generated_transforms = pickle.load(f_in)


    run_info = {}
    if parent_folder:
        is_bug = check_bug(objectives)
        if is_bug:
            bug_type, bug_str = classify_bug_type(objectives, object_type)
        else:
            bug_type, bug_str = None, None
        if is_bug:
            with open(mean_objectives_across_generations_path, 'a') as f_out:
                f_out.write(str(counter)+','+bug_str+'\n')

        bug_folder = make_hierarchical_dir([parent_folder, 'bugs'])
        non_bug_folder = make_hierarchical_dir([parent_folder, 'non_bugs'])
        if is_bug:
            cur_folder = make_hierarchical_dir([bug_folder, str(counter)])
        else:
            cur_folder = make_hierarchical_dir([non_bug_folder, str(counter)])


        xl = [pair[1] for pair in fuzzing_content.parameters_min_bounds.items()]
        xu = [pair[1] for pair in fuzzing_content.parameters_max_bounds.items()]

        run_info = {
        'episode_max_time':fuzzing_arguments.episode_max_time,
        'ego_car_model':fuzzing_arguments.ego_car_model,
        'route_type':fuzzing_arguments.route_type,
        'call_from_dt':dt_arguments.call_from_dt,
        'town_name':sim_specific_arguments.town_name,
        'scenario':sim_specific_arguments.scenario,
        'direction':sim_specific_arguments.direction,
        'route_str':sim_specific_arguments.route_str,

        'waypoints_num_limit':fuzzing_content.search_space_info.waypoints_num_limit, 'num_of_static_max':fuzzing_content.search_space_info.num_of_static_max, 'num_of_pedestrians_max':fuzzing_content.search_space_info.num_of_pedestrians_max, 'num_of_vehicles_max':fuzzing_content.search_space_info.num_of_vehicles_max,
        'xl': np.array(xl),
        'xu': np.array(xu),
        'customized_center_transforms':fuzzing_content.customized_center_transforms,
        'parameters_min_bounds':fuzzing_content.parameters_min_bounds,
        'parameters_max_bounds':fuzzing_content.parameters_max_bounds,
        'labels': fuzzing_content.labels,
        'mask': fuzzing_content.mask,
        'customized_constraints': fuzzing_content.customized_constraints,

        'x': x,
        'objectives': objectives,
        'is_bug': is_bug,
        'bug_type': bug_type,
        'loc': loc,
        'object_type': object_type,
        'route_completion': route_completion,
        'all_final_generated_transforms': all_final_generated_transforms,
        'fuzzing_content': fuzzing_content,
        'fuzzing_arguments': fuzzing_arguments,
        'sim_specific_arguments': sim_specific_arguments,
        'dt_arguments': dt_arguments,
        'tmp_save_path': tmp_save_path,
        'port': customized_data['port']
        }


        print('counter:', counter, 'bug:', is_bug, 'objectives:', objectives)

        with open(cur_folder+'/'+'cur_info.pickle', 'wb') as f_out:
            pickle.dump(run_info, f_out)

        try:
            # print('tmp_save_path, cur_folder', tmp_save_path, cur_folder)
            copy_tree(tmp_save_path, cur_folder)
        except:
            print('fail to copy from', tmp_save_path)
            traceback.print_exc()


    return objectives, run_info


def parse_route_and_scenario(
    location_list, town_name, scenario, direction, route_str, scenario_file
):

    # Parse Route
    TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
    <routes>
    %s
    </routes>"""

    print(location_list, town_name, scenario, direction, route_str)

    pitch = 0
    roll = 0
    yaw = 0
    z = 0

    start_str = '<route id="{}" town="{}">\n'.format(route_str, town_name)
    waypoint_template = (
        '\t<waypoint pitch="{}" roll="{}" x="{}" y="{}" yaw="{}" z="{}" />\n'
    )
    end_str = "</route>"

    wp_str = ""

    for x, y in location_list:
        wp = waypoint_template.format(pitch, roll, x, y, yaw, z)
        wp_str += wp

    final_str = start_str + wp_str + end_str

    folder = make_hierarchical_dir(
        ["carla_lbc/leaderboard/data/customized_routes", town_name, scenario, direction]
    )

    pathlib.Path(folder + "/route_{}.xml".format(route_str)).write_text(
        TEMPLATE % final_str
    )

    # Parse Scenario
    x_0, y_0 = location_list[0]
    parse_scenario(scenario_file, town_name, route_str, x_0, y_0)

def parse_scenario(scenario_file, town_name, route_str, x_0, y_0):
    # Parse Scenario
    x_0_str = str(x_0)
    y_0_str = str(y_0)

    new_scenario = {
        "available_scenarios": [
            {
                town_name: [
                    {
                        "available_event_configurations": [
                            {
                                "route": int(route_str),
                                "center": {
                                    "pitch": "0.0",
                                    "x": x_0_str,
                                    "y": y_0_str,
                                    "yaw": "270",
                                    "z": "0.0",
                                },
                                "transform": {
                                    "pitch": "0.0",
                                    "x": x_0_str,
                                    "y": y_0_str,
                                    "yaw": "270",
                                    "z": "0.0",
                                },
                            }
                        ],
                        "scenario_type": "Scenario12",
                    }
                ]
            }
        ]
    }

    with open(scenario_file, "w") as f_out:
        annotation_dict = json.dump(new_scenario, f_out, indent=4)


def initialize_carla_specific(fuzzing_arguments):

    route_info = customized_routes[fuzzing_arguments.route_type]

    town_name = route_info['town_name']
    scenario = 'Scenario12' # This is only for compatibility purpose
    direction = route_info['direction']
    route = route_info['route_id']
    location_list = route_info['location_list']


    scenario_file = initialize_tmp_scenario_file()


    route_str = str(route)
    if route < 10:
        route_str = '0'+route_str

    parse_route_and_scenario(location_list, town_name, scenario, direction, route_str, scenario_file)

    sim_specific_arguments = emptyobject(
    town_name=town_name,
    scenario=scenario,
    direction=direction,
    route_str=route_str,
    scenario_file=scenario_file,
    location_list=location_list,
    correct_spawn_locations_after_run=False,
    correct_spawn_locations=correct_spawn_locations_all)


    return sim_specific_arguments


def initialize_tmp_scenario_file():
    now = datetime.now()
    time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    scenario_folder = 'carla_lbc/scenario_files'
    if not os.path.exists(scenario_folder):
        os.mkdir(scenario_folder)
    scenario_file = scenario_folder+'/'+'current_scenario_'+time_str+'.json'

    return scenario_file





def get_event_location_and_object_type(subfolders, verbose=True):
    location_list = []
    object_type_list = []

    for subfolder in subfolders:
        _, (x, y), object_type, route_completion = estimate_objectives(subfolder, verbose=verbose)
        location_list.append((x, y))
        object_type_list.append(object_type)
    locations = np.array(location_list)
    return locations, object_type_list



# ---------------- Bug, Objective -------------------
def check_bug(objectives):
    # speed needs to be larger than 0.1 to avoid false positive
    return objectives[0] > 0.1 or objectives[-3] or objectives[-2] or objectives[-1]

def get_if_bug_list(objectives_list):
    if_bug_list = []
    for objective in objectives_list:
        if_bug_list.append(check_bug(objective))
    return np.array(if_bug_list)


def process_specific_bug(
    bug_type_ind, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
):
    if len(bugs) == 0:
        return [], [], 0
    verbose = True
    chosen_bugs = np.array(bugs_type_list) == bug_type_ind

    specific_bugs = np.array(bugs)[chosen_bugs]
    specific_bugs_inds_list = np.array(bugs_inds_list)[chosen_bugs]

    # unique_specific_bugs, specific_distinct_inds = get_distinct_data_points(
    #     specific_bugs, mask, xl, xu, p, c, th
    # )

    specific_distinct_inds = is_distinct_vectorized(specific_bugs, [], mask, xl, xu, p, c, th, verbose=verbose)
    unique_specific_bugs = specific_bugs[specific_distinct_inds]

    unique_specific_bugs_inds_list = specific_bugs_inds_list[specific_distinct_inds]

    return (
        list(unique_specific_bugs),
        list(unique_specific_bugs_inds_list),
        len(unique_specific_bugs),
    )

def classify_bug_type(objectives, object_type=''):
    bug_str = ''
    bug_type = 5
    if objectives[0] > 0.1:
        collision_types = {'pedestrian_collision':pedestrian_types, 'car_collision':car_types, 'motercycle_collision':motorcycle_types, 'cyclist_collision':cyclist_types, 'static_collision':static_types}
        for k,v in collision_types.items():
            if object_type in v:
                bug_str = k
        if not bug_str:
            bug_str = 'unknown_collision'+'_'+object_type
        bug_type = 1
    elif objectives[-3]:
        bug_str = 'offroad'
        bug_type = 2
    elif objectives[-2]:
        bug_str = 'wronglane'
        bug_type = 3
    if objectives[-1]:
        bug_str += 'run_red_light'
        if bug_type > 4:
            bug_type = 4
    return bug_type, bug_str

def get_unique_bugs(
    X, objectives_list, mask, xl, xu, unique_coeff, objective_weights, return_mode='unique_inds_and_interested_and_bugcounts', consider_interested_bugs=1, bugs_type_list=[], bugs=[], bugs_inds_list=[], trajectory_vector_list=[]
):
    p, c, th = unique_coeff
    # hack:
    if len(bugs) == 0:
        for i, (x, objectives) in enumerate(zip(X, objectives_list)):
            if check_bug(objectives):
                bug_type, _ = classify_bug_type(objectives)
                bugs_type_list.append(bug_type)
                bugs.append(x)
                bugs_inds_list.append(i)

    (
        unique_collision_bugs,
        unique_collision_bugs_inds_list,
        unique_collision_num,
    ) = process_specific_bug(
        1, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )
    (
        unique_offroad_bugs,
        unique_offroad_bugs_inds_list,
        unique_offroad_num,
    ) = process_specific_bug(
        2, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )
    (
        unique_wronglane_bugs,
        unique_wronglane_bugs_inds_list,
        unique_wronglane_num,
    ) = process_specific_bug(
        3, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )
    (
        unique_redlight_bugs,
        unique_redlight_bugs_inds_list,
        unique_redlight_num,
    ) = process_specific_bug(
        4, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )

    unique_bugs = unique_collision_bugs + unique_offroad_bugs + unique_wronglane_bugs + unique_redlight_bugs
    unique_bugs_num = len(unique_bugs)
    unique_bugs_inds_list = unique_collision_bugs_inds_list + unique_offroad_bugs_inds_list + unique_wronglane_bugs_inds_list + unique_redlight_bugs_inds_list

    if consider_interested_bugs:
        collision_activated = np.sum(objective_weights[:3] != 0) > 0
        offroad_activated = (np.abs(objective_weights[3]) > 0) | (
            np.abs(objective_weights[5]) > 0
        )
        wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
            np.abs(objective_weights[5]) > 0
        )
        red_light_activated = np.abs(objective_weights[-1]) > 0

        interested_unique_bugs = []
        if collision_activated:
            interested_unique_bugs += unique_collision_bugs
        if offroad_activated:
            interested_unique_bugs += unique_offroad_bugs
        if wronglane_activated:
            interested_unique_bugs += unique_wronglane_bugs
        if red_light_activated:
            interested_unique_bugs += unique_redlight_bugs
    else:
        interested_unique_bugs = unique_bugs

    num_of_collisions = np.sum(np.array(bugs_type_list)==1)
    num_of_offroad = np.sum(np.array(bugs_type_list)==2)
    num_of_wronglane = np.sum(np.array(bugs_type_list)==3)
    num_of_redlight = np.sum(np.array(bugs_type_list)==4)

    if return_mode == 'unique_inds_and_interested_and_bugcounts':
        return unique_bugs, unique_bugs_inds_list, interested_unique_bugs, [num_of_collisions, num_of_offroad, num_of_wronglane, num_of_redlight,
        unique_collision_num, unique_offroad_num, unique_wronglane_num, unique_redlight_num]
    elif return_mode == 'return_bug_info':
        return unique_bugs, (bugs, bugs_type_list, bugs_inds_list, interested_unique_bugs)
    elif return_mode == 'return_indices':
        return unique_bugs, unique_bugs_inds_list
    else:
        return unique_bugs


def choose_weight_inds(objective_weights):
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    red_light_activated = np.abs(objective_weights[-1]) > 0

    if collision_activated:
        weight_inds = np.arange(0,3)
    elif offroad_activated or wronglane_activated:
        weight_inds = np.arange(3,6)
    elif red_light_activated:
        weight_inds = np.arange(9,10)
    else:
        raise
    return weight_inds

def determine_y_upon_weights(objective_list, objective_weights):
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    red_light_activated = np.abs(objective_weights[-1]) > 0

    y = np.zeros(len(objective_list))
    for i, obj in enumerate(objective_list):
        cond = 0
        if collision_activated:
            cond |= obj[0] > 0.1
        if offroad_activated:
            cond |= obj[-3] == 1
        if wronglane_activated:
            cond |= obj[-2] == 1
        if red_light_activated:
            cond |= obj[-1] == 1
        y[i] = cond

    return y

def get_all_y(objective_list, objective_weights):
    # is_collision, is_offroad, is_wrong_lane, is_run_red_light
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    red_light_activated = np.abs(objective_weights[-1]) > 0

    y_list = np.zeros((4, len(objective_list)))

    for i, obj in enumerate(objective_list):
        if collision_activated:
            y_list[0, i] = obj[0] > 0.1
        if offroad_activated:
            y_list[1, i] = obj[-3] == 1
        if wronglane_activated:
            y_list[2, i] = obj[-2] == 1
        if red_light_activated:
            y_list[3, i] = obj[-1] == 1

    return y_list
# ---------------- Bug, Objective -------------------





if __name__ == '__main__':
    print('ok')
