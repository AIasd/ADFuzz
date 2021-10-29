import sys
import os
sys.path.append('pymoo')
carla_root = '../carla_0994_no_rss'
sys.path.append(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append(carla_root+'/PythonAPI/carla')
sys.path.append(carla_root+'/PythonAPI')
sys.path.append('.')
sys.path.append('leaderboard')
sys.path.append('leaderboard/team_code')
sys.path.append('scenario_runner')
sys.path.append('scenario_runner')
sys.path.append('carla_project/src')




from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.duplicate import ElementwiseDuplicateElimination, NoDuplicateElimination

from pymoo.model.population import Population
from pymoo.model.evaluator import Evaluator

from pymoo.algorithms.nsga2 import NSGA2, binary_tournament
from pymoo.algorithms.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.algorithms.random import RandomAlgorithm

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover

from pymoo.performance_indicator.hv import Hypervolume

import matplotlib.pyplot as plt

from object_types import WEATHERS, pedestrian_types, vehicle_types, static_types, vehicle_colors, car_types, motorcycle_types, cyclist_types, weather_names

from customized_utils import  rand_real, make_hierarchical_dir, exit_handler, arguments_info, is_critical_region, setup_bounds_mask_labels_distributions_stage1, setup_bounds_mask_labels_distributions_stage2, customize_parameters, customized_bounds_and_distributions, static_general_labels, pedestrian_general_labels, vehicle_general_labels, waypoint_labels, waypoints_num_limit, if_violate_constraints, customized_routes, get_distinct_data_points, is_similar, check_bug, is_distinct, filter_critical_regions, parse_scenario, estimate_objectives, parse_route_and_scenario_plain

from carla_specific_utils.carla_specific import convert_x_to_customized_data, parse_route_file
from carla_specific_utils.carla_specific_tools import create_transform
from collections import deque


import numpy as np
import carla

from leaderboard.fuzzing import LeaderboardEvaluator
from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.statistics_manager import StatisticsManager
from carla_specific_utils.object_params import Static, Pedestrian, Vehicle

import traceback
import json
import re
import time
from datetime import datetime

import pathlib
import shutil
import dill as pickle
# import pickle
import argparse
import atexit
import traceback
import math



import copy

from pymoo.factory import get_termination
from pymoo.model.termination import Termination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination, SingleObjectiveDefaultTermination
from pymoo.util.termination.max_time import TimeBasedTermination
from pymoo.model.individual import Individual
from pymoo.model.repair import Repair
from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_crossover, get_mutation
from pymoo.model.mating import Mating

from dask.distributed import Client, LocalCluster

from pymoo.model.initialization import Initialization
from pymoo.model.duplicate import NoDuplicateElimination
from pymoo.model.individual import Individual
from pymoo.operators.sampling.random_sampling import FloatRandomSampling


import pandas as pd






parser = argparse.ArgumentParser()
parser.add_argument('-p','--ports', type=int, default=2003, help='TCP port to listen to (default: 2003)')
parser.add_argument("-r", "--route_type", type=str, default='town05_right_0')
parser.add_argument("-c", "--scenario_type", type=str, default='default')

parser.add_argument("-m", "--ego_car_model", type=str, default='lbc')
parser.add_argument("--has_display", type=str, default='0')
parser.add_argument("--root_folder", type=str, default='run_results')

parser.add_argument("--episode_max_time", type=int, default=50)

arguments = parser.parse_args()


ports = arguments.ports


# ['none', 'town01_left_0', 'town07_front_0', 'town05_front_0', 'town05_right_0']
route_type = arguments.route_type
# ['default', 'leading_car_braking', 'vehicles_only', 'no_static']
scenario_type = arguments.scenario_type
# ['lbc', 'auto_pilot', 'pid_agent']
ego_car_model = arguments.ego_car_model

os.environ['HAS_DISPLAY'] = arguments.has_display
root_folder = arguments.root_folder


episode_max_time = arguments.episode_max_time




random_seed = 1000
rng = np.random.default_rng(random_seed)

now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

scenario_folder = 'carla_lbc/scenario_files'
if not os.path.exists(scenario_folder):
    os.mkdir(scenario_folder)
scenario_file = scenario_folder+'/'+'current_scenario_'+time_str+'.json'

# This is used to control how this program use GPU
# '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
default_objectives = [0, 7, 7, 7, 0, 0, 0, 0, 0]




def run_simulation(customized_data, launch_server, episode_max_time, route_path, route_str, scenario_file, ego_car_model, background_vehicles=False, changing_weather=False):
    arguments = arguments_info()
    arguments.port = customized_data['port']

    arguments.background_vehicles = background_vehicles

    # model path and checkpoint path
    if ego_car_model == 'lbc':
        arguments.agent = 'scenario_runner/team_code/image_agent.py'
        arguments.agent_config = 'models/epoch=24.ckpt'
        base_save_folder = 'collected_data_customized'
    elif ego_car_model == 'auto_pilot':
        arguments.agent = 'leaderboard/team_code/auto_pilot.py'
        arguments.agent_config = ''
        base_save_folder = 'collected_data_autopilot'
    elif ego_car_model == 'pid_agent':
        arguments.agent = 'scenario_runner/team_code/pid_agent.py'
        arguments.agent_config = ''
        base_save_folder = 'collected_data_pid_agent'
    elif ego_car_model == 'map_model':
        arguments.agent = 'scenario_runner/team_code/map_agent.py'
        arguments.agent_config = 'models/stage1_default_50_epoch=16.ckpt'
        base_save_folder = 'collected_data_map_model'
    else:
        print('unknown ego_car_model:', ego_car_model)
        raise




    statistics_manager = StatisticsManager()
    # Fixed Hyperparameters
    sample_factor = 5
    arguments.debug = 0


    # Laundry Stuff-------------------------------------------------------------
    arguments.weather_index = customized_data['weather_index']
    os.environ['WEATHER_INDEX'] = str(customized_data['weather_index'])


    # used to read scenario file
    arguments.scenarios = scenario_file

    # used to compose folder to save real-time data
    os.environ['SAVE_FOLDER'] = arguments.save_folder

    # used to read route to run; used to compose folder to save real-time data
    arguments.routes = route_path
    os.environ['ROUTES'] = arguments.routes

    # used to record real time deviation data
    arguments.deviations_folder = arguments.save_folder + '/' + pathlib.Path(os.environ['ROUTES']).stem

    # used to read real-time data
    save_path = arguments.save_folder + '/' + pathlib.Path(os.environ['ROUTES']).stem


    arguments.changing_weather = changing_weather



    # extract waypoints along route
    import xml.etree.ElementTree as ET
    tree = ET.parse(arguments.routes)
    route_waypoints = []


    for route in tree.iter("route"):
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

        objectives, loc, object_type, route_completion = estimate_objectives(save_path, default_objectives)


    return objectives, loc, object_type, route_completion







def sample_within_bounds(xl, xu, mask, labels, parameters_distributions, customized_constraints, ego_x=0, ego_y=0):

    def sample_one_feature(typ, lower, upper, dist, label):
        assert lower <= upper, label+','+str(lower)+'>'+str(upper)
        if typ == 'int':
            val = rng.integers(lower, upper+1)
        elif typ == 'real':
            if dist[0] == 'normal':
                if dist[1] == None:
                    mean = (lower+upper)/2
                else:
                    mean = dist[1]
                val = rng.normal(mean, dist[2], 1)[0]
            else: # default is uniform
                val = rand_real(rng, lower, upper)
            val = np.clip(val, lower, upper)
        return val


    max_sample_times = 100
    max_inner_sample_times = 500
    sample_times = 0
    while sample_times < max_sample_times:
        sample_times += 1
        x = []
        for i, dist in enumerate(parameters_distributions):
            typ = mask[i]
            lower = xl[i]
            upper = xu[i]
            label = labels[i]
            val = sample_one_feature(typ, lower, upper, dist, label)
            x.append(val)


        # handle case of being too close to ego car
        if ego_x != 0 and ego_y != 0:
            x_radius = 2
            y_radius = 3

            ped_x_min = ego_x - x_radius*1.5
            ped_x_max = ego_x + x_radius*1.5
            ped_y_min = ego_y - y_radius*1.5
            ped_y_max = ego_y + y_radius*1.5

            car_x_min = ego_x - x_radius*1.5
            car_x_max = ego_x + x_radius*1.5
            car_y_min = ego_y - y_radius*1.5
            car_y_max = ego_y + y_radius*1.5



            x_violation = False
            ind = 0
            dim = len(labels)
            inner_sample_times = 0
            while ind < dim and inner_sample_times < max_inner_sample_times:
                if (label.startswith('pedestrian_x') and ped_x_min <= val <= ped_x_max) or (label.startswith('vehicle_x') and car_x_min <= val <= car_x_max):
                    x_violation = True
                elif ((label.startswith('pedestrian_y') and ped_y_min <= val <= ped_y_max) or (label.startswith('vehicle_y') and car_y_min <= val <= car_y_max)) and x_violation == True:
                    x[ind-1] = sample_one_feature(typ, lower, upper, dist, label)
                    x[ind] = sample_one_feature(typ, lower, upper, dist, label)
                    x_violation = False
                    ind -= 2
                    inner_sample_times += 1
                    print(inner_sample_times)
                ind += 1



        if not if_violate_constraints(x, customized_constraints, labels):
            x = np.array(x).astype(float)
            break
    return x



def get_bounds(customized_config, use_fine_grained_weather):
    customized_parameters_bounds = customized_config['customized_parameters_bounds']
    customized_parameters_distributions = customized_config['customized_parameters_distributions']
    customized_center_transforms = customized_config['customized_center_transforms']
    customized_constraints = customized_config['customized_constraints']

    fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels = setup_bounds_mask_labels_distributions_stage1(use_fine_grained_weather)
    customize_parameters(parameters_min_bounds, customized_parameters_bounds)
    customize_parameters(parameters_max_bounds, customized_parameters_bounds)


    fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels, parameters_distributions, n_var = setup_bounds_mask_labels_distributions_stage2(fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels)
    customize_parameters(parameters_min_bounds, customized_parameters_bounds)
    customize_parameters(parameters_max_bounds, customized_parameters_bounds)
    customize_parameters(parameters_distributions, customized_parameters_distributions)



    xl = [pair[1] for pair in parameters_min_bounds.items()]
    xu = [pair[1] for pair in parameters_max_bounds.items()]


    return xl, xu, mask, labels, parameters_distributions, parameters_min_bounds, parameters_max_bounds, customized_center_transforms, customized_constraints





def get_customized_data(port, scenario_type, route_type, ego_x=0, ego_y=0):
    if route_type == 'none':
        use_fine_grained_weather = False
    else:
        use_fine_grained_weather = False
    customized_d = customized_bounds_and_distributions[scenario_type]
    xl, xu, mask, labels, parameters_distributions, parameters_min_bounds, parameters_max_bounds, customized_center_transforms, customized_constraints = get_bounds(customized_d, use_fine_grained_weather)
    x = sample_within_bounds(xl, xu, mask, labels, parameters_distributions, customized_constraints, ego_x, ego_y)


    num_of_static_max = parameters_max_bounds['num_of_static_max']
    num_of_pedestrians_max = parameters_max_bounds['num_of_pedestrians_max']
    num_of_vehicles_max = parameters_max_bounds['num_of_vehicles_max']


    x = np.append(x, port)

    customized_data = convert_x_to_customized_data(x, waypoints_num_limit, num_of_static_max, num_of_pedestrians_max, num_of_vehicles_max, static_types, pedestrian_types, vehicle_types, vehicle_colors, customized_center_transforms, parameters_min_bounds, parameters_max_bounds)


    other_info = [x, xl, xu, mask, labels]

    return customized_data, other_info



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
            # print('x', pop[i].X[x_j_ind], '->', x)
            # print('y', pop[i].X[y_j_ind], '->', y)
            # print('yaw', pop[i].X[yaw_j_ind], '->', yaw)
            x_data[x_j_ind] = x
            x_data[y_j_ind] = y
            x_data[yaw_j_ind] = yaw

            if len(empty_slots) > 0:
                q = empty_slots.popleft()
                print('shift', j, 'to', q)
                for k in keys:
                    print(k)
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



def run_one_route(port, scenario_type, route_type, launch_server):
    episode_max_time = 50

    route_info = customized_routes[route_type]
    town_name = route_info['town_name']
    route = route_info['route_id']
    location_list = route_info['location_list']

    route_str = str(route)
    if route < 10:
        route_str = '0'+route_str

    route_path = parse_route_and_scenario_plain(location_list, town_name, route_str, scenario_file)

    customized_data, other_info = get_customized_data(port, scenario_type, route_type)

    objectives, loc, object_type, route_completion = run_simulation(customized_data, launch_server, episode_max_time, route_path, route_str, scenario_file, ego_car_model)


    ego_linear_speed, min_d, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light, is_collision = objectives
    accident_x, accident_y = loc

    is_bug = check_bug(objectives)

    result_info = [ego_linear_speed, min_d, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light, is_collision, accident_x, accident_y, is_bug, route_completion]


    x, xl, xu, mask, labels = other_info
    x = x[:-1]
    print(x.shape, x)



    data = x.copy()
    # correct x for generation discrepancy
    from collections import OrderedDict
    assert len(x) == len(labels)
    label_to_id = {label:i for i, label in enumerate(labels)}

    filename = 'tmp_folder/'+str(port)+'.pickle'
    with open(filename, 'rb') as f_in:
        all_final_generated_transforms = pickle.load(f_in)

    correct_spawn_locations(data, label_to_id, all_final_generated_transforms, 'static', static_general_labels)
    correct_spawn_locations(data, label_to_id, all_final_generated_transforms, 'pedestrian', pedestrian_general_labels)
    correct_spawn_locations(data, label_to_id, all_final_generated_transforms, 'vehicle', vehicle_general_labels)

    # add label and value of resulting variables to x
    id_to_label = {}
    id_to_dist = {}
    with open(customized_data['tmp_travel_dist_file'], 'r') as f_in:
        for line in f_in:
            tokens = line.strip().split(',')
            if len(tokens) == 3:
                actor_id, general_actor_type, index  = tokens
                id_to_label[actor_id] = '_'.join([general_actor_type, 'dist_to_travel', index])
            elif len(tokens) == 2:
                actor_id = tokens[0]
                dist = float(tokens[1])
                if actor_id not in id_to_dist or (actor_id in id_to_dist and dist > id_to_dist[actor_id]):
                    id_to_dist[actor_id] = dist

    for actor_id in id_to_label:
        label = id_to_label[actor_id]
        if actor_id in id_to_dist:
            dist = id_to_dist[actor_id]
        else:
            dist = 0
        print(labels)
        entry_i = labels.index(label)
        data[entry_i] = dist



    x = np.concatenate([x, result_info, [0]])
    data = np.concatenate([data, result_info, [1]])

    labels.extend(['ego_linear_speed', 'min_d', 'offroad_d', 'wronglane_d', 'dev_dist', 'is_offroad', 'is_wrong_lane', 'is_run_red_light', 'is_collision', 'accident_x', 'accident_y', 'is_bug', 'route_completion', 'is_adjusted_travel_dist'])





    return x, data, labels, objectives






def get_bug_type(objectives, object_type):
    bug_str = None
    if objectives[0] > 0:
        collision_types = {'pedestrian_collision':pedestrian_types, 'car_collision':car_types, 'motercycle_collision':motorcycle_types, 'cyclist_collision':cyclist_types, 'static_collision':static_types}
        for k,v in collision_types.items():
            if object_type in v:
                bug_str = k
        if not bug_str:
            bug_str = 'unknown_collision'+'_'+object_type
        bug_type = 1
    elif objectives[5]:
        bug_str = 'offroad'
        bug_type = 2
    elif objectives[6]:
        bug_str = 'wronglane'
        bug_type = 3
    else:
        bug_str = 'unknown'
        bug_type = 4

    return bug_str, bug_type



def run_multiple_routes(port, scenario_type, route_type):
    start_with_new_json = False
    continue_index = 29
    filter_out_center_npcs = False
    changing_weather = True
    background_vehicles = False
    # route_folder = 'leaderboard/data/new_routes'
    route_folder = 'leaderboard/data/routes'
    # episode_max_time = 50
    episode_max_time = 900


    # TBD: remove this repetition from route_scenario.py
    background_vehicle_num = {
            'Town01': 120,
            'Town02': 100,
            'Town03': 120,
            'Town04': 200,
            'Town05': 120,
            'Town06': 150,
            'Town07': 110,
            'Town08': 180,
            'Town09': 300,
            'Town10HD': 120,
        }



    total_bugs = 0

    # hack: this filename is hardcoded
    with open('route_to_center_d.pkl', 'rb') as f_in:
        route_to_center_d = pickle.load(f_in)


    route_files = sorted(os.listdir(route_folder))[continue_index:]


    # initialize json
    #########################################
    parent_folder = 'collected_data_customized'
    if not os.path.exists(parent_folder):
        os.mkdir(parent_folder)
    json_path = os.path.join(parent_folder, 'customized.json')

    if start_with_new_json or not os.path.exists(json_path):
        with open(json_path, 'w') as f_out:
            json.dump({"envs":{}, "package_name":"customized"}, f_out, indent=4)
    #########################################


    for i, route_file in enumerate(route_files):
        if i == 0:
            launch_server = True
        else:
            launch_server = False

        if not route_file.endswith('xml'):
            continue

        route_path = os.path.join(route_folder, route_file)
        route_str = route_file[6:-4]


        route_id, town_name, transform_list = parse_route_file(route_path)[0]



        x_0, y_0 = transform_list[0][:2]
        parse_scenario(scenario_file, town_name, str(route_id), x_0, y_0)

        if filter_out_center_npcs:
            center_transform = route_to_center_d[route_file]
            ego_x = x_0 - center_transform[0]
            ego_y = y_0 - center_transform[1]
        else:
            ego_x = 0
            ego_y = 0

        customized_data, _ = get_customized_data(port, scenario_type, route_type, ego_x, ego_y)


        objectives, _, object_type, _ = run_simulation(customized_data, launch_server, episode_max_time, route_path, route_str, scenario_file, ego_car_model, background_vehicles=background_vehicles, changing_weather=changing_weather)


        if background_vehicles:
            npc_vehicles_num = background_vehicle_num[town_name]
        else:
            npc_vehicles_num = 0


        # add the current route to json
        #########################################
        with open(json_path, 'r') as f_in:
            json_data = json.load(f_in)

        json_data["envs"][route_file[:-4]] = {
            "route": {
                "file": os.path.join('new_routes', route_file),
                "id": route_id
            },
            "scenarios": {
                "background_activity": {
                    "cross_factor": 0,
                    "vehicle.*": npc_vehicles_num,
                    "walker.*": 0
                }
            },
            "town_name": town_name,
            "vehicle_model": "vehicle.lincoln.mkz2017",
            "weather_profile": weather_names[customized_data['weather_index']]
        }

        with open(json_path, 'w') as f_out:
            json.dump(json_data, f_out, indent=4)

        # ------------------- meta json ------------------
        # TBD: make this path not hardcoded
        current_run_folder = os.path.join(parent_folder, route_file[:-4])
        if not os.path.exists(current_run_folder):
            os.mkdir(current_run_folder)
        metajson_path = os.path.join(current_run_folder, 'metadata.json')

        # we let id of the central camera to be "rgb_central" here to be compatible with action_based but it is actually "rgb" which is used to collect data for multiple scripts.
        # TBD: fix this inconsistency
        metajson = {
            "full_name":route_file[:-4],
            "scenarios": {
                "name": "ServerSideSensor"
            },
            "sensors": [
                    {
                        'type': 'sensor.camera.rgb',
                        'x': 1.3, 'y': 0.0, 'z': 1.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'width': 256, 'height': 144, 'fov': 90,
                        'id': 'rgb_central'
                        },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': 1.2, 'y': -0.25, 'z': 1.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                        'width': 256, 'height': 144, 'fov': 90,
                        'id': 'rgb_left'
                        },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': 1.2, 'y': 0.25, 'z': 1.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                        'width': 256, 'height': 144, 'fov': 90,
                        'id': 'rgb_right'
                        }
                    ],
            "set_of_weathers": weather_names[customized_data['weather_index']]
        }
        with open(metajson_path, 'w') as f_out:
            json.dump(metajson, f_out, indent=4)


        # ------------------- summary json ------------------
        summaryjson_path = os.path.join(current_run_folder, 'summary.json')

        # TBD: record the following info and potentially add more rather than feed in hardcoded meaningless numbers
        summaryjson = {
            "exp_name": route_file[:-4]+"_0_0",
            "help_text": "",
            "number_red_lights": 0,
            "result": "SUCCESS",
            "score_composed": 100.0,
            "score_penalty": 0.0,
            "score_route": 100.0,
            "total_number_traffic_lights": 1
        }
        with open(summaryjson_path, 'w') as f_out:
            json.dump(summaryjson, f_out, indent=4)

        #########################################


        if check_bug(objectives):
            bug_str, bug_type = get_bug_type(objectives, object_type)
            total_bugs += 1

        print(route_file, total_bugs, '/', i+1, 'are bugs')


if __name__ == '__main__':
    start_ind = 703
    port = 2003
    atexit.register(exit_handler, [port])

    scenario_type = 'one_pedestrians_cross_street_town05'
    route_type = 'town05_right_0'

    # scenario_type = 'none'
    # route_type = 'none'

    os.environ['HAS_DISPLAY'] = '0'


    if route_type == 'none':
        run_multiple_routes(port, scenario_type, route_type)
    else:
        new_path = 'collected_data_customized/one_ped_only'
        # if os.path.exists(new_path):
        #     shutil.rmtree(new_path)
        #     os.mkdir(new_path)

        labels = None
        trial_num = 298
        data_list = []
        collision_bug_count = 0
        for i in range(trial_num):
            if i == 0:
                launch_server = True
            else:
                launch_server = False
            ind = start_ind + i
            print('-'*50, ind, '-'*50)
            x, data, labels, objectives = run_one_route(port, scenario_type, route_type, launch_server)

            data_list.append(x)
            data_list.append(data)

            # hack: this is currently hard-coded, should be more customizable
            save_path = 'collected_data_customized/route_00'

            new_path_ind = new_path+'/'+str(ind)
            shutil.copytree(save_path, new_path_ind)
            np.savez(os.path.join(new_path_ind, 'results'), x=x, data=data, labels=labels, objectives=objectives)

            collision_bug_count += objectives[-1]
            print('collision bugs:', collision_bug_count, '/', ind)

        print(labels)
        for data in data_list:
            print(data)
        df = pd.DataFrame(np.array(data_list), columns=labels)
        df.to_pickle("initial_observations_1.pkl")
