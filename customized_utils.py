import os
import sys
import argparse
import socket
import random
import torch
import subprocess
import re
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from psutil import process_iter
from sklearn import tree
from sklearn.preprocessing import StandardScaler, OneHotEncoder



# ---------------- Misc -------------------
class emptyobject():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __str__(self):
        return str(self.__dict__)

class arguments_info:
    def __init__(self):
        self.host = "localhost"
        self.port = "2000"
        self.sync = False
        self.debug = 0
        self.spectator = True
        self.record = ""
        self.timeout = "30.0"
        self.challenge_mode = True
        self.routes = None
        self.scenarios = "leaderboard/data/all_towns_traffic_scenarios_public.json"
        self.repetitions = 1
        self.agent = "scenario_runner/team_code/image_agent.py"
        self.agent_config = "models/epoch=24.ckpt"
        self.track = "SENSORS"
        self.resume = False
        self.checkpoint = ""
        self.weather_index = 19
        self.save_folder = "carla_lbc/collected_data_customized"
        self.deviations_folder = ""
        self.background_vehicles = False
        self.save_action_based_measurements = 0
        self.changing_weather = False
        self.record_every_n_step = 2000

def specify_args():
    # general parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", default="localhost", help="IP of the host server (default: localhost)"
    )
    parser.add_argument(
        "--port", default="2000", help="TCP port to listen to (default: 2000)"
    )
    parser.add_argument(
        "--sync", action="store_true", help="Forces the simulation to run synchronously"
    )
    parser.add_argument("--debug", type=int, help="Run with debug output", default=0)
    parser.add_argument(
        "--spectator", type=bool, help="Switch spectator view on?", default=True
    )
    parser.add_argument(
        "--record",
        type=str,
        default="",
        help="Use CARLA recording feature to create a recording of the scenario",
    )
    # modification: 30->40
    parser.add_argument(
        "--timeout",
        default="30.0",
        help="Set the CARLA client timeout value in seconds",
    )

    # simulation setup
    parser.add_argument(
        "--challenge-mode", action="store_true", help="Switch to challenge mode?"
    )
    parser.add_argument(
        "--routes",
        help="Name of the route to be executed. Point to the route_xml_file to be executed.",
        required=False,
    )
    parser.add_argument(
        "--scenarios",
        help="Name of the scenario annotation file to be mixed with the route.",
        required=False,
    )
    parser.add_argument(
        "--repetitions", type=int, default=1, help="Number of repetitions per route."
    )

    # agent-related options
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        help="Path to Agent's py file to evaluate",
        required=False,
    )
    parser.add_argument(
        "--agent-config",
        type=str,
        help="Path to Agent's configuration file",
        default="",
    )

    parser.add_argument(
        "--track", type=str, default="SENSORS", help="Participation track: SENSORS, MAP"
    )
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="Resume execution from last checkpoint?",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./simulation_results.json",
        help="Path to checkpoint used for saving statistics and resuming",
    )

    # addition
    parser.add_argument(
        "--weather-index", type=int, default=0, help="see WEATHER for reference"
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="collected_data",
        help="Path to save simulation data",
    )
    parser.add_argument(
        "--deviations-folder",
        type=str,
        default="",
        help="Path to the folder that saves deviations data",
    )
    parser.add_argument("--save_action_based_measurements", type=int, default=0)
    parser.add_argument("--changing_weather", type=int, default=0)

    parser.add_argument('--record_every_n_step', type=int, default=2000)

    arguments = parser.parse_args()

    return arguments

def parse_fuzzing_arguments():
    # [ego_linear_speed, min_d, d_angle_norm, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light, is_collision]
    default_objective_weights = np.array([-1., 1., 1., 1., 1., -1., 0., 0., 0., 0.])
    default_objectives = np.array([0., 20., 1., 7., 7., 0., 0., 0., 0., 0.])
    default_check_unique_coeff = [0, 0.1, 0.5]


    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("-r", "--route_type", type=str, default='town05_right_0')
    parser.add_argument("-c", "--scenario_type", type=str, default='default')
    parser.add_argument("-m", "--ego_car_model", type=str, default='lbc')
    parser.add_argument('-a','--algorithm_name', type=str, default='nsga2')

    parser.add_argument('-p','--ports', nargs='+', type=int, default=[2003], help='TCP port(s) to listen to (default: 2003)')
    parser.add_argument("-s", "--scheduler_port", type=int, default=8785)
    parser.add_argument("-d", "--dashboard_address", type=int, default=8786)

    parser.add_argument('--simulator', type=str, default='carla')

    parser.add_argument('--random_seed', type=int, default=0)


    # carla specific
    parser.add_argument("--has_display", type=str, default='0')
    parser.add_argument("--debug", type=int, default=1, help="whether using the debug mode: planned paths will be visualized.")
    parser.add_argument('--correct_spawn_locations_after_run', type=int, default=0)

    # carla_op specific
    parser.add_argument('--carla_path', type=str, default="../carla_0911_rss/CarlaUE4.sh")

    # no_simulation specific
    parser.add_argument('--no_simulation_data_path', type=str, default=None)
    parser.add_argument('--objective_labels', type=str, nargs='+', default=[])



    # logistic
    parser.add_argument("--root_folder", type=str, default='carla_lbc/run_results')
    parser.add_argument("--parent_folder", type=str, default='') # will be automatically created
    parser.add_argument("--mean_objectives_across_generations_path", type=str, default='') # will be automatically created
    parser.add_argument("--episode_max_time", type=int, default=60)
    parser.add_argument('--record_every_n_step', type=int, default=2000)
    parser.add_argument('--gpus', type=str, default='0,1')


    # algorithm related
    parser.add_argument("--n_gen", type=int, default=2)
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--survival_multiplier", type=int, default=1)
    parser.add_argument("--n_offsprings", type=int, default=300)
    parser.add_argument("--has_run_num", type=int, default=1000)
    parser.add_argument('--sample_multiplier', type=int, default=200)
    parser.add_argument('--mating_max_iterations', type=int, default=200)
    parser.add_argument('--only_run_unique_cases', type=int, default=1)
    parser.add_argument('--consider_interested_bugs', type=int, default=1)

    parser.add_argument("--outer_iterations", type=int, default=3)
    parser.add_argument('--objective_weights', nargs='+', type=float, default=default_objective_weights)
    parser.add_argument('--default_objectives', nargs='+', type=float, default=default_objectives)
    parser.add_argument("--standardize_objective", type=int, default=1)
    parser.add_argument("--normalize_objective", type=int, default=1)
    parser.add_argument('--traj_dist_metric', type=str, default='nearest')



    parser.add_argument('--check_unique_coeff', nargs='+', type=float, default=default_check_unique_coeff)
    parser.add_argument('--use_single_objective', type=int, default=1)
    parser.add_argument('--rank_mode', type=str, default='none')
    parser.add_argument('--ranking_model', type=str, default='nn_pytorch')
    parser.add_argument('--initial_fit_th', type=int, default=100, help='minimum number of instances needed to train a DNN.')
    parser.add_argument('--min_bug_num_to_fit_dnn', type=int, default=10, help='minimum number of bug instances needed to train a DNN.')

    parser.add_argument('--pgd_eps', type=float, default=1.01)
    parser.add_argument('--adv_conf_th', type=float, default=-4)
    parser.add_argument('--attack_stop_conf', type=float, default=0.9)
    parser.add_argument('--use_single_nn', type=int, default=1)

    parser.add_argument('--warm_up_path', type=str, default=None)
    parser.add_argument('--warm_up_len', type=int, default=-1)
    parser.add_argument('--regression_nn_use_running_data', type=int, default=1)

    parser.add_argument('--sample_avoid_ego_position', type=int, default=0)


    parser.add_argument('--uncertainty', type=str, default='')
    parser.add_argument('--model_type', type=str, default='one_output')


    parser.add_argument('--termination_condition', type=str, default='generations')
    parser.add_argument('--max_running_time', type=int, default=3600*24)

    parser.add_argument('--emcmc', type=int, default=0)
    parser.add_argument('--use_unique_bugs', type=int, default=1)
    parser.add_argument('--finish_after_has_run', type=int, default=1)

    fuzzing_arguments = parser.parse_args()


    os.environ['HAS_DISPLAY'] = fuzzing_arguments.has_display
    os.environ['CUDA_VISIBLE_DEVICES'] = fuzzing_arguments.gpus
    fuzzing_arguments.objective_weights = np.array(fuzzing_arguments.objective_weights)
    # ['BNN', 'one_output']
    # BALD and BatchBALD only support BNN
    if fuzzing_arguments.uncertainty.split('_')[0] in ['BALD', 'BatchBALD']:
        fuzzing_arguments.model_type = 'BNN'

    if 'un' in fuzzing_arguments.algorithm_name:
        fuzzing_arguments.use_unique_bugs = 1
    else:
        fuzzing_arguments.use_unique_bugs = 0

    if fuzzing_arguments.algorithm_name in ['nsga2-emcmc', 'nsga2-un-emcmc']:
        fuzzing_arguments.emcmc = 1
    else:
        fuzzing_arguments.emcmc = 0

    return fuzzing_arguments



def make_hierarchical_dir(folder_names):
    cur_folder_name = ""
    for i in range(len(folder_names)):
        cur_folder_name += folder_names[i]
        if not os.path.exists(cur_folder_name):
            os.mkdir(cur_folder_name)
        cur_folder_name += "/"
    return cur_folder_name

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", int(port))) == 0

def port_to_gpu(port):
    n = torch.cuda.device_count()
    # n = 2
    gpu = port % n

    return gpu

# TBD: separate the two exit handlers
def exit_handler(ports):
    # carla
    for port in ports:
        while is_port_in_use(port):
            try:
                subprocess.run("kill -9 $(lsof -t -i :" + str(port) + ")", shell=True)
                # subprocess.run("sudo kill $(lsof -t -i :" + str(port) + ")", shell=True)
                print("-" * 20, "kill server at port", port)
            except:
                continue
    # svl
    import psutil
    PROC_NAME = "mainboard"
    for proc in psutil.process_iter():
        # check whether the process to kill name matches
        if proc.name() == PROC_NAME:
            proc.kill()
            # subprocess.run("sudo kill -9 " + str(proc.pid), shell=True)


def get_sorted_subfolders(parent_folder, folder_type='all'):
    if 'rerun_bugs' in os.listdir(parent_folder):
        bug_folder = os.path.join(parent_folder, "rerun_bugs")
        non_bug_folder = os.path.join(parent_folder, "rerun_non_bugs")
    else:
        bug_folder = os.path.join(parent_folder, "bugs")
        non_bug_folder = os.path.join(parent_folder, "non_bugs")

    if folder_type == 'all':
        sub_folders = [
            os.path.join(bug_folder, sub_name) for sub_name in os.listdir(bug_folder)
        ] + [
            os.path.join(non_bug_folder, sub_name)
            for sub_name in os.listdir(non_bug_folder)
        ]
    elif folder_type == 'bugs':
        sub_folders = [
            os.path.join(bug_folder, sub_name) for sub_name in os.listdir(bug_folder)
        ]
    elif folder_type == 'non_bugs':
        sub_folders = [
            os.path.join(non_bug_folder, sub_name) for sub_name in os.listdir(non_bug_folder)
        ]
    else:
        raise

    ind_sub_folder_list = []
    for sub_folder in sub_folders:
        if os.path.isdir(sub_folder):
            ind = int(re.search(".*bugs/([0-9]*)", sub_folder).group(1))
            ind_sub_folder_list.append((ind, sub_folder))
            # print(sub_folder)
    ind_sub_folder_list_sorted = sorted(ind_sub_folder_list)
    subfolders = [filename for i, filename in ind_sub_folder_list_sorted]
    # print('len(subfolders)', len(subfolders))
    return subfolders

def load_data(subfolders):
    data_list = []
    is_bug_list = []

    objectives_list = []
    mask, labels, cur_info = None, None, None
    for sub_folder in subfolders:
        if os.path.isdir(sub_folder):
            pickle_filename = os.path.join(sub_folder, "cur_info.pickle")

            with open(pickle_filename, "rb") as f_in:
                cur_info = pickle.load(f_in)

                data, objectives, is_bug, mask, labels = cur_info["x"], cur_info["objectives"], int(cur_info["is_bug"]), cur_info["mask"], cur_info["labels"]
                # hack: backward compatibility that removes the port info in x
                if data.shape[0] == len(labels) + 1:
                    data = data[:-1]

                data_list.append(data)

                is_bug_list.append(is_bug)
                objectives_list.append(objectives)

    return data_list, np.array(is_bug_list), np.array(objectives_list), mask, labels, cur_info


def get_picklename(parent_folder):
    pickle_folder = parent_folder + "/bugs/"
    if not os.path.isdir(pickle_folder):
        pickle_folder = parent_folder + "/0/bugs/"
    i = 1
    while i < len(os.listdir(pickle_folder)):
        if os.path.isdir(pickle_folder + str(i)):
            pickle_folder = pickle_folder + str(i) + "/cur_info.pickle"
            break
        i += 1
    return pickle_folder


def set_general_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

def rand_real(rng, low, high):
    return rng.random() * (high - low) + low
# ---------------- Misc -------------------



# ---------------- Uniqueness -------------------
def is_distinct_vectorized(cur_X, prev_X, mask, xl, xu, p, c, th, verbose=True):
    if len(cur_X) == 0:
        return []
    cur_X = np.array(cur_X)
    prev_X = np.array(prev_X)
    eps = 1e-10
    remaining_inds = np.arange(cur_X.shape[0])

    mask = np.array(mask)
    xl = np.array(xl)
    xu = np.array(xu)

    n = len(mask)

    variant_fields = (xu - xl) > eps
    variant_fields_num = np.sum(variant_fields)
    th_num = np.max([np.round(th * variant_fields_num), 1])

    mask = mask[variant_fields]
    int_inds = mask == "int"
    real_inds = mask == "real"
    xl = xl[variant_fields]
    xu = xu[variant_fields]
    xl = np.concatenate([np.zeros(np.sum(int_inds)), xl[real_inds]])
    xu = np.concatenate([0.99*np.ones(np.sum(int_inds)), xu[real_inds]])

    # hack: backward compatibility with previous run data
    # if cur_X.shape[1] == n-1:
    #     cur_X = np.concatenate([cur_X, np.zeros((cur_X.shape[0], 1))], axis=1)

    cur_X = cur_X[:, variant_fields]
    cur_X = np.concatenate([cur_X[:, int_inds], cur_X[:, real_inds]], axis=1) / (np.abs(xu - xl) + eps)

    if len(prev_X) > 0:
        prev_X = prev_X[:, variant_fields]
        prev_X = np.concatenate([prev_X[:, int_inds], prev_X[:, real_inds]], axis=1) / (np.abs(xu - xl) + eps)

        diff_raw = np.abs(np.expand_dims(cur_X, axis=1) - np.expand_dims(prev_X, axis=0))
        diff = np.ones(diff_raw.shape) * (diff_raw > c)
        diff_norm = np.linalg.norm(diff, p, axis=2)
        equal = diff_norm < th_num
        remaining_inds = np.mean(equal, axis=1) == 0
        remaining_inds = np.arange(cur_X.shape[0])[remaining_inds]

        # print('remaining_inds', remaining_inds, np.arange(cur_X.shape[0])[remaining_inds], cur_X[np.arange(cur_X.shape[0])[remaining_inds]])
        if verbose:
            print('prev X filtering:',cur_X.shape[0], '->', len(remaining_inds))

    if len(remaining_inds) == 0:
        return []

    cur_X_remaining = cur_X[remaining_inds]
    print('len(cur_X_remaining)', len(cur_X_remaining))
    unique_inds = []
    for i in range(len(cur_X_remaining)-1):
        diff_raw = np.abs(np.expand_dims(cur_X_remaining[i], axis=0) - cur_X_remaining[i+1:])
        diff = np.ones(diff_raw.shape) * (diff_raw > c)
        diff_norm = np.linalg.norm(diff, p, axis=1)
        equal = diff_norm < th_num
        if np.mean(equal) == 0:
            unique_inds.append(i)

    unique_inds.append(len(cur_X_remaining)-1)

    if verbose:
        print('cur X filtering:',cur_X_remaining.shape[0], '->', len(unique_inds))

    if len(unique_inds) == 0:
        return []
    remaining_inds = remaining_inds[np.array(unique_inds)]


    return remaining_inds

def eliminate_repetitive_vectorized(cur_X, mask, xl, xu, p, c, th, verbose=True):
    cur_X = np.array(cur_X)
    eps = 1e-8
    verbose = False
    remaining_inds = np.arange(cur_X.shape[0])
    if len(cur_X) == 0:
        return remaining_inds
    else:
        mask = np.array(mask)
        xl = np.array(xl)
        xu = np.array(xu)

        variant_fields = (xu - xl) > eps
        variant_fields_num = np.sum(variant_fields)
        th_num = np.max([np.round(th * variant_fields_num), 1])

        mask = mask[variant_fields]
        xl = xl[variant_fields]
        xu = xu[variant_fields]

        cur_X = cur_X[:, variant_fields]

        int_inds = mask == "int"
        real_inds = mask == "real"

        xl = np.concatenate([np.zeros(np.sum(int_inds)), xl[real_inds]])
        xu = np.concatenate([0.99*np.ones(np.sum(int_inds)), xu[real_inds]])

        cur_X = np.concatenate([cur_X[:, int_inds], cur_X[:, real_inds]], axis=1) / (np.abs(xu - xl) + eps)


        unique_inds = []
        for i in range(len(cur_X)-1):
            diff_raw = np.abs(np.expand_dims(cur_X[i], axis=0) - cur_X[i+1:])
            diff = np.ones(diff_raw.shape) * (diff_raw > c)
            diff_norm = np.linalg.norm(diff, p, axis=1)
            equal = diff_norm < th_num
            if np.mean(equal) == 0:
                unique_inds.append(i)

        if len(unique_inds) == 0:
            return []
        remaining_inds = np.array(unique_inds)
        if verbose:
            print('cur X filtering:',cur_X.shape[0], '->', len(remaining_inds))

        return remaining_inds
# ---------------- Uniqueness -------------------

from sklearn.preprocessing import MinMaxScaler

# ---------------- Bug, Objective -------------------
def get_F(current_objectives, all_objectives, objective_weights, use_single_objective, standardize=False, normalize=False):
    # standardize current objectives using all objectives so far
    all_objectives = np.stack(all_objectives)
    current_objectives = np.stack(current_objectives).astype(np.float64)

    # standardize objectives
    if standardize:
        standardizer = StandardScaler()
        standardizer.fit(all_objectives)
        all_objectives_std = standardizer.transform(all_objectives)
        current_objectives_std = standardizer.transform(current_objectives)
    else:
        all_objectives_std = all_objectives
        current_objectives_std = current_objectives

    # normalize objectives
    if normalize:
        normalizer = MinMaxScaler()
        normalizer.fit(all_objectives_std)
        current_objectives_norm = normalizer.transform(current_objectives_std)
    else:
        current_objectives_norm = current_objectives_std

    print('current_objectives')
    print(current_objectives)
    print('current_objectives_norm')
    print(current_objectives_norm)

    current_Fs = current_objectives_norm * objective_weights

    if use_single_objective:
        current_F = np.expand_dims(np.sum(current_Fs, axis=1), axis=1)
    else:
        current_F = np.row_stack(current_Fs)
    return current_F
# ---------------- Bug, Objective -------------------



# ---------------- NN -------------------
# dependent on description labels
def encode_fields(x, labels, labels_to_encode, keywords_dict):

    x = np.array(x).astype(np.float)

    encode_fields = []
    inds_to_encode = []
    for label in labels_to_encode:
        for k, v in keywords_dict.items():
            if k in label:
                ind = labels.index(label)
                inds_to_encode.append(ind)

                encode_fields.append(v)
                break
    inds_non_encode = list(set(range(x.shape[1])) - set(inds_to_encode))

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

    embed_dims = int(np.sum(encode_fields))
    embed_fields_num = len(encode_fields)
    data_for_fit_encode = np.zeros((embed_dims, embed_fields_num))
    counter = 0
    for i, encode_field in enumerate(encode_fields):
        for j in range(encode_field):
            data_for_fit_encode[counter, i] = j
            counter += 1
    enc.fit(data_for_fit_encode)

    embed = np.array(x[:, inds_to_encode].astype(np.int))
    embed = enc.transform(embed)

    x = np.concatenate([embed, x[:, inds_non_encode]], axis=1).astype(np.float)

    return x, enc, inds_to_encode, inds_non_encode, encode_fields

# dependent on description labels
def get_labels_to_encode(labels, keywords_for_encode):
    labels_to_encode = []
    for label in labels:
        for keyword in keywords_for_encode:
            if keyword in label:
                labels_to_encode.append(label)
    return labels_to_encode

def max_one_hot_op(images, encode_fields):
    m = np.sum(encode_fields)
    one_hotezed_images_embed = np.zeros([images.shape[0], m])
    s = 0
    for field_len in encode_fields:
        max_inds = np.argmax(images[:, s : s + field_len], axis=1)
        one_hotezed_images_embed[np.arange(images.shape[0]), s + max_inds] = 1
        s += field_len
    images[:, :m] = one_hotezed_images_embed

def customized_fit(X_train, standardize, one_hot_fields_len, partial=True):
    # print('\n'*2, 'customized_fit X_train.shape', X_train.shape, '\n'*2)
    if partial:
        standardize.fit(X_train[:, one_hot_fields_len:])
    else:
        standardize.fit(X_train)

def customized_standardize(X, standardize, m, partial=True, scale_only=False):
    # print(X[:, :m].shape, standardize.transform(X[:, m:]).shape)
    if partial:
        if scale_only:
            res_non_encode = X[:, m:] * standardize.scale_
        else:
            res_non_encode = standardize.transform(X[:, m:])
        res = np.concatenate([X[:, :m], standardize.transform(X[:, m:])], axis=1)
    else:
        if scale_only:
            res = X * standardize.scale_
        else:
            res = standardize.transform(X)
    return res

def customized_inverse_standardize(X, standardize, m, partial=True, scale_only=False):
    if partial:
        if scale_only:
            res_non_encode = X[:, m:] * standardize.scale_
        else:
            res_non_encode = standardize.inverse_transform(X[:, m:])
        res = np.concatenate([X[:, :m], res_non_encode], axis=1)
    else:
        if scale_only:
            res = X * standardize.scale_
        else:
            res = standardize.inverse_transform(X)
    return res

def decode_fields(x, enc, inds_to_encode, inds_non_encode, encode_fields, adv=False):
    n = x.shape[0]
    m = len(inds_to_encode) + len(inds_non_encode)
    embed_dims = np.sum(encode_fields)

    embed = x[:, :embed_dims]
    kept = x[:, embed_dims:]

    if adv:
        one_hot_embed = np.zeros(embed.shape)
        s = 0
        for field_len in encode_fields:
            max_inds = np.argmax(x[:, s : s + field_len], axis=1)
            one_hot_embed[np.arange(x.shape[0]), s + max_inds] = 1
            s += field_len
        embed = one_hot_embed

    x_encoded = enc.inverse_transform(embed)
    # print('encode_fields', encode_fields)
    # print('embed', embed[0], x_encoded[0])
    x_decoded = np.zeros([n, m])
    x_decoded[:, inds_non_encode] = kept
    x_decoded[:, inds_to_encode] = x_encoded

    return x_decoded

def remove_fields_not_changing(x, embed_dims=0, xl=[], xu=[]):
    eps = 1e-8
    if len(xl) > 0:
        cond = xu - xl > eps
    else:
        cond = np.std(x, axis=0) > eps
    kept_fields = np.where(cond)[0]
    if embed_dims > 0:
        kept_fields = list(set(kept_fields).union(set(range(embed_dims))))

    removed_fields = list(set(range(x.shape[1])) - set(kept_fields))
    x_removed = x[:, removed_fields]
    x = x[:, kept_fields]
    return x, x_removed, kept_fields, removed_fields

def recover_fields_not_changing(x, x_removed, kept_fields, removed_fields):
    n = x.shape[0]
    m = len(kept_fields) + len(removed_fields)

    # this is True usually when adv is used
    if x_removed.shape[0] != n:
        x_removed = np.array([x_removed[0] for _ in range(n)])
    x_recovered = np.zeros([n, m])
    x_recovered[:, kept_fields] = x
    x_recovered[:, removed_fields] = x_removed

    return x_recovered

def process_X(
    initial_X,
    labels,
    xl_ori,
    xu_ori,
    cutoff,
    cutoff_end,
    partial,
    unique_bugs_len,
    keywords_dict,
    standardize_prev=None,
):
    keywords_for_encode = list(keywords_dict.keys())

    labels_to_encode = get_labels_to_encode(labels, keywords_for_encode)
    X, enc, inds_to_encode, inds_non_encode, encoded_fields = encode_fields(
        initial_X, labels, labels_to_encode, keywords_dict
    )
    one_hot_fields_len = np.sum(encoded_fields)

    xl, xu = encode_bounds(
        xl_ori, xu_ori, inds_to_encode, inds_non_encode, encoded_fields
    )

    labels_non_encode = np.array(labels)[inds_non_encode]
    # print(np.array(X).shape)
    X, X_removed, kept_fields, removed_fields = remove_fields_not_changing(
        X, one_hot_fields_len, xl=xl, xu=xu
    )
    # print(np.array(X).shape)

    param_for_recover_and_decode = (
        X_removed,
        kept_fields,
        removed_fields,
        enc,
        inds_to_encode,
        inds_non_encode,
        encoded_fields,
        xl_ori,
        xu_ori,
        unique_bugs_len,
    )

    xl = xl[kept_fields]
    xu = xu[kept_fields]

    kept_fields_non_encode = kept_fields - one_hot_fields_len
    kept_fields_non_encode = kept_fields_non_encode[kept_fields_non_encode >= 0]
    labels_used = labels_non_encode[kept_fields_non_encode]

    X_train, X_test = X[:cutoff], X[cutoff:cutoff_end]
    # print('X_train.shape, X_test.shape', X_train.shape, X_test.shape, one_hot_fields_len)
    if standardize_prev:
        standardize = standardize_prev
    else:
        standardize = StandardScaler()
        customized_fit(X_train, standardize, one_hot_fields_len, partial)
    X_train = customized_standardize(X_train, standardize, one_hot_fields_len, partial)
    if len(X_test) > 0:
        X_test = customized_standardize(X_test, standardize, one_hot_fields_len, partial)
    xl = customized_standardize(
        np.array([xl]), standardize, one_hot_fields_len, partial
    )[0]
    xu = customized_standardize(
        np.array([xu]), standardize, one_hot_fields_len, partial
    )[0]

    return (
        X_train,
        X_test,
        xl,
        xu,
        labels_used,
        standardize,
        one_hot_fields_len,
        param_for_recover_and_decode,
    )


def inverse_process_X(
    initial_test_x_adv_list,
    standardize,
    one_hot_fields_len,
    partial,
    X_removed,
    kept_fields,
    removed_fields,
    enc,
    inds_to_encode,
    inds_non_encode,
    encoded_fields,
):
    test_x_adv_list = customized_inverse_standardize(
        initial_test_x_adv_list, standardize, one_hot_fields_len, partial
    )
    X = recover_fields_not_changing(
        test_x_adv_list, X_removed, kept_fields, removed_fields
    )
    X_final_test = decode_fields(
        X, enc, inds_to_encode, inds_non_encode, encoded_fields, adv=True
    )
    return X_final_test
# ---------------- NN -------------------



# ---------------- ADV -------------------
def if_violate_constraints_vectorized(X, customized_constraints, labels, ego_start_position=None, verbose=False):
    labels_to_id = {label: i for i, label in enumerate(labels)}

    keywords = ["coefficients", "labels", "value"]
    extra_keywords = ["power"]

    if_violate = False
    violated_constraints = []
    involved_labels = set()

    X = np.array(X)
    remaining_inds = np.arange(X.shape[0])

    for i, constraint in enumerate(customized_constraints):
        for k in keywords:
            assert k in constraint
        assert len(constraint["coefficients"]) == len(constraint["labels"])

        ids = np.array([labels_to_id[label] for label in constraint["labels"]])


        # x_ids = [x[id] for id in ids]
        if "powers" in constraint:
            powers = np.array(constraint["powers"])
        else:
            powers = np.array([1 for _ in range(len(ids))])

        coeff = np.array(constraint["coefficients"])

        if_violate_current = (
            np.sum(coeff * np.power(X[remaining_inds[:, None], ids], powers), axis=1) > constraint["value"]
        )
        remaining_inds = remaining_inds[if_violate_current==0]

    # beta: eliminate NPC vehicles having generation collision with the ego car
    # TBD: consider customized_center_transforms, customizable NPC vehicle size
    # also only consider OP for now
    print('remaining_inds before', len(remaining_inds))
    tmp_remaining_inds = remaining_inds.copy()
    if ego_start_position:
        j = 0
        ego_x, ego_y, ego_yaw = ego_start_position
        ego_w = 0.93
        vehicle_w_j = 0.93
        ego_l = 2.35
        vehicle_l_j = 2.35
        dw = ego_w + vehicle_w_j
        dl = ego_l + vehicle_l_j
        while 'vehicle_x_'+str(j) in labels:
            remaining_inds_i = remaining_inds.copy()

            x_ind = labels.index('vehicle_x_'+str(j))
            y_ind = labels.index('vehicle_y_'+str(j))

            vehicle_x_j = X[remaining_inds_i, x_ind]
            vehicle_y_j = X[remaining_inds_i, y_ind]

            dx_rel = vehicle_x_j
            dy_rel = vehicle_y_j


            x_far_inds = remaining_inds_i[np.abs(dx_rel) > dw]
            x_close_inds = remaining_inds_i[np.abs(dx_rel) <= dw]

            y_far_inds = x_close_inds[np.abs(dy_rel[x_close_inds]) > dl]

            remaining_inds_i = np.concatenate([x_far_inds, y_far_inds])
            tmp_remaining_inds = np.intersect1d(tmp_remaining_inds, remaining_inds_i)
            j += 1
    remaining_inds = tmp_remaining_inds


    if verbose:
        print('constraints filtering', len(X), '->', len(remaining_inds))

    return remaining_inds

def rotate_via_numpy(xy, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    x, y = xy
    c, s = np.cos(radians), np.sin(radians)
    j = np.array([[c, -s], [s, c]])
    m = np.dot(j, np.array([x, y]))

    return m[0], m[1]

def if_violate_constraints(x, customized_constraints, labels, verbose=False):
    labels_to_id = {label: i for i, label in enumerate(labels)}

    keywords = ["coefficients", "labels", "value"]
    extra_keywords = ["power"]

    if_violate = False
    violated_constraints = []
    involved_labels = set()

    for i, constraint in enumerate(customized_constraints):
        for k in keywords:
            assert k in constraint
        assert len(constraint["coefficients"]) == len(constraint["labels"])

        ids = [labels_to_id[label] for label in constraint["labels"]]
        x_ids = [x[id] for id in ids]
        if "powers" in constraint:
            powers = np.array(constraint["powers"])
        else:
            powers = np.array([1 for _ in range(len(ids))])

        coeff = np.array(constraint["coefficients"])
        features = np.array(x_ids)

        if_violate_current = (
            np.sum(coeff * np.power(features, powers)) > constraint["value"]
        )
        if if_violate_current:
            if_violate = True
            violated_constraints.append(constraint)
            involved_labels = involved_labels.union(set(constraint["labels"]))
            if verbose:
                print("\n" * 1, "violate_constraints!!!!", "\n" * 1)
                print(
                    coeff,
                    features,
                    powers,
                    np.sum(coeff * np.power(features, powers)),
                    constraint["value"],
                    constraint["labels"],
                )

    return if_violate, [violated_constraints, involved_labels]

def encode_bounds(xl, xu, inds_to_encode, inds_non_encode, encode_fields):
    m1 = np.sum(encode_fields)
    m2 = len(inds_non_encode)
    m = m1 + m2

    xl_embed, xu_embed = np.zeros(m1), np.ones(m1)

    xl_new = np.concatenate([xl_embed, xl[inds_non_encode]])
    xu_new = np.concatenate([xu_embed, xu[inds_non_encode]])

    return xl_new, xu_new
# ---------------- ADV -------------------



# ---------------- NSGA2-DT -------------------
# check if x is in critical regions of the tree
def is_critical_region(x, estimator, critical_unique_leaves):
    leave_id = estimator.apply(x.reshape(1, -1))[0]
    print(leave_id, critical_unique_leaves)
    return leave_id in critical_unique_leaves

def filter_critical_regions(X, y):
    print("\n" * 20)
    print("+" * 100, "filter_critical_regions", "+" * 100)

    min_samples_split = np.max([int(0.1 * X.shape[0]), 2])
    # estimator = tree.DecisionTreeClassifier(min_samples_split=min_samples_split, min_impurity_decrease=0.01, random_state=0)
    estimator = tree.DecisionTreeClassifier(
        min_samples_split=min_samples_split,
        min_impurity_decrease=0.0001,
        random_state=0,
    )
    print(X.shape, y.shape)
    # print(X, y)
    estimator = estimator.fit(X, y)

    leave_ids = estimator.apply(X)
    print("leave_ids", leave_ids)

    unique_leave_ids = np.unique(leave_ids)
    unique_leaves_bug_num = np.zeros(unique_leave_ids.shape[0])
    unique_leaves_normal_num = np.zeros(unique_leave_ids.shape[0])

    for j, unique_leave_id in enumerate(unique_leave_ids):
        for i, leave_id in enumerate(leave_ids):
            if leave_id == unique_leave_id:
                if y[i] == 0:
                    unique_leaves_normal_num[j] += 1
                else:
                    unique_leaves_bug_num[j] += 1

    for i, unique_leave_i in enumerate(unique_leave_ids):
        print(
            "unique_leaves",
            unique_leave_i,
            unique_leaves_bug_num[i],
            unique_leaves_normal_num[i],
        )

    critical_unique_leaves = unique_leave_ids[
        unique_leaves_bug_num >= unique_leaves_normal_num
    ]

    print("critical_unique_leaves", critical_unique_leaves)

    inds = np.array([leave_id in critical_unique_leaves for leave_id in leave_ids])
    print("\n" * 20)

    return estimator, inds, critical_unique_leaves
# ---------------- NSGA2-DT -------------------



# ---------------- NSGA2-SM -------------------
def pretrain_regression_nets(initial_X, initial_objectives_list, objective_weights, xl_ori, xu_ori, labels, customized_constraints, cutoff, cutoff_end, keywords_dict, choose_weight_inds):

    # we are not using it so set it to 0 for placeholding
    unique_bugs_len = 0
    partial = True

    print(np.array(initial_X).shape, cutoff, cutoff_end)
    (
        X_train,
        X_test,
        xl,
        xu,
        labels_used,
        standardize,
        one_hot_fields_len,
        param_for_recover_and_decode,
    ) = process_X(
        initial_X, labels, xl_ori, xu_ori, cutoff, cutoff_end, partial, unique_bugs_len, keywords_dict
    )

    (
        X_removed,
        kept_fields,
        removed_fields,
        enc,
        inds_to_encode,
        inds_non_encode,
        encoded_fields,
        _,
        _,
        unique_bugs_len,
    ) = param_for_recover_and_decode

    weight_inds = choose_weight_inds(objective_weights)


    from pgd_attack import train_regression_net
    chosen_weights = objective_weights[weight_inds]
    clfs = []
    confs = []
    for weight_ind in weight_inds:
        y_i = np.array([obj[weight_ind] for obj in initial_objectives_list])
        y_train_i, y_test_i = y_i[:cutoff], y_i[cutoff:cutoff_end]

        clf_i, conf_i = train_regression_net(
            X_train, y_train_i, X_test, y_test_i, batch_train=200, return_test_err=True
        )
        clfs.append(clf_i)
        confs.append(conf_i)

    confs = np.array(confs)*chosen_weights
    return clfs, confs, chosen_weights, standardize
# ---------------- NSGA2-SM -------------------


# ---------------- AVFuzzer -------------------
def choose_farthest_offs(tmp_off_candidates_X, all_pop_run_X, pop_size):
    from sklearn.preprocessing import Normalizer
    Normalizer
    # transformer = Normalizer().fit(tmp_off_candidates_X)
    # tmp_off_candidates_X = transformer.transform(tmp_off_candidates_X)
    # all_pop_run_X = transformer.transform(all_pop_run_X)

    # mean = np.mean(tmp_off_candidates_X, axis=0)
    # std = np.std(tmp_off_candidates_X, axis=0)
    # tmp_off_candidates_X = (tmp_off_candidates_X-mean)/std
    # all_pop_run_X = (all_pop_run_X-mean)/std

    dis = tmp_off_candidates_X[:, np.newaxis,:] - all_pop_run_X
    # print('\n'*5, 'choose_farthest_offs')
    # print('tmp_off_candidates_X', tmp_off_candidates_X)
    # print('all_pop_run_X', all_pop_run_X)
    # print('dis', dis)
    # print('\n'*5)
    dis_sum = np.mean(np.mean(np.abs(dis), axis=2), axis=1)
    chosen_inds = np.argsort(dis_sum)[-pop_size:]
    # with open('tmp_log.txt', 'a') as f_out:
    #     f_out.write('shapes: '+str(np.shape(tmp_off_candidates_X[:, np.newaxis,:]))+','+str(np.shape(all_pop_run_X))+str(np.shape(dis))+str(np.shape(dis_sum))+str(dis_sum)+'\n\n'+str(dis_sum[chosen_inds])+'\n')
    return chosen_inds
# ---------------- AVFuzzer -------------------


# ---------------- acquisition related -------------------
# TBD: greedily add point
def calculate_rep_d(clf, X_train, X_test):
    X_train_embed = clf.extract_embed(X_train)
    X_test_embed = clf.extract_embed(X_test)
    X_combined_embed = np.concatenate([X_train_embed, X_test_embed])

    d_list = []
    for x_test_embed in X_test_embed:
        d = np.linalg.norm(X_combined_embed - x_test_embed, axis=1)
        # sorted_d = np.sort(d)
        # d_list.append(sorted_d[1])
        d_list.append(d)
    return np.array(d_list)

def select_batch_max_d_greedy(d_list, train_test_cutoff, batch_size):
    consider_inds = np.arange(train_test_cutoff)
    remaining_inds = np.arange(len(d_list))
    chosen_inds = []

    print('d_list.shape', d_list.shape)
    print('remaining_inds.shape', remaining_inds.shape)
    print('consider_inds.shape', consider_inds.shape)
    for i in range(batch_size):
        # print(i)
        # print('d_list[np.ix_(remaining_inds, consider_inds)].shape', d_list[np.ix_(remaining_inds, consider_inds)].shape)
        min_d_list = np.min(d_list[np.ix_(remaining_inds, consider_inds)], axis=1)
        # print('min_d_list', min_d_list.shape, min_d_list)
        remaining_inds_top_ind = np.argmax(min_d_list)
        chosen_ind = remaining_inds[remaining_inds_top_ind]

        # print('chosen_ind', chosen_ind)
        consider_inds = np.append(consider_inds, chosen_ind)
        # print('remaining_inds before', remaining_inds)
        # print('remaining_inds_top_ind', remaining_inds_top_ind)
        remaining_inds = np.delete(remaining_inds, remaining_inds_top_ind)
        # print('remaining_inds after', remaining_inds)
        chosen_inds.append(chosen_ind)
    return chosen_inds
# ---------------- acquisition related -------------------


def get_job_results(tmp_run_info_list, x_sublist, objectives_sublist_non_traj, trajectory_vector_sublist, x_list, objectives_list, trajectory_vector_list, traj_dist_metric=None):

    job_results = objectives_sublist_non_traj

    x_list.extend(x_sublist)
    objectives_list.extend(job_results)
    trajectory_vector_list.extend(trajectory_vector_sublist)


    return job_results, x_list, objectives_list, trajectory_vector_list


if __name__ == "__main__":
    print('ok')
