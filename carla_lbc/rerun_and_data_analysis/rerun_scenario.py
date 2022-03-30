'''
This script is used to rerun previously run scenarios during the fuzzing process
'''
import sys
import os

sys.path.append('pymoo')
carla_root = '../carla_0994_no_rss'
sys.path.append(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append(carla_root+'/PythonAPI/carla')
sys.path.append(carla_root+'/PythonAPI')


sys.path.append('.')
sys.path.append('fuzzing_utils')
sys.path.append('carla_lbc')
sys.path.append('carla_lbc/leaderboard')
sys.path.append('carla_lbc/leaderboard/team_code')
sys.path.append('carla_lbc/scenario_runner')
sys.path.append('carla_lbc/carla_project')
sys.path.append('carla_lbc/carla_project/src')
sys.path.append('carla_lbc/carla_specific_utils')

# os.system('export PYTHONPATH=/home/zhongzzy9/anaconda3/envs/carla99/bin/python')



import random
import pickle
import atexit
import numpy as np
from datetime import datetime
import traceback
from distutils.dir_util import copy_tree
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.utils
from torchvision import models
import argparse

from carla_lbc.carla_specific_utils.carla_specific import run_carla_simulation, get_event_location_and_object_type, check_bug, get_unique_bugs, get_if_bug_list
from object_types import pedestrian_types, vehicle_types, static_types, vehicle_colors

from customized_utils import make_hierarchical_dir, exit_handler, process_X, inverse_process_X, get_sorted_subfolders, load_data, get_picklename


parser = argparse.ArgumentParser()
parser.add_argument('-p','--port', type=int, default=2045, help='TCP port(s) to listen to')
parser.add_argument('--ego_car_model', type=str, default='', help='model to rerun chosen scenarios. If not specified, the original one will be used.')

parser.add_argument('--rerun_mode', type=str, default='all', help="need to set to one of ['all', 'train', 'test']")
parser.add_argument('--data_category', type=str, default='bugs', help="need to set to one of ['bugs', 'non_bugs']")

parser.add_argument('--parent_folder', type=str, default='', help='the parent folder that consists of fuzzing data. It should include both bugs and non_bugs folder.')

parser.add_argument('--record_every_n_step', type=int, default=5, help='how many frames to save camera images')
parser.add_argument('--is_save', type=int, default=1, help='save rerun results')
parser.add_argument('--has_display', type=int, default=1, help='display the simulation during rerun.')
parser.add_argument("--debug", type=int, default=0, help="whether using the debug mode: planned paths will be visualized.")

arguments = parser.parse_args()
port = arguments.port
# ['lbc', 'lbc_augment', 'auto_pilot']
ego_car_model = arguments.ego_car_model
# ['train', 'test']
rerun_mode = arguments.rerun_mode
# ['bugs', 'non_bugs']
data_category = arguments.data_category
parent_folder = arguments.parent_folder
record_every_n_step = arguments.record_every_n_step
is_save = arguments.is_save
debug = arguments.debug

assert os.path.isdir(parent_folder), parent_folder+' does not exist locally'


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
os.environ['HAS_DISPLAY'] = str(arguments.has_display)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def rerun_simulation(pickle_filename, is_save, rerun_save_folder, ind, sub_folder_name, scenario_file, ego_car_model='', x=[], record_every_n_step=10):
    is_bug = False

    # parameters preparation
    if ind == 0:
        launch_server = True
    else:
        launch_server = False
    counter = ind

    with open(pickle_filename, 'rb') as f_in:
        pf = pickle.load(f_in)

        x = pf['x']
        fuzzing_content = pf['fuzzing_content']
        fuzzing_arguments = pf['fuzzing_arguments']
        sim_specific_arguments = pf['sim_specific_arguments']
        dt_arguments = pf['dt_arguments']


        route_type = pf['route_type']
        route_str = pf['route_str']

        if not ego_car_model:
            ego_car_model = pf['ego_car_model']

        mask = pf['mask']
        labels = pf['labels']

        tmp_save_path = pf['tmp_save_path']

        fuzzing_arguments.record_every_n_step = record_every_n_step
        fuzzing_arguments.ego_car_model = ego_car_model
        fuzzing_arguments.debug = debug

    folder = '_'.join([route_type, route_str, ego_car_model])


    parent_folder = make_hierarchical_dir([rerun_save_folder, folder])
    fuzzing_arguments.parent_folder = parent_folder
    fuzzing_arguments.mean_objectives_across_generations_path = os.path.join(parent_folder, 'mean_objectives_across_generations.txt')

    # TBD: temporary fix to be compatible with earlier data
    fuzzing_arguments.terminate_on_collision = True

    objectives, run_info = run_carla_simulation(x, fuzzing_content, fuzzing_arguments, sim_specific_arguments, dt_arguments, launch_server, counter, port)

    is_bug = int(check_bug(objectives))

    # save data
    if is_save:
        print('sub_folder_name', sub_folder_name)
        if is_bug:
            rerun_folder = make_hierarchical_dir([parent_folder, 'rerun_bugs'])
            print('\n'*3, 'rerun also causes a bug!!!', '\n'*3)
        else:
            rerun_folder = make_hierarchical_dir([parent_folder, 'rerun_non_bugs'])

        try:
            new_path = os.path.join(rerun_folder, sub_folder_name)
            copy_tree(tmp_save_path, new_path)
        except:
            print('fail to copy from', tmp_save_path)
            traceback.print_exc()
            raise

        cur_info = {'x':x, 'objectives':objectives, 'labels':run_info['labels'], 'mask':run_info['mask'], 'is_bug':is_bug, 'fuzzing_content': run_info['fuzzing_content'], 'fuzzing_arguments': run_info['fuzzing_arguments'], 'sim_specific_arguments': run_info['sim_specific_arguments'], 'dt_arguments': run_info['dt_arguments'], 'route_type': run_info['route_type'], 'route_str': run_info['route_str']}

        with open(new_path+'/'+'cur_info.pickle', 'wb') as f_out:
            pickle.dump(cur_info, f_out)


    return is_bug, objectives



def rerun_list_of_scenarios(parent_folder, rerun_save_folder, scenario_file, data_category, mode, ego_car_model, record_every_n_step=10, is_save=True):
    import re

    subfolder_names = get_sorted_subfolders(parent_folder, data_category)
    print('len(subfolder_names)', len(subfolder_names))

    mid = len(subfolder_names) // 2
    random.shuffle(subfolder_names)

    train_subfolder_names = subfolder_names[:mid]
    test_subfolder_names = subfolder_names[-mid:]

    if mode == 'train':
        chosen_subfolder_names = train_subfolder_names
    elif mode == 'test':
        chosen_subfolder_names = test_subfolder_names
    elif mode == 'all':
        chosen_subfolder_names = subfolder_names

    bug_num = 0
    objectives_avg = 0

    for ind, sub_folder in enumerate(chosen_subfolder_names):
        print('episode:', ind+1, '/', len(chosen_subfolder_names), 'bug num:', bug_num)

        sub_folder_name = re.search(".*/([0-9]*)$", sub_folder).group(1)
        print('sub_folder', sub_folder)
        print('sub_folder_name', sub_folder_name)
        if os.path.isdir(sub_folder):
            pickle_filename = os.path.join(sub_folder, 'cur_info.pickle')
            if os.path.exists(pickle_filename):
                print('pickle_filename', pickle_filename)
                is_bug, objectives = rerun_simulation(pickle_filename, is_save, rerun_save_folder, ind, sub_folder_name, scenario_file, ego_car_model=ego_car_model, record_every_n_step=record_every_n_step)


                objectives_avg += np.array(objectives)

                if is_bug:
                    bug_num += 1

    print('bug_ratio :', bug_num / len(chosen_subfolder_names))
    print('objectives_avg :', objectives_avg / len(chosen_subfolder_names))





if __name__ == '__main__':
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    scenario_folder = 'carla_lbc/scenario_files'
    if not os.path.exists(scenario_folder):
        os.mkdir(scenario_folder)
    scenario_file = scenario_folder+'/'+'current_scenario_'+time_str+'.json'

    atexit.register(exit_handler, [port])


    print('ego_car_model', ego_car_model, 'data_category', data_category, 'mode', rerun_mode)
    rerun_save_folder = make_hierarchical_dir(['carla_lbc', 'rerun', rerun_mode, time_str])

    rerun_list_of_scenarios(parent_folder, rerun_save_folder, scenario_file, data_category, rerun_mode, ego_car_model, record_every_n_step=record_every_n_step)
