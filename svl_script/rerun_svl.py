import sys
import os
sys.path.append('pymoo')
sys.path.append('fuzzing_utils')

sys.path.append('..')
carla_lbc_root = 'carla_lbc'
sys.path.append(carla_lbc_root)
sys.path.append(carla_lbc_root+'/leaderboard')
sys.path.append(carla_lbc_root+'/leaderboard/team_code')
sys.path.append(carla_lbc_root+'/scenario_runner')
sys.path.append(carla_lbc_root+'/carla_project')
sys.path.append(carla_lbc_root+'/carla_project/src')
sys.path.append(carla_lbc_root+'/carla_specific_utils')

sys.path.append('.')

import pickle
from datetime import datetime
from customized_utils import make_hierarchical_dir
from svl_specific import run_svl_simulation

parent_folder = 'svl_script/run_results_svl/random/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_11_25_13_41_25,10_10_none_100_coeff_0.0_0.1_0.5_only_unique_1/rerun'

now = datetime.now()
dt_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

sim = None
for subfolder in os.listdir(parent_folder):
    cur_folder = os.path.join(parent_folder, subfolder)
    pickle_path = os.path.join(cur_folder, 'cur_info.pickle')
    print('pickle_path', pickle_path)
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f_in:
            d = pickle.load(f_in)

            x = d['x']
            fuzzing_content = d['fuzzing_content']
            fuzzing_arguments = d['fuzzing_arguments']
            sim_specific_arguments = d['sim_specific_arguments']
            dt_arguments = d['dt_arguments']

            sim_specific_arguments.sim = sim

            if 'counter' in d:
                counter = d['counter']
            else:
                counter = int(subfolder)
            launch_server = True
            port = 2003

            fuzzing_arguments.root_folder = 'svl_script/rerun_svl'
            fuzzing_arguments.parent_folder = make_hierarchical_dir([fuzzing_arguments.root_folder, fuzzing_arguments.algorithm_name, fuzzing_arguments.route_type, fuzzing_arguments.scenario_type, fuzzing_arguments.ego_car_model, dt_time_str])
            fuzzing_arguments.mean_objectives_across_generations_path = os.path.join(parent_folder, 'mean_objectives_across_generations.txt')

            objectives, run_info = run_svl_simulation(x, fuzzing_content, fuzzing_arguments, sim_specific_arguments, dt_arguments, launch_server, counter, port)

            sim = sim_specific_arguments.sim

            print('\n'*3)
            print("run_info['is_bug'], run_info['bug_type'], objectives", run_info['is_bug'], run_info['bug_type'], objectives)
            print('\n'*3)
