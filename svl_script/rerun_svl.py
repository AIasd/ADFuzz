import sys
import os
sys.path.append('pymoo')
carla_root = '../carla_0994_no_rss'
sys.path.append(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append(carla_root+'/PythonAPI/carla')
sys.path.append(carla_root+'/PythonAPI')

sys.path.append('leaderboard')
sys.path.append('leaderboard/team_code')
sys.path.append('scenario_runner')
sys.path.append('carla_project')
sys.path.append('carla_project/src')

sys.path.append('fuzzing_utils')
sys.path.append('carla_specific_utils')
# os.system('export PYTHONPATH=/home/zhongzzy9/anaconda3/envs/carla99/bin/python')

sys.path.append('..')

import pickle
from datetime import datetime
from customized_utils import make_hierarchical_dir
from svl_specific import run_svl_simulation

parent_folder = 'run_results_svl/nsga2-un/BorregasAve_forward/default/apollo_6_with_signal/0.25_500_0.1_0.5'
parent_folder = os.path.join(parent_folder, 'bugs')

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

            fuzzing_arguments.root_folder = 'rerun_svl'
            fuzzing_arguments.parent_folder = make_hierarchical_dir([fuzzing_arguments.root_folder, fuzzing_arguments.algorithm_name, fuzzing_arguments.route_type, fuzzing_arguments.scenario_type, fuzzing_arguments.ego_car_model, dt_time_str])
            fuzzing_arguments.mean_objectives_across_generations_path = os.path.join(parent_folder, 'mean_objectives_across_generations.txt')

            objectives, run_info = run_svl_simulation(x, fuzzing_content, fuzzing_arguments, sim_specific_arguments, dt_arguments, launch_server, counter, port)

            sim = sim_specific_arguments.sim

            print('\n'*3)
            print("run_info['is_bug'], run_info['bug_type'], objectives", run_info['is_bug'], run_info['bug_type'], objectives)
            print('\n'*3)
