import sys
sys.path.append('..')
import os
import numpy as np
import pickle
from customized_utils import make_hierarchical_dir, emptyobject, is_distinct_vectorized
from matplotlib import pyplot as plt
from object_types import pedestrian_types, car_types, large_car_types


def get_bug_traj(events_path, npc_events_path):
    fields_limit = {
    'type_num': 4,
    'instance_num': 4,
    'ego_speed': 7,
    'other_speed': 7,
    'collision_ind': 8,
    }
    bug = (0, 0, 0, 0, 0)

    if os.path.exists(events_path) and os.path.exists(npc_events_path):
        with open(events_path, 'r') as f_in:
            tokens = f_in.read().split('\n')[0].split(',')

            if tokens[0] == 'collision':
                other_agent_type, agent2_uid = tokens[1:3]
                collision_float_tokens = [float(tok) for tok in tokens[3:]]
                ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz, other_x, other_y, other_z, other_vx, other_vy, other_vz, contact_x, contact_y, contact_z = collision_float_tokens


        with open(npc_events_path, "r") as f_in:
            for line in f_in:
                tokens = line.strip('\n').split(",")
                if tokens[0] == 'npc_agents_uids':
                    agents_uids = tokens[1:]




        # type num
        if other_agent_type in pedestrian_types:
            type_num = 0
        elif other_agent_type in car_types:
            type_num = 1
        elif other_agent_type in large_car_types:
            type_num = 2
        else:
            type_num = 3

        # instance num
        instance_num = int(agents_uids.index(agent2_uid))

        # ego speed
        ego_speed = np.linalg.norm([ego_vx, ego_vy, ego_vz])
        ego_speed_ind = np.min([int(ego_speed // 3), fields_limit['ego_speed']])

        # other speed
        other_speed = np.linalg.norm([other_vx, other_vy, other_vz])
        other_speed_ind = np.min([int(other_speed // 3), fields_limit['other_speed']])

        # collision angle
        collision_angle = np.arctan2(contact_z-ego_z, contact_x-ego_x)
        collision_angle = collision_angle % 360
        collision_angle_ind = int(collision_angle // 45)



        bug = (type_num, instance_num, ego_speed_ind, other_speed_ind, collision_angle_ind)

    return bug, fields_limit



def count_bugs(folder, save_folder):
    make_hierarchical_dir(save_folder)
    save_folder = os.path.join(*save_folder)

    unique_bugs_list = []
    all_bugs_list = []
    sorted_folder = sorted(os.listdir(folder), key=lambda f:int(f))
    for subfolder in sorted_folder:
        events_path = os.path.join(folder, subfolder, 'events.txt')
        npc_events_path = os.path.join(folder, subfolder, 'npc_events.txt')

        bug, _ = get_bug_traj(events_path, npc_events_path)

        diff = True
        for _, u_bug in unique_bugs_list:
            if bug == u_bug:
                diff = False
                break
        if diff:
            unique_bugs_list.append((subfolder, bug))
        all_bugs_list.append((subfolder, bug))

    print('unique_bugs_list:', len(unique_bugs_list))
    print(unique_bugs_list)

    # objs = []
    # for f, obj in all_bugs_list:
    #     objs.append(obj)
    # objs = np.stack(objs)
    # plt.hist(objs[:, 0])
    # plt.title('type_num')
    # plt.savefig(os.path.join(save_folder, 'type_num'))
    # plt.close()
    #
    # plt.hist(objs[:, 1])
    # plt.title('instance_num')
    # plt.savefig(os.path.join(save_folder, 'instance_num'))
    # plt.close()
    #
    # plt.hist(objs[:, 2])
    # plt.title('ego_speed_ind')
    # plt.savefig(os.path.join(save_folder, 'ego_speed_ind'))
    # plt.close()
    #
    # plt.hist(objs[:, 3])
    # plt.title('other_speed_ind')
    # plt.savefig(os.path.join(save_folder, 'other_speed_ind'))
    # plt.close()
    #
    # plt.hist(objs[:, 4])
    # plt.title('collision_angle_ind')
    # plt.savefig(os.path.join(save_folder, 'collision_angle_ind'))
    # plt.close()

    return unique_bugs_list

def analyze_detection_results():
    pass

def check_fuzzing_param(path):
    with open(os.path.join(path, 'cur_info.pickle'), 'rb') as f_in:
        run_info = pickle.load(f_in)

    for i in range(len(run_info['x'])):
        print(run_info['fuzzing_content'].labels[i], run_info['x'][i])

def plot_comparison():
    random = [[3, 4, 4, 4, 5, 6, 6, 7, 7, 7]]
    nsga2_un = [[3, 4, 5, 6, 6, 7, 8, 8, 8, 9]]
    nsga2_un_div = [[2, 4, 7, 7, 7, 10, 11, 11, 11, 11]]

    r = np.arange(len(random[0]))*20
    plt.errorbar(r, np.mean(random, axis=0), yerr=np.std(random, axis=0), label='random')
    plt.errorbar(r, np.mean(nsga2_un, axis=0), yerr=np.std(nsga2_un, axis=0), label='ga_un')
    plt.errorbar(r, np.mean(nsga2_un_div, axis=0), yerr=np.std(nsga2_un_div, axis=0), label='ga_un_div')
    plt.legend()
    plt.savefig('output_count.png')


if __name__ == '__main__':
    save_folder = ['visualization', 'nsga2-un']

    # nsga2-un div
    folder = 'run_results_svl/nsga2-un/BorregasAve_forward/go_across_junction_ba/apollo_6_with_signal/2022_01_28_09_51_50,20_10_none_200_coeff_0.0_0.1_0.5_only_unique_1/bugs'

    count_bugs(folder, save_folder)

    # plot_comparison()
