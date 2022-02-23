import sys
import os
sys.path.append('.')
import numpy as np
import pickle
from customized_utils import make_hierarchical_dir, emptyobject, is_distinct_vectorized
from matplotlib import pyplot as plt
from svl_script.object_types import pedestrian_types, car_types, large_car_types


def get_bug_traj(events_path, npc_events_path):
    # instance_num might need to be increased based on specific scenarios used
    fields_limit = {
    'type_num': 4,
    'instance_num': 100,
    'ego_speed': 8,
    'other_speed': 8,
    'collision_ind': 8,
    }
    bug = (0, 0, 0, 0, 0)

    if os.path.exists(events_path) and os.path.exists(npc_events_path):
        with open(events_path, 'r') as f_in:
            tokens = f_in.read().split('\n')[0].split(',')
            ego_collision = False
            if tokens[0] == 'collision':
                other_agent_type, agent2_uid = tokens[1:3]
                collision_float_tokens = [float(tok) for tok in tokens[3:]]
                ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz, other_x, other_y, other_z, other_vx, other_vy, other_vz, contact_x, contact_y, contact_z = collision_float_tokens
                ego_collision = True

            if not ego_collision:
                return bug, fields_limit


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
            print('other_agent_type:', other_agent_type)
            type_num = 3

        # instance num
        if agent2_uid in agents_uids:
            instance_num = int(agents_uids.index(agent2_uid))+1
        else:
            instance_num = 0

        # ego speed
        ego_speed = np.linalg.norm([ego_vx, ego_vy, ego_vz])
        ego_speed_ind = np.min([int(ego_speed // 3), fields_limit['ego_speed']-1])

        # other speed
        other_speed = np.linalg.norm([other_vx, other_vy, other_vz])
        other_speed_ind = np.min([int(other_speed // 3), fields_limit['other_speed']-1])

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

def generate_data_summary(original_folder):
    bugs_folder = os.path.join(original_folder, 'bugs')
    non_bugs_folder = os.path.join(original_folder, 'non_bugs')
    X_non_zero = []
    objectives_list = []
    is_bug_list = []
    for folder in [bugs_folder, non_bugs_folder]:
        sorted_folder = sorted(os.listdir(folder), key=lambda f:int(f))
        for subfolder in sorted_folder:
            subfolder_path = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_path):
                run_info_path = os.path.join(subfolder_path, 'cur_info.pickle')
                if os.path.exists(run_info_path):
                    with open(run_info_path, 'rb') as handle:
                        run_info = pickle.load(handle)
                    x, labels, xl, xu, mask, is_bug, objectives = run_info['x'], run_info['labels'], run_info['xl'], run_info['xu'], run_info['mask'], run_info['is_bug'], run_info['objectives']
                    labels = np.array(labels)
                    mask = np.array(mask)

                    non_zero_inds = xu-xl>1e-4
                    labels_non_zero = labels[non_zero_inds]
                    xl_non_zero = xl[non_zero_inds]
                    xu_non_zero = xu[non_zero_inds]
                    mask_non_zero = mask[non_zero_inds]
                    X_non_zero.append(x[non_zero_inds])
                    objectives_list.append(objectives[0:3] + [objectives[6]])
                    is_bug_list.append(is_bug)
                    # print('is_bug, objectives', is_bug, objectives, '\n')
    X_non_zero = np.stack(X_non_zero)
    objectives_list = np.stack(objectives_list)
    objectives_label = np.array(['ego_linear_speed_at_collision', 'min_d', 'npc_collisions', 'ego_collision'])
    is_bug_list = np.array(is_bug_list)

    # X: config data for each run
    # x_types: data type of each config
    # x_labels: label of each config
    # xl: lower bound of each config
    # xu: upper bound of each config
    # objectives: objective values for each run
    # objectives_label: label of each objective
    # is_bug: if the run results in a bug
    print('labels', len(labels_non_zero), labels_non_zero)
    data_summary = {
        'X': X_non_zero,

        'x_types': mask_non_zero,
        'x_labels': labels_non_zero,
        'xl': xl_non_zero,
        'xu': xu_non_zero,

        'objectives': objectives_list,
        'objectives_label': objectives_label,

        'is_bug': is_bug_list,
    }

    with open(os.path.join(original_folder, 'data_summary.pickle'), 'wb') as f_out:
        pickle.dump(data_summary, f_out)

    # print(X_non_zero.shape, mask_non_zero.shape, labels_non_zero.shape, xl_non_zero.shape, xu_non_zero.shape, objectives_list.shape, objectives_label.shape, is_bug_list.shape)
    # print('data_summary', data_summary)

    # one-hot encode int fields of X and x_labels
    p = 0
    n = data_summary['X'].shape[0]
    m = data_summary['X'].shape[1]
    new_X = []
    new_x_labels = []
    for i in range(m):
        x_type_i = data_summary['x_types'][i]
        x_label_i = data_summary['x_labels'][i]
        xl_i = data_summary['xl'][i]
        xu_i = data_summary['xu'][i]
        X = data_summary['X']

        if x_type_i == 'int':
            num_fileds = xu_i - xl_i + 1
            X_i_one_hot = np.zeros((n, num_fileds))

            X_i_one_hot[:, X[:, i].astype('int')] = 1

            if p < i:
                new_X.append(X[:, p:i])
            new_X.append(X_i_one_hot)
            p = i+1

            for j in range(num_fileds):
                new_x_labels.append(x_label_i+'_'+str(j))
            print(len(new_x_labels))
        else:
            new_x_labels.append(x_label_i)
    if p < m:
        new_X.append(X[:, p:m])

    new_X = np.concatenate(new_X+[objectives_list]+[np.expand_dims(is_bug_list, axis=1)], axis=1)
    new_x_labels = np.array(new_x_labels+objectives_label.tolist()+['is_bug'])
    print(new_X.shape, new_x_labels.shape)

    import pandas as pd
    df = pd.DataFrame(data=new_X, columns=new_x_labels)
    df.to_csv(os.path.join(original_folder, 'apollo.csv'), index=False)



if __name__ == '__main__':
    # save_folder = ['visualization', 'nsga2-un']
    # nsga2-un div
    folder = 'svl_script/run_results_svl/nsga2-un/BorregasAve_forward/go_across_junction_ba/apollo_6_with_signal/2022_01_28_09_51_50,20_10_none_200_coeff_0.0_0.1_0.5_only_unique_1/bugs'
    # count_bugs(folder, save_folder)

    generate_data_summary(folder)
