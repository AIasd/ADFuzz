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

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE
from customized_utils import filter_critical_regions, get_sorted_subfolders, load_data, get_picklename, is_distinct_vectorized
from matplotlib.lines import Line2D

from carla_lbc.carla_specific_utils.carla_specific import get_event_location_and_object_type


from svl_script.svl_specific import check_bug
default_objectives = np.array([0., 20., 1., 7., 7., 0., 0., 0., 0., 0.])


from carla_lbc.carla_specific_utils import setup_labels_and_bounds


def draw_hv(bug_res_path, save_folder):
    with open(bug_res_path, 'rb') as f_in:
        res = pickle.load(f_in)
    hv = res['hv']
    n_evals = res['n_evals'].tolist()

    # hv = [0] + hv
    # n_evals = [0] + n_evals


    # visualze the convergence curve
    plt.plot(n_evals, hv, '-o')
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.savefig(os.path.join(save_folder, 'hv_across_generations'))
    plt.close()



def draw_performance(bug_res_path, save_folder):
    with open(bug_res_path, 'rb') as f_in:
        res = pickle.load(f_in)

    time_bug_num_list = res['time_bug_num_list']

    t_list = []
    n_list = []
    for t, n in time_bug_num_list:
        t_list.append(t)
        n_list.append(n)
    print(t_list)
    print(n_list)
    plt.plot(t_list, n_list, '-o')
    plt.title("Time VS Number of Bugs")
    plt.xlabel("Time")
    plt.ylabel("Number of Bugs")
    plt.savefig(os.path.join(save_folder, 'bug_num_across_time'))
    plt.close()


def analyze_causes(folder, save_folder, total_num, pop_size):



    avg_f = [0 for _ in range(int(total_num // pop_size))]

    causes_list = []
    counter = 0
    for sub_folder_name in os.listdir(folder):
        sub_folder = os.path.join(folder, sub_folder_name)
        if os.path.isdir(sub_folder):
            for filename in os.listdir(sub_folder):
                if filename.endswith(".npz"):
                    filepath = os.path.join(sub_folder, filename)
                    bug = np.load(filepath, allow_pickle=True)['bug'][()]

                    ego_linear_speed = float(bug['ego_linear_speed'])
                    causes_list.append((sub_folder_name, ego_linear_speed, bug['offroad_dist'], bug['is_wrong_lane'], bug['is_run_red_light'], bug['status'], bug['loc'], bug['object_type']))

                    ind = int(int(sub_folder_name) // pop_size)
                    avg_f[ind] += (ego_linear_speed / pop_size)*-1

    causes_list = sorted(causes_list, key=lambda t: int(t[0]))
    for c in causes_list:
        print(c)
    print(avg_f)

    plt.plot(np.arange(len(avg_f)), avg_f)
    plt.title("average objective value across generations")
    plt.xlabel("Generations")
    plt.ylabel("average objective value")
    plt.savefig(os.path.join(save_folder, 'f_across_generations'))

    plt.close()

def show_gen_f(bug_res_path):
    with open(bug_res_path, 'rb') as f_in:
        res = pickle.load(f_in)

    val = res['val']
    plt.plot(np.arange(len(val)), val)
    plt.show()

def plot_each_bug_num_and_objective_num_over_generations(generation_data_paths):
    # X=X, y=y, F=F, objectives=objectives, time=time_list, bug_num=bug_num_list, labels=labels, hv=hv
    pop_size = 100
    data_list = []
    for generation_data_path in generation_data_paths:
        data = []
        with open(generation_data_path[1], 'r') as f_in:
            for line in f_in:
                tokens = line.split(',')
                if len(tokens) == 2:
                    pass
                else:
                    tokens = [float(x.strip('\n')) for x in line.split(',')]
                    num, has_run, time, bugs, collisions, offroad_num, wronglane_num, speed, min_d, offroad, wronglane, dev = tokens[:12]
                    out_of_road = offroad_num + wronglane_num
                    data.append(np.array([num/pop_size, has_run, time, bugs, collisions, offroad_num, wronglane_num, out_of_road, speed, min_d, offroad, wronglane, dev]))

        data = np.stack(data)
        data_list.append(data)

    labels = [generation_data_paths[i][0] for i in range(len(data_list))]
    data = np.concatenate([data_list[1], data_list[2]], axis=0)

    for i in range(len(data_list[1]), len(data_list[1])+len(data_list[2])):
        data[i] += data_list[1][-1]
    data_list.append(data)

    labels.append('collision+out-of-road')

    fig = plt.figure(figsize=(15, 9))


    plt.suptitle("values over time", fontsize=14)


    info = [(1, 3, 'Bug Numbers'), (6, 4, 'Collision Numbers'), (7, 5, 'Offroad Numbers'), (8, 6, 'Wronglane Numbers'), (9, 7, 'Out-of-road Numbers'), (11, 8, 'Collision Speed'), (12, 9, 'Min object distance'), (13, 10, 'Offroad Directed Distance'), (14, 11, 'Wronglane Directed Distance'), (15, 12, 'Max Deviation')]

    for loc, ind, ylabel in info:
        ax = fig.add_subplot(3, 5, loc)
        for i in [0, 3, 1, 2]:
            if loc < 11 or i < 3:
                label = labels[i]
                if loc >= 11:
                    y = []
                    for j in range(data_list[i].shape[0]):
                        y.append(np.mean(data_list[i][:j+1, ind]))
                else:
                    y = data_list[i][:, ind]
                ax.plot(data_list[i][:, 0], y, label=label)
        if loc == 1:
            ax.legend()
        plt.xlabel("Generations")
        plt.ylabel(ylabel)
    plt.savefig('bug_num_and_objective_num_over_generations')







# list bug types and their run numbers
def list_bug_categories_with_numbers(folder_path):
    l = []
    for sub_folder_name in os.listdir(folder_path):
        sub_folder = os.path.join(folder_path, sub_folder_name)
        if os.path.isdir(sub_folder):
            for filename in os.listdir(sub_folder):
                if filename.endswith(".npz"):
                    filepath = os.path.join(sub_folder, filename)
                    bug = np.load(filepath, allow_pickle=True)['bug'][()]
                    if bug['ego_linear_speed'] > 0:
                        cause_str = 'collision'
                    elif bug['is_offroad']:
                        cause_str = 'offroad'
                    elif bug['is_wrong_lane']:
                        cause_str = 'wronglane'
                    else:
                        cause_str = 'unknown'
                    l.append((sub_folder_name, cause_str))


    for n,s in sorted(l, key=lambda t: int(t[0])):
        print(n,s)



def unique_bug_num(all_X, all_y, mask, xl, xu, cutoff):
    if cutoff == 0:
        return 0, []
    X = all_X[:cutoff]
    y = all_y[:cutoff]

    bug_inds = np.where(y>0)
    bugs = X[bug_inds]


    p = 0
    c = 0.15
    th = int(len(mask)*0.5)

    # TBD: count different bugs separately
    filtered_bugs, unique_bug_inds = get_distinct_data_points(bugs, mask, xl, xu, p, c, th)


    print(cutoff, len(filtered_bugs), len(bugs))
    return len(filtered_bugs), np.array(unique_bug_inds), bug_inds

# plot two tsne plots for bugs VS normal and data points across generations
def apply_tsne(path, n_gen, pop_size):
    d = np.load(path, allow_pickle=True)
    X = d['X']
    y = d['y']
    mask = d['mask']
    xl = d['xl']
    xu = d['xu']


    cutoff = n_gen * pop_size
    _, unique_bug_inds, bug_inds = unique_bug_num(X, y, mask, xl, xu, cutoff)

    y[bug_inds] = 1
    y[unique_bug_inds] = 2


    generations = []
    for i in range(n_gen):
        generations += [i for _ in range(pop_size)]


    X_embedded = TSNE(n_components=2).fit_transform(X)

    fig = plt.figure(figsize=(9, 9))
    plt.suptitle("tSNE of bugs and unique bugs", fontsize=14)
    ax = fig.add_subplot(111)
    scatter_bug = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=y, cmap=plt.cm.rainbow)
    plt.title("bugs VS normal")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(handles=scatter_bug.legend_elements()[0], labels=['normal', 'bugs'])

    plt.savefig('tsne')




def get_bug_num(cutoff, X, y, mask, xl, xu, p=0, c=0.15, th=0.5):

    if cutoff == 0:
        return 0, 0, 0
    p = p
    c = c
    th = int(len(mask)*th)

    def process_specific_bug(bug_ind):
        chosen_bugs = y == bug_ind
        specific_bugs = X[chosen_bugs]
        unique_specific_bugs, specific_distinct_inds = get_distinct_data_points(specific_bugs, mask, xl, xu, p, c, th)

        return len(unique_specific_bugs)

    unique_collision_num = process_specific_bug(1)
    unique_offroad_num = process_specific_bug(2)
    unique_wronglane_num = process_specific_bug(3)
    unique_redlight_num = process_specific_bug(3)

    return unique_collision_num, unique_offroad_num, unique_wronglane_num, unique_redlight_num


def unique_bug_num_seq_partial_objectives(path_list):


    all_X_list = []
    all_y_list = []

    for i, (label, pth) in enumerate(path_list):
        d = np.load(pth, allow_pickle=True)

        xl = d['xl']
        xu = d['xu']
        mask = d['mask']
        objectives = np.stack(d['objectives'])
        df_objectives = np.array(default_objectives)

        eps = 1e-7
        diff = np.sum(objectives - df_objectives, axis=1)


        inds = np.abs(diff) > eps

        all_X = d['X'][inds]
        all_y = d['y'][inds]
        objectives = objectives[inds]

        all_X_list.append(all_X)
        all_y_list.append(all_y)

    all_X = np.concatenate(all_X_list)
    all_y = np.concatenate(all_y_list)

    collision_num, offroad_num, wronglane_num, redlight_num = get_bug_num(700, all_X, all_y, mask, xl, xu)
    print(collision_num, offroad_num, wronglane_num, redlight_num)


def analyze_objectives(path_list, filename='objectives_bug_num_over_simulations', scene_name=''):

    cutoffs = [100*i for i in range(0, 8)]
    data_list = []
    labels = []

    for i, (label, pth) in enumerate(path_list):
        d = np.load(pth, allow_pickle=True)
        labels.append(label)

        xl = d['xl']
        xu = d['xu']
        mask = d['mask']
        objectives = np.stack(d['objectives'])
        df_objectives = np.array(default_objectives)

        eps = 1e-7
        diff = np.sum(objectives - df_objectives, axis=1)


        inds = np.abs(diff) > eps

        all_X = d['X'][inds]
        all_y = d['y'][inds]
        objectives = objectives[inds]


        data = []
        for cutoff in cutoffs:
            X = all_X[:cutoff]
            y = all_y[:cutoff]
            collision_num, offroad_num, wronglane_num, redlight_num = get_bug_num(cutoff, X, y, mask, xl, xu)

            if cutoff == 1400:
                print(collision_num, offroad_num, wronglane_num, redlight_num)

            speed = np.mean(objectives[:cutoff, 0])
            min_d = np.mean(objectives[:cutoff, 1])
            offroad = np.mean(objectives[:cutoff, 2])
            wronglane = np.mean(objectives[:cutoff, 3])
            dev = np.mean(objectives[:cutoff, 4])

            bug_num = collision_num+offroad_num+wronglane_num
            out_of_road_num = offroad_num+wronglane_num
            data.append(np.array([bug_num, collision_num, offroad_num, wronglane_num, out_of_road_num, speed, min_d, offroad, wronglane, dev]))

        data = np.stack(data)
        data_list.append(data)




    fig = plt.figure(figsize=(12.5, 5))

    # fig = plt.figure(figsize=(15, 9))
    # plt.suptitle("values over simulations", fontsize=14)


    # info = [(1, 0, 'Bug Numbers'), (6, 1, 'Collision Numbers'), (7, 2, 'Offroad Numbers'), (8, 3, 'Wronglane Numbers'), (9, 4, 'Out-of-road Numbers'), (11, 5, 'Collision Speed'), (12, 6, 'Min object distance'), (13, 7, 'Offroad Directed Distance'), (14, 8, 'Wronglane Directed Distance'), (15, 9, 'Max Deviation')]

    info = [(1, 1, '# unique collision'), (2, 4, '# unique out-of-road')]

    for loc, ind, ylabel in info:
        ax = fig.add_subplot(1, 2, loc)
        for i in range(len(data_list)):
            label = labels[i]
            y = data_list[i][:, ind]
            ax.plot(cutoffs, y, label=label, linewidth=2, marker='o', markersize=10)
        if loc == 1:
            ax.legend(loc=2, prop={'size': 26}, fancybox=True, framealpha=0.2)

            # import pylab
            # fig_p = pylab.figure()
            # figlegend = pylab.figure(figsize=(3,2))
            # ax = fig_p.add_subplot(111)
            # lines = ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10))
            # figlegend.legend(lines, ('collision-', 'two'), 'center')
            # fig.show()
            # figlegend.show()
            # figlegend.savefig('legend.png')

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        ax.set_xlabel("# simulations", fontsize=26)
        ax.set_ylabel(ylabel, fontsize=26)

    fig.suptitle(scene_name, fontsize=38)

    fig.tight_layout()



    plt.savefig(filename)


def ablate_thresholds(path_list, thresholds_list, cutoff):

    p = 0

    xl = None
    xu = None
    mask = None

    for c, th in thresholds_list:
        print('(', c, th, ')')
        for i, (label, pth) in enumerate(path_list):
            print(label)
            d = np.load(pth, allow_pickle=True)

            if i == 0:
                xl = d['xl']
                xu = d['xu']
                mask = d['mask']
            objectives = np.stack(d['objectives'])
            df_objectives = np.array(default_objectives)

            eps = 1e-7
            diff = np.sum(objectives - df_objectives, axis=1)


            inds = np.abs(diff) > eps

            all_X = d['X'][inds]
            all_y = d['y'][inds]
            objectives = objectives[inds]


            X = all_X[:cutoff]
            y = all_y[:cutoff]
            collision_num, offroad_num, wronglane_num, redlight_num = get_bug_num(cutoff, X, y, mask, xl, xu, p=p, c=c, th=th)

            print(collision_num+offroad_num+wronglane_num, collision_num, offroad_num, wronglane_num)


def check_unique_bug_num(folder, path1, path2):
    d = np.load(folder+'/'+path1, allow_pickle=True)
    xl = d['xl']
    xu = d['xu']
    mask = d['mask']

    d = np.load(folder+'/'+path2, allow_pickle=True)
    all_X = d['X']
    all_y = d['y']
    cutoffs = [100*i for i in range(0, 15)]


    def subroutine(cutoff):
        if cutoff == 0:
            return 0, []
        X = all_X[:cutoff]
        y = all_y[:cutoff]

        bugs = X[y>0]


        p = 0
        c = 0.15
        th = 0.5

        filtered_bugs, inds = get_distinct_data_points(bugs, mask, xl, xu, p, c, th)
        print(cutoff, len(filtered_bugs), len(bugs))
        return len(filtered_bugs), inds


    num_of_unique_bugs = []
    for cutoff in cutoffs:
        num, inds = subroutine(cutoff)
        num_of_unique_bugs.append(num)
    print(inds)
    # print(bug_counters)
    # counter_inds = np.array(bug_counters)[inds] - 1
    # print(all_X[counter_inds[-2]])
    # print(all_X[counter_inds[-1]])

    plt.plot(cutoffs, num_of_unique_bugs, marker='o', markersize=10)
    plt.xlabel('# simulations')
    plt.ylabel('# unique violations')
    plt.savefig('num_of_unique_bugs')




def draw_hv_and_gd(path_list):
    from pymoo.factory import get_performance_indicator
    def is_pareto_efficient_dumb(costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype = bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
        return is_efficient

    for i, (label, pth) in enumerate(path_list):
        d = np.load(pth, allow_pickle=True)
        X = d['X']
        y = d['y']
        objectives = d['objectives'][:, :5] * np.array([-1, 1, 1, 1, -1])

        pareto_set = objectives[is_pareto_efficient_dumb(objectives)]
        # print(label, np.sum(is_pareto_efficient_dumb(objectives)))




        gd = get_performance_indicator("gd", pareto_set)
        hv = get_performance_indicator("hv", ref_point=np.array([0.01, 7.01, 7.01, 7.01, 0.01]))

        print(label)
        for j in range(16):
            cur_objectives = objectives[:(j+1)*100]
            print(j)
            print("GD", gd.calc(cur_objectives))
            print("hv", hv.calc(cur_objectives))




def calculate_pairwise_dist(path_list):
    xl = None
    xu = None
    mask = None
    for i, (label, pth) in enumerate(path_list):
        print(label)
        d = np.load(pth, allow_pickle=True)
        if i == 0:
            xl = d['xl']
            xu = d['xu']
            mask = d['mask']
            print(len(mask))

        p = 0
        c = 0.15
        th = 0.5

        objectives = np.stack(d['objectives'])
        df_objectives = np.array(default_objectives)

        eps = 1e-7
        diff = np.sum(objectives - df_objectives, axis=1)


        inds = np.abs(diff) > eps

        all_y = d['y'][inds][:1500]
        all_X = d['X'][inds][:1500]


        # all_X, inds = get_distinct_data_points(all_X, mask, xl, xu, p, c, th)
        # all_y = all_y[inds]

        int_inds = mask == 'int'
        real_inds = mask == 'real'
        eps = 1e-8




        def pair_dist(x_1, x_2):
            int_diff_raw = np.abs(x_1[int_inds] - x_2[int_inds])
            int_diff = np.ones(int_diff_raw.shape) * (int_diff_raw > eps)

            real_diff_raw = np.abs(x_1[real_inds] - x_2[real_inds]) / (np.abs(xu[real_inds] - xl[real_inds]) + eps)

            real_diff = np.ones(real_diff_raw.shape) * (real_diff_raw > c)

            diff = np.concatenate([int_diff, real_diff])

            diff_norm = np.linalg.norm(diff, p)
            # print(diff, diff_norm)
            return diff_norm



        dist_list = []
        for i in range(len(all_X)-1):
            for j in range(i+1, len(all_X)):
                if check_bug(objectives[i]) > 0 and check_bug(objectives[j]) > 0:
                    diff = pair_dist(all_X[i], all_X[j])
                    if diff:
                        dist_list.append(diff)


        dist = np.array(dist_list) / len(mask)
        print(np.mean(dist), np.std(dist))




def visualize_ped_over_time(path, save_filename='ped_over_time', bug_type='collision', unique_coeffs=[0.1, 0.1], range_upper_bound=5, warmup_path=None, warmup_len=None):

    def plot_arrow(ax, x_i, value_inds, color, width=0.002, head_width=0.2):
        x_0_ind, y_0_ind, yaw_0_ind, speed_0_ind = value_inds

        x, y = x_i[x_0_ind], x_i[y_0_ind]
        yaw = np.deg2rad(x_i[yaw_0_ind])
        speed = x_i[speed_0_ind]

        dx = np.cos(yaw)*speed/5
        dy = np.sin(yaw)*speed/5

        ax.arrow(x, y, dx, dy, color=color, head_width=head_width, alpha=0.5, width=width)
        ax.set_ylim(-11, 11)
        ax.set_xlim(11, -11)


    label = ''
    cutoffs = [250*i for i in range(1, range_upper_bound)]
    p = 0
    c, th = unique_coeffs

    subfolders = get_sorted_subfolders(path)
    cur_X, _, cur_objectives, _, _, _ = load_data(subfolders)
    cur_X = np.array(cur_X)

    if warmup_path:
        subfolders = get_sorted_subfolders(warmup_path)
        prev_X, _, prev_objectives, _, _, _ = load_data(subfolders)
        prev_X = np.array(prev_X)[:warmup_len]
        prev_objectives = np.array(prev_objectives)[:warmup_len]

        cur_X = np.concatenate([prev_X, cur_X])
        cur_objectives = np.concatenate([prev_objectives, cur_objectives])

    pickle_filename = get_picklename(path)


    with open(pickle_filename, 'rb') as f_in:
        d = pickle.load(f_in)
        xl = d['xl']
        xu = d['xu']
        mask = d['mask']
        labels = np.array(d['labels'])
        labels_to_inds = {label:i for i, label in enumerate(labels)}

        changeable_inds = np.arange(len(xu))[xu-xl>0.001]

        print(len(labels), labels)
        print('changeable inds', changeable_inds)
        print('changeable labels', len(labels[changeable_inds]), labels[changeable_inds])
        print('changeable xu', xu[changeable_inds])
        print('cur_X[0].shape', cur_X[0].shape)

        x_0_ind = labels_to_inds['pedestrian_x_0']
        y_0_ind = labels_to_inds['pedestrian_y_0']
        yaw_0_ind = labels_to_inds['pedestrian_yaw_0']
        speed_0_ind = labels_to_inds['pedestrian_speed_0']

        value_inds = (x_0_ind, y_0_ind, yaw_0_ind, speed_0_ind)


        fig, axes = plt.subplots(1, len(cutoffs), figsize=(8*len(cutoffs),5))

        num_of_unique_bugs = []

        bug_inds = np.arange(len(cur_X))[cur_objectives[:, 0] > 0.1]
        non_bug_inds = np.arange(len(cur_X))[cur_objectives[:, 0] <= 0.1]
        cur_X_bug = cur_X[bug_inds]


        inds = is_distinct_vectorized(cur_X_bug, [], mask, xl, xu, p, c, th, verbose=False)
        print('p,c,th', p, c, th)
        print('len(cur_X_bug)', len(cur_X_bug))
        print('len(inds)', len(inds))

        unique_bug_inds = bug_inds[inds]

        non_bug_color = 'green'
        bug_color = 'blue'
        unique_bug_color = 'red'


        for j, cutoff in enumerate(cutoffs):
            ax = axes[j]
            cutoff_start = cutoff - 250

            current_range = np.arange(cutoff_start, cutoff)

            # print('unique_bug_inds', unique_bug_inds)
            # print('bug_inds', bug_inds)
            for ind in non_bug_inds:
                if ind in current_range:
                    color = non_bug_color
                    plot_arrow(ax, cur_X[ind], value_inds, color)
            for ind in bug_inds:
                if ind not in unique_bug_inds:
                    if ind in current_range:
                        color = bug_color
                        plot_arrow(ax, cur_X[ind], value_inds, color)
            for ind in unique_bug_inds:
                if ind in current_range:
                    color = unique_bug_color
                    plot_arrow(ax, cur_X[ind], value_inds, color, width=0.1, head_width=0.4)

            if j == 0:
                legend_elements = [Line2D([0], [0], color=non_bug_color, lw=4, label='Non Bugs'), Line2D([0], [0], color=bug_color, lw=4, label='Bugs'), Line2D([0], [0], color=unique_bug_color, lw=4, label='Unique Bugs')]
                ax.legend(handles=legend_elements, loc=2, prop={'size': 23}, fancybox=True, framealpha=0.2)

            ax.set_title(str(cutoff_start)+'-'+str(cutoff), fontsize=20)

        fig.suptitle(save_filename+' with bugs '+str(len(unique_bug_inds))+'/'+str(len(bug_inds)), fontsize=26)
        fig.tight_layout()
        fig.savefig('tmp/'+save_filename)



def count_unique_bug_num(prev_X, cur_X, prev_objectives, cur_objectives, cutoff, label, bug_type, xl, xu, mask, p, c, th):
    if cutoff == 0:
        return 0, 0

    cutoff_start = 0
    cutoff = np.min([cutoff, len(cur_X)])
    # if label == 'ga-un':
    #     cutoff_start += warmup_pth_cutoff
    #     cutoff += warmup_pth_cutoff
    verbose = True
    prev_X_bug = []
    bug_num = 0
    if bug_type == 'collision':
        if len(prev_X) > 0:
            prev_inds = prev_objectives[:, 0] > 0.1
            prev_X_bug = prev_X[prev_inds]
        cur_inds = cur_objectives[cutoff_start:cutoff, 0] > 0.1
        cur_X_bug = cur_X[cutoff_start:cutoff][cur_inds]
        # print('prev X bug num:', len(is_distinct_vectorized(prev_X_bug, [], mask, xl, xu, p, c, th, verbose=verbose)))
        inds = is_distinct_vectorized(cur_X_bug, prev_X_bug, mask, xl, xu, p, c, th, verbose=verbose)
        bug_num += len(inds)
        print('all bug num: ', len(cur_X_bug))
        all_bug_num = len(cur_X_bug)

    elif bug_type == 'out-of-road':
        if len(prev_X) > 0:
            prev_inds = prev_objectives[:, -3] == 1
            prev_X_bug = prev_X[prev_inds]
        cur_inds = cur_objectives[cutoff_start:cutoff, -3] == 1
        cur_X_bug_1 = cur_X[cutoff_start:cutoff][cur_inds]
        inds = is_distinct_vectorized(cur_X_bug_1, prev_X_bug, mask, xl, xu, p, c, th, verbose=verbose)
        bug_num += len(inds)

        if len(prev_X) > 0:
            prev_inds = prev_objectives[:, -2] == 1
            prev_X_bug = prev_X[prev_inds]
        cur_inds = cur_objectives[cutoff_start:cutoff, -2] == 1
        cur_X_bug_2 = cur_X[cutoff_start:cutoff][cur_inds]
        inds = is_distinct_vectorized(cur_X_bug_2, prev_X_bug, mask, xl, xu, p, c, th, verbose=verbose)
        bug_num += len(inds)

        print('all bug num: ', len(cur_X_bug_1)+len(cur_X_bug_2))
        all_bug_num = len(cur_X_bug_1)+len(cur_X_bug_2)
    elif bug_type == 'all':
        if len(prev_X) > 0:
            prev_inds = prev_objectives[:, 0] > 0.1
            prev_X_bug = prev_X[prev_inds]

        cur_inds = cur_objectives[cutoff_start:cutoff, 0] > 0.1

        cur_X_bug_1 = cur_X[cutoff_start:cutoff][cur_inds]
        inds = is_distinct_vectorized(cur_X_bug_1, prev_X_bug, mask, xl, xu, p, c, th, verbose=verbose)
        bug_num += len(inds)

        if len(prev_X) > 0:
            prev_inds = prev_objectives[:, -3] == 1
            prev_X_bug = prev_X[prev_inds]

        cur_inds = cur_objectives[cutoff_start:cutoff, -3] == 1

        cur_X_bug_2 = cur_X[cutoff_start:cutoff][cur_inds]
        inds = is_distinct_vectorized(cur_X_bug_2, prev_X_bug, mask, xl, xu, p, c, th, verbose=verbose)
        bug_num += len(inds)

        prev_inds = prev_objectives[:, -2] == 1
        cur_inds = cur_objectives[cutoff_start:cutoff, -2] == 1
        prev_X_bug = prev_X[prev_inds]
        cur_X_bug_3 = cur_X[cutoff_start:cutoff][cur_inds]

        inds = is_distinct_vectorized(cur_X_bug_3, prev_X_bug, mask, xl, xu, p, c, th, verbose=verbose)
        bug_num += len(inds)

        # prev_inds = prev_objectives[:, -1] == 1
        # cur_inds = cur_objectives[cutoff_start:cutoff, -1] == 1
        # prev_X_bug = prev_X[prev_inds]
        # cur_X_bug_4 = cur_X[cutoff_start:cutoff][cur_inds]
        #
        # inds = is_distinct_vectorized(cur_X_bug_4, prev_X_bug, mask, xl, xu, p, c, th, verbose=verbose)
        # bug_num += len(inds)

        print('collision:', len(cur_X_bug_1))
        print('wronglane:', len(cur_X_bug_2))
        print('off-road:', len(cur_X_bug_3))
        # print('red-light:', len(cur_X_bug_4))
        print('all bug num:', len(cur_X_bug_1)+len(cur_X_bug_2)+len(cur_X_bug_3))
        all_bug_num = len(cur_X_bug_1)+len(cur_X_bug_2)+len(cur_X_bug_3)

    print(cutoff, bug_num)
    # print(inds)
    return bug_num, all_bug_num


def draw_unique_bug_num_over_simulations(path_list, warmup_pth_list, warmup_pth_cutoff, save_filename='num_of_unique_bugs', scene_name='', legend=True, range_upper_bound=6, bug_type='collision', unique_coeffs=[[]], plot_prev_X=False, step=50):

    fig = plt.figure()
    axes = fig.add_subplot(1,1,1)
    line_style = ['-', ':', '--', '-.', '-', ':', '--', '-.']

    p = 0

    cutoffs = [step*i for i in range(0, range_upper_bound)]

    for i, (label, pth_list) in enumerate(path_list):
        if len(unique_coeffs) == 0:
            c = 0.1
            th = 0.5
        else:
            c, th = unique_coeffs[i]

        if len(warmup_pth_list) == 1:
            warmup_pth = warmup_pth_list[0]
        else:
            warmup_pth = warmup_pth_list[i]


        num_of_unique_bugs_list = []
        num_of_all_bugs_list = []

        for pth in pth_list:
            prefix_folder = '../2020_CARLA_challenge'
            pth = os.path.join(prefix_folder, pth)
            if warmup_pth:
                warmup_pth = os.path.join(prefix_folder, warmup_pth)
                subfolders = get_sorted_subfolders(warmup_pth)
                prev_X, _, prev_objectives, _, _, _ = load_data(subfolders)
                prev_X = np.array(prev_X)[:warmup_pth_cutoff]
                prev_objectives = prev_objectives[:warmup_pth_cutoff]
            else:
                prev_X = []
                prev_objectives = []

            print('\n'*3, 'prev_X', len(prev_X), '\n'*3)
            print('-'*30, label, '-'*30)

            pickle_filename = get_picklename(pth)
            with open(pickle_filename, 'rb') as f_in:
                d = pickle.load(f_in)
                xl = d['xl']
                xu = d['xu']
                mask = d['mask']

            # print('labels', d['labels'])
            # print('xl', xl)
            # print('xu', xu)
            print('len(prev_X)', len(prev_X))
            print('len(xu-xl>0)', np.sum((np.array(xu)-np.array(xl))>1e-7))


            print('-'*20, 'pth', pth, '-'*20)
            if 'dt' in label:
                cur_X = []
                cur_objectives = []
                for filename in os.listdir(pth):
                    filepath = os.path.join(pth, filename)
                    if os.path.isdir(filepath) and len(os.listdir(filepath)) > 1:
                        subfolders = get_sorted_subfolders(filepath)
                        tmp_X, _, tmp_objectives, _, _, _ = load_data(subfolders)
                        if len(tmp_X) > 0:
                            cur_X.append(tmp_X)
                            cur_objectives.append(tmp_objectives)
                cur_X = np.concatenate(cur_X)
                cur_objectives = np.concatenate(cur_objectives)
            else:
                subfolders = get_sorted_subfolders(pth)
                cur_X, _, cur_objectives, _, _, _ = load_data(subfolders)
                cur_X = np.array(cur_X)



            if plot_prev_X and len(prev_X) > 0:
                cur_X = np.concatenate([prev_X, cur_X])
                cur_objectives = np.concatenate([prev_objectives, cur_objectives])

                prev_X = []
                prev_objectives = []

            num_of_unique_bugs = []
            num_of_all_bugs = []
            # print('prev_X.shape, cur_X.shape', prev_X.shape, cur_X.shape)
            best_num = 0
            for cutoff in cutoffs:
                num, all_bug_num = count_unique_bug_num(prev_X, cur_X, prev_objectives, cur_objectives, cutoff, label, bug_type, xl, xu, mask, p, c, th)

                # adjust for suboptimal unique bugs selection when couting
                if num > best_num:
                    best_num = num
                elif num < best_num:
                    num = best_num

                num_of_unique_bugs.append(num)
                num_of_all_bugs.append(all_bug_num)

            num_of_unique_bugs_list.append(num_of_unique_bugs)
            num_of_all_bugs_list.append(num_of_all_bugs)
        num_of_unique_bugs_list = np.array(num_of_unique_bugs_list)
        num_of_all_bugs_list = np.array(num_of_all_bugs_list)
        # print(num_of_unique_bugs_list.shape)

        if len(pth_list) == 1:
            axes.plot(cutoffs, num_of_unique_bugs_list.squeeze(), label=label, linewidth=2, linestyle=line_style[i], markersize=5, marker='.')
        else:
            num_of_unique_bugs_std = np.std(num_of_unique_bugs_list, axis=0)
            num_of_unique_bugs_mean = np.mean(num_of_unique_bugs_list, axis=0)

            axes.errorbar(cutoffs, num_of_unique_bugs, yerr=num_of_unique_bugs_std, label=label, linewidth=2, linestyle=line_style[i], capsize=5)


            unique_perc_list = num_of_unique_bugs_list / num_of_all_bugs_list
            print('mean unique percentage: ', np.mean(unique_perc_list, axis=0)[-1]*100)
            print('std unique percentage: ', np.std(unique_perc_list, axis=0)[-1]*100)



    axes.set_title(scene_name, fontsize=26)
    if legend:
        axes.legend(loc=2, prop={'size': 23}, fancybox=True, framealpha=0.2)
    axes.set_xlabel('# simulations', fontsize=26)
    axes.set_ylabel('# unique violations', fontsize=26)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fig.tight_layout()
    fig.savefig(save_filename)



def draw_simulation_wrapper(town_path_list, warmup_pth, bug_type, town, range_upper_bound, unique_coeffs, warmup_pth_cutoff=500, plot_prev_X=False, step=50):
    # 'collision', 'out-of-road'
    save_filename = 'num_of_unique_bugs_'+town+'_'+bug_type+'.pdf'
    scene_name = town + ' ' + bug_type
    print('-'*20, scene_name, '-'*20)

    draw_unique_bug_num_over_simulations(town_path_list, warmup_pth, warmup_pth_cutoff, save_filename=save_filename, scene_name=scene_name, legend=True, range_upper_bound=range_upper_bound, bug_type=bug_type, unique_coeffs=unique_coeffs, plot_prev_X=plot_prev_X, step=step)


def draw_accident_location(town_list, plot_prev_X=True):

    for label, town_path, warmup_pth in town_list:
        print(label)
        if warmup_pth:
            subfolders = get_sorted_subfolders(warmup_pth)
            prev_X, _, prev_objectives, _, _, _ = load_data(subfolders)
            prev_locations, prev_object_type_list = get_event_location_and_object_type(subfolders, verbose=False)
            prev_X = np.array(prev_X)[:warmup_pth_cutoff]
            prev_objectives = prev_objectives[:warmup_pth_cutoff]
            prev_locations = prev_locations[:warmup_pth_cutoff]
            prev_object_type_list = prev_object_type_list[:warmup_pth_cutoff]
        else:
            prev_X = []
            prev_objectives = []
            prev_locations = []
            prev_object_type_list = []


        pickle_filename = get_picklename(town_path)
        with open(pickle_filename, 'rb') as f_in:
            d = pickle.load(f_in)
            xl = d['xl']
            xu = d['xu']
            mask = d['mask']


        # print('len(prev_X)', len(prev_X))

        subfolders = get_sorted_subfolders(town_path)
        cur_X, _, cur_objectives, _, _, _ = load_data(subfolders)
        cur_X = np.array(cur_X)
        cur_locations, cur_object_type_list = get_event_location_and_object_type(subfolders, verbose=False)

        if plot_prev_X and len(prev_X) > 0:
            cur_X = np.concatenate([prev_X, cur_X])
            cur_objectives = np.concatenate([prev_objectives, cur_objectives])

            cur_locations = np.concatenate([prev_locations, cur_locations])
            cur_object_type_list = prev_object_type_list + cur_object_type_list


        collision_inds = cur_objectives[:, 0] > 0.01
        cur_object_type_list = np.array(cur_object_type_list)[collision_inds]
        cur_locations = cur_locations[collision_inds]

        pedestrian_collision_inds = cur_object_type_list == 'walker.pedestrian.0001'
        vehicle_collision_inds = cur_object_type_list == 'vehicle.dodge_charger.police'

        pedestrian_cur_locations = cur_locations[pedestrian_collision_inds]
        vehicle_cur_locations = cur_locations[vehicle_collision_inds]

        print(len(pedestrian_cur_locations), len(vehicle_cur_locations))
        plt.xlim([-151, -119])
        plt.ylim([-5, 16])
        plt.scatter(pedestrian_cur_locations[:, 0], pedestrian_cur_locations[:, 1], label='pedestrian collision')
        plt.scatter(vehicle_cur_locations[:, 0], vehicle_cur_locations[:, 1], label='vehicle collision')
        plt.scatter([-120], [15], label='start')
        plt.scatter([-144], [-4], label='end')
        plt.legend()
        plt.title(label)
        plt.gca().invert_yaxis()
        plt.savefig(label)
        plt.clf()

def count_bug(town_list):
    def count_and_group_output_unique_bugs(inds, outputs, labels, min_bounds, max_bounds, diff_th):
        '''
        ***grid counting: maximum number of distinct elements
        distinct counting: minimum number of distinct elements
        1.general
        bug type, normalized (/|start location - end location|) bug location, ego car speed when bug happens

        2.collision specific
        collision object type (i.e. pedestrian, bicyclist, small vehicle, or truck), normalized (/car width) relative angle of the other involved object at collision

        '''

        m = len(labels)

        outputs_grid_inds = ((outputs - min_bounds)*diff_th) / (max_bounds - min_bounds)
        outputs_grid_inds = outputs_grid_inds.astype(int)

        from collections import defaultdict
        unique_bugs_group = defaultdict(list)

        for i in range(outputs.shape[0]):
            unique_bugs_group[tuple(outputs_grid_inds[i])].append((inds[i], outputs[i]))

        return unique_bugs_group


    for label, town_path, warmup_pth, warmup_pth_cutoff in town_list:

        subfolders = get_sorted_subfolders(town_path)
        cur_X, _, cur_objectives, _, _, cur_info = load_data(subfolders)
        cur_locations, cur_object_type_list = get_event_location_and_object_type(subfolders, verbose=False)
        cur_X = np.array(cur_X)

        if warmup_pth:
            subfolders = get_sorted_subfolders(warmup_pth)
            prev_X, _, prev_objectives, _, _, cur_info = load_data(subfolders)
            prev_locations, prev_object_type_list = get_event_location_and_object_type(subfolders, verbose=False)
            prev_X = np.array(prev_X)[:warmup_pth_cutoff]
            prev_objectives = prev_objectives[:warmup_pth_cutoff]
            prev_locations = prev_locations[:warmup_pth_cutoff]
            prev_object_type_list = prev_object_type_list[:warmup_pth_cutoff]

            cur_X = np.concatenate([prev_X, cur_X])
            cur_objectives = np.concatenate([prev_objectives, cur_objectives])

            cur_locations = np.concatenate([prev_locations, cur_locations])
            cur_object_type_list = prev_object_type_list + cur_object_type_list

        bug_inds = np.arange(cur_X.shape[0])[cur_objectives[:, 0] > 0.01]
        cur_locations = np.array(cur_locations)[bug_inds]
        ego_speed = cur_objectives[:, 0][bug_inds]
        cur_object_type_list = np.array(cur_object_type_list)[bug_inds]
        obj_type_inds = np.zeros(len(cur_object_type_list))


        for i, obj_type in enumerate(cur_object_type_list):
            obj_type_ind = 0
            if obj_type == 'walker.pedestrian.0001':
                obj_type_ind = 1
            elif obj_type == 'vehicle.dodge_charger.police':
                obj_type_ind = 2
            obj_type_inds[i] = obj_type_ind
        ego_speed = np.expand_dims(ego_speed, axis=1)
        obj_type_inds = np.expand_dims(obj_type_inds, axis=1)

        outputs = np.concatenate([cur_locations, ego_speed, obj_type_inds], axis=1)
        from scene_configs import customized_routes
        start, end = customized_routes[cur_info['info']['route_type']]['location_list']
        x_min = np.min([start[0], end[0]])
        y_min = np.min([start[1], end[1]])
        x_max = np.max([start[0], end[0]])
        y_max = np.max([start[1], end[1]])
        print(label)
        print('x_min, x_max, y_min, y_max', x_min, x_max, y_min, y_max)
        labels = ['x', 'y', 'speed', 'object_type']

        min_bounds = np.array([x_min-1, y_min-1, 0, 1])
        max_bounds = np.array([x_max+1, y_max+1, 9, 2])


        # town07_front:
        # diff_th = np.array([1, 10, 3, 1])
        # town05_left:
        diff_th = np.array([10, 10, 3, 1])

        unique_bugs_group = count_and_group_output_unique_bugs(bug_inds, outputs, labels, min_bounds, max_bounds, diff_th)
        print('len(unique_bugs_group)', len(unique_bugs_group))
        print('unique_bugs_group', unique_bugs_group.keys())
        unique_bugs = np.array(list(unique_bugs_group.keys()))
        print(unique_bugs)




        # plt.ylim([-1, 10])
        # plt.xlim([-1, 3])
        # plt.scatter(unique_bugs[:, 2], unique_bugs[:, 1])
        # print(unique_bugs[:, 1])
        # print(unique_bugs[:, 2])
        # plt.xlabel('speed grid')
        # plt.ylabel('y axis grid')
        # plt.title(label+','+str(len(unique_bugs_group))+'/30')
        # plt.gca().invert_yaxis()
        # plt.savefig(label)
        # plt.clf()



        # plt.ylim([-1, 10])
        # plt.xlim([-1, 3])
        ped_collision_inds = unique_bugs[:, 3] == 0
        vehicle_collision_inds = unique_bugs[:, 3] == 1
        print(unique_bugs)
        unique_bugs_ped = unique_bugs[ped_collision_inds]
        unique_bugs_vehicle = unique_bugs[vehicle_collision_inds]

        # plt.scatter(unique_bugs_ped[:, 0], unique_bugs_ped[:, 1], label='pedestrian_collision')
        # plt.scatter(unique_bugs_vehicle[:, 0], unique_bugs_vehicle[:, 1], label='vehicle_collision')


        def plot_arrow(x_list, y_list, speed_list, color, width=0.002, head_width=0.06):
            for x, y, speed in zip(x_list, y_list, speed_list):
                if color == 'red':
                    dx = 0
                    dy = speed/5
                    label = 'pedestrian collision'
                elif color == 'blue':
                    dx = ((speed+0.5)/5)*np.cos(45)
                    dy =((speed+0.5)/5)*np.sin(45)
                    label = 'vehicle collision'
                plt.arrow(x, y, dx, dy, color=color, head_width=head_width, alpha=0.5, width=width, label=label)



        plot_arrow(unique_bugs_ped[:, 0], unique_bugs_ped[:, 1], unique_bugs_ped[:, 2], 'red')
        plot_arrow(unique_bugs_vehicle[:, 0], unique_bugs_vehicle[:, 1], unique_bugs_vehicle[:, 2], 'blue')


        plt.xlabel('x axis grid')
        plt.ylabel('y axis grid')
        plt.title(label+','+str(len(unique_bugs_group))+'/30')
        plt.gca().invert_yaxis()
        plt.savefig(label)
        plt.clf()








if __name__ == '__main__':
    from data_path.lbc_data_path import *

    # town_path_lists = [town07_path_list, town01_path_list, town03_out_of_road_path_list, town05_out_of_road_path_list, town07_ablation_path_list, town07_seeds_ablation_path_list]
    # warmup_pths = [warmup_pth_town07, warmup_pth_town01, warmup_pth_town03_out_of_road, warmup_pth_town05_out_of_road, warmup_pth_town07, warmup_pth_town07]
    # bug_types = ['collision', 'collision', 'out-of-road', 'out-of-road', 'collision', 'collision']
    # towns = ['town07', 'town01', 'town03', 'town05', 'town07_ablation', 'town07_seeds_ablation']
    # range_upper_bounds = [15, 15, 15, 15, 15, 7]
    # unique_coeffs_list = [[], [], [], [], [], []]

    town_path_lists = [town03_out_of_road_path_list, town05_out_of_road_path_list, town07_path_list, town01_path_list]
    warmup_pths = [[warmup_pth_town03_out_of_road, warmup_pth_town03_out_of_road, warmup_pth_town03_out_of_road, warmup_pth_town03_out_of_road, None], [warmup_pth_town05_out_of_road, warmup_pth_town05_out_of_road, warmup_pth_town05_out_of_road, warmup_pth_town05_out_of_road, None], [warmup_pth_town07, warmup_pth_town07, warmup_pth_town07, warmup_pth_town07, None], [warmup_pth_town01, warmup_pth_town01, warmup_pth_town01, warmup_pth_town01, None]]
    bug_types = ['out-of-road', 'out-of-road', 'collision', 'collision']
    towns = ['town03', 'town05', 'town07', 'town01']
    range_upper_bounds = [25, 25, 25, 25]
    unique_coeffs_list = [[], [], [], []]


    # town_path_lists = [town07_path_list]
    # warmup_pths = [[warmup_pth_town07]]
    # bug_types = ['collision']
    # towns = ['town07']
    # range_upper_bounds = [15]
    # unique_coeffs_list = [[]]

    # town_path_lists = [town01_path_list]
    # warmup_pths = [[warmup_pth_town01]]
    # bug_types = ['collision']
    # towns = ['town01']
    # range_upper_bounds = [15]
    # unique_coeffs_list = [[]]

    # town_path_lists = [town05_out_of_road_path_list]
    # warmup_pths = [[warmup_pth_town05_out_of_road]]
    # bug_types = ['out-of-road']
    # towns = ['town05']
    # range_upper_bounds = [15]
    # unique_coeffs_list = [[]]

    # ------------------------- LBC -------------------------


    # ------------------------- Apollo -------------------------

    # borresgas_path_list = [
    # ('ga-un-nn-grad',
    # ['svl_script/run_results_svl/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_05_13_29_54,10_14_adv_nn_140_coeff_0.0_0.1_0.5_only_unique_1',
    # 'svl_script/run_results_svl/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_06_17_44_01,10_14_adv_nn_140_coeff_0.0_0.1_0.5_only_unique_1',
    # 'svl_script/run_results_svl/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_06_19_28_27,10_14_adv_nn_140_coeff_0.0_0.1_0.5_only_unique_1',
    # 'svl_script/run_results_svl/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_07_18_06_10,10_14_adv_nn_140_coeff_0.0_0.1_0.5_only_unique_1',
    # 'svl_script/run_results_svl/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_07_20_20_06,10_14_adv_nn_140_coeff_0.0_0.1_0.5_only_unique_1',
    # 'svl_script/run_results_svl/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_08_09_27_08,10_14_adv_nn_140_coeff_0.0_0.1_0.5_only_unique_1'
    # ]),
    # ('nsga2-sm-un-a',
    # ['svl_script/run_results_svl/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_05_15_48_26,10_14_regression_nn_280_coeff_0.0_0.1_0.5_only_unique_1', 'svl_script/run_results_svl/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_06_21_18_15,10_14_regression_nn_280_coeff_0.0_0.1_0.5_only_unique_1', 'svl_script/run_results_svl/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_07_13_19_37,10_14_regression_nn_280_coeff_0.0_0.1_0.5_only_unique_1', 'svl_script/run_results_svl/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_07_15_31_36,10_14_regression_nn_280_coeff_0.0_0.1_0.5_only_unique_1', 'svl_script/run_results_svl/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_07_22_38_30,10_14_regression_nn_280_coeff_0.0_0.1_0.5_only_unique_1', 'svl_script/run_results_svl/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_08_00_35_23,10_14_regression_nn_280_coeff_0.0_0.1_0.5_only_unique_1'
    # ]),
    # ('av-fuzzer',
    # ['svl_script/run_results_svl/avfuzzer/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_04_23_07_34,4_120_none_240_coeff_0.0_0.1_0.5_only_unique_0', 'svl_script/run_results_svl/avfuzzer/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_05_21_09_58,4_120_none_240_coeff_0.0_0.1_0.5_only_unique_0',
    # 'svl_script/run_results_svl/avfuzzer/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_06_09_36_45,4_120_none_240_coeff_0.0_0.1_0.5_only_unique_0',
    # 'svl_script/run_results_svl/avfuzzer/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_06_13_52_55,4_120_none_240_coeff_0.0_0.1_0.5_only_unique_0',
    # 'svl_script/run_results_svl/avfuzzer/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_06_23_42_55,4_120_none_240_coeff_0.0_0.1_0.5_only_unique_0',
    # 'svl_script/run_results_svl/avfuzzer/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_07_09_29_17,4_120_none_240_coeff_0.0_0.1_0.5_only_unique_0']),
    # ]

    # borregas_warmup = 'svl_script/run_results_svl/seeds/nsga2-un/BorregasAve_left/turn_left_one_ped_and_one_vehicle/apollo_6_with_signal/2021_12_05_09_22_50,10_24_none_240_coeff_0.0_0.1_0.5_only_unique_1'
    # town_path_lists = [borresgas_path_list]
    # warmup_pths = [[borregas_warmup, borregas_warmup, None]]
    # bug_types = ['collision']
    # towns = ['Borregas Ave']
    # range_upper_bounds = [24]
    # unique_coeffs_list = [[]]

    # ------------------------- Apollo -------------------------



    # town_path_lists = [town03_out_of_road_path_list]
    # warmup_pths = [[warmup_pth_town03_out_of_road]]
    # bug_types = ['out-of-road']
    # towns = ['town03']
    # range_upper_bounds = [15]
    # unique_coeffs_list = [[]]


    # town_path_lists = [town07_path_list_500randominitial]
    # warmup_pths = [warmup_pth_town07]
    # bug_types = ['collision']
    # towns = ['town07 (500 random)']
    # range_upper_bounds = [15]
    # unique_coeffs_list = [[]]


    # town_path_lists = [town05_controllers_path_list]
    # warmup_pths = [warmup_pth_town05_collision]
    # bug_types = ['all']
    # towns = ['town05']
    # range_upper_bounds = [15]
    # unique_coeffs_list = [[]]


    # town_path_lists = [town07_nsga2_dt_unique_ablation_path_list]
    # warmup_pths = [warmup_pth_town07]
    # bug_types = ['collision']
    # towns = ['town07_unique_ablation']
    # range_upper_bounds = [7]
    # unique_coeffs_list = [[[0.05, 0.25], [0.1, 0.25], [0.2, 0.25], [0.05, 0.5], [0.1, 0.5], [0.2, 0.5], [0.05, 0.75]]]




    for i in range(len(town_path_lists)):
        town_path_list = town_path_lists[i]
        warmup_pth = warmup_pths[i]
        bug_type = bug_types[i]
        town = towns[i]
        range_upper_bound = range_upper_bounds[i]
        unique_coeffs = unique_coeffs_list[i]
        draw_simulation_wrapper(town_path_list, warmup_pth, bug_type, town, range_upper_bound, unique_coeffs, warmup_pth_cutoff=500, plot_prev_X=True, step=50)



    one_ped_path_list = [('random', 'run_results/random/town07_front_0/go_straight_town07_one_ped/lbc/2021_03_15_10_55_32,50_20_none_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.1__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', None, None), ('random-un', 'run_results/random-un/town07_front_0/go_straight_town07_one_ped/lbc/2021_03_15_10_55_36,50_25_none_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.1__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', None, None), ('ga', 'run_results/nsga2/town07_front_0/go_straight_town07_one_ped/lbc/2021_03_15_20_55_46,50_20_none_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.1__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', None, None), ('ga-un', 'run_results/nsga2-un/town07_front_0/go_straight_town07_one_ped/lbc/2021_03_15_10_55_24,50_25_none_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.1__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', None, None), ('ga-un-nn', 'run_results/nsga2-un/town07_front_0/go_straight_town07_one_ped/lbc/2021_03_15_20_55_58,50_20_nn_500_100_1.01_-4_0.9_coeff_0.0_0.1_0.1__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town07_front_0/go_straight_town07_one_ped/lbc/2021_03_15_10_55_24,50_25_none_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.1__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', 500), ('ga-un-adv-nn', 'run_results/nsga2-un/town07_front_0/go_straight_town07_one_ped/lbc/2021_03_15_20_55_52,50_20_adv_nn_500_100_1.01_-4_0.9_coeff_0.0_0.1_0.1__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town07_front_0/go_straight_town07_one_ped/lbc/2021_03_15_10_55_24,50_25_none_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.1__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', 500)]
    # for alg, path, warmup_path, warmup_len in one_ped_path_list:
    #     print('\n'*3, '-'*20, alg, '-'*20, '\n'*3)
    #     visualize_ped_over_time(path, save_filename='ped_over_time'+'_'+alg, bug_type='collision', unique_coeffs=[0.1, 0.1], range_upper_bound=5, warmup_path=warmup_path, warmup_len=warmup_len)



    town05_left_collision_list = [('nsga2', 'run_results/nsga2/town05_left_0/turn_left_town05/lbc/2021_03_30_19_06_09,50_20_none_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.1__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', None, None),
    ('nsga2-un', 'run_results/nsga2-un/town05_left_0/turn_left_town05/lbc/2021_03_30_10_54_55,50_25_none_1000_100_1.01_-4_0.9_coeff_0.0_0.2_0.2__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', None, None),
    ('random', 'run_results/random/town05_left_0/turn_left_town05/lbc/2021_03_30_19_06_02,50_20_none_1000_100_1.01_-4_0.9_coeff_0.0_0.1_0.1__one_output_n_offsprings_300_200_200_only_unique_0_eps_1.01', None, None),
    ('adv_nn', 'run_results/nsga2-un/town05_left_0/turn_left_town05/lbc/2021_03_30_22_55_47,50_25_adv_nn_500_100_1.01_-4_0.9_coeff_0.0_0.1_0.1__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town05_left_0/turn_left_town05/lbc/2021_03_30_10_54_55,50_25_none_1000_100_1.01_-4_0.9_coeff_0.0_0.2_0.2__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 500),
    ('nn', 'run_results/nsga2-un/town05_left_0/turn_left_town05/lbc/2021_03_30_22_55_54,50_25_nn_500_100_1.01_-4_0.9_coeff_0.0_0.1_0.1__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 'run_results/nsga2-un/town05_left_0/turn_left_town05/lbc/2021_03_30_10_54_55,50_25_none_1000_100_1.01_-4_0.9_coeff_0.0_0.2_0.2__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', 500),
    ('nsga2-un-lbc-augment-ped', 'run_results/nsga2-un/town05_left_0/turn_left_town05/lbc_augment_ped/2021_04_04_15_54_46,50_25_none_1000_100_1.01_-4_0.9_coeff_0.0_0.2_0.2__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', None, None),
    ('nsga2-un-lbc-augment-ped-2', 'run_results/nsga2-un/town05_left_0/turn_left_town05/lbc_augment_ped/2021_04_04_22_48_47,50_25_none_1000_100_1.01_-4_0.9_coeff_0.0_0.2_0.2__one_output_n_offsprings_300_200_200_only_unique_1_eps_1.01', None, None)
    ]

    # draw_accident_location(town05_left_collision_list)


    # town_path_list = [(p[0], [p[1]]) for p in one_ped_path_list]
    # warmup_pth_list = [p[2] for p in one_ped_path_list]
    # draw_simulation_wrapper(town_path_list, warmup_pth_list, bug_type='collision', town='town07', range_upper_bound=21, unique_coeffs=[], warmup_pth_cutoff=500, plot_prev_X=True)






    # count_bug(borregas_collision_list)

    # rerun/bugs/train/2021_04_04_15_00_30_non_train_lbc_agent_ped_no_debug/town05_left_0_Scenario12_lbc_augment_00
    # rerun/bugs/train/2021_04_04_14_59_45_vehicle_train_lbc_agent_ped_no_debug/town05_left_0_Scenario12_lbc_augment_00
