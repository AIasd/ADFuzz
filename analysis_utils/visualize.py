import sys
sys.path.append('.')
sys.path.append('pymoo')
sys.path.append('fuzzing_utils')
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()


def plot_arrow(ax, values, label, color, legend=False, width=0.001, head_width=0.01):
    if len(values) == 2:
        x, y = values
        yaw = 0
        length = 0
        head_width = 0
        if legend:
            ax.scatter(x, y, color=color, label=str(label))
        else:
            ax.scatter(x, y, color=color)
    else:
        if len(values) == 3:
            x, y, yaw = values
            length = 0.05
        else:
            x, y, yaw, length = values

        # since yaw will be represented by orientation, its value range is different from others
        yaw = yaw * 360
        yaw = np.deg2rad(yaw)

        dx = np.cos(yaw)*length*0.1
        dy = np.sin(yaw)*length*0.1
        if legend:
            ax.arrow(x, y, dx, dy, color=color, head_width=head_width, alpha=0.5, width=width, label=str(label))
        else:
            ax.arrow(x, y, dx, dy, color=color, head_width=head_width, alpha=0.5, width=width)

    # ax.set_ylim(-11, 11)
    # ax.set_xlim(11, -11)

# plain
def visualize_data(folder_path, num_subplots, mode, dim, chosen_labels):
    data_path = os.path.join(folder_path, 'data.pickle')
    with open(data_path, 'rb') as f_in:
        data_d = pickle.load(f_in)

    x_list = data_d['x_list']
    y_list = data_d['y_list']
    labels = np.array(data_d['labels'])
    print('all labels', labels)

    xl = data_d['xl']
    xu = data_d['xu']
    used_labels_inds = xu - xl > 0

    x_list = x_list[:, used_labels_inds]
    labels = labels[used_labels_inds]
    print('used labels', labels)

    # normalize the data first
    from sklearn.preprocessing import MinMaxScaler
    transformer = MinMaxScaler().fit(x_list)
    x_list = transformer.transform(x_list)

    assert x_list.shape[0] >= num_subplots

    if mode == 'plain':
        assert len(chosen_labels) == dim
        inds_list = []
        for chosen_label in chosen_labels:
            assert chosen_label in labels
            inds_list.append(np.where(labels==chosen_label)[0][0])
    else:
        # dimensionality reduction is only used when input dimension is larger than the visualization dimension
        assert x_list.shape[1] > dim

        if mode == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=dim)
            pca.fit(x_list)
            x_list = pca.transform(x_list)
            inds_list = [i for i in range(dim)]
        elif mode == 'tsne':
            assert dim == 2
            from sklearn.manifold import TSNE
            print('x_list.shape', x_list.shape)
            x_list = TSNE(n_components=dim).fit_transform(x_list)
            inds_list = [i for i in range(dim)]
        else:
            print('mode', mode)
            raise

    print('mode', mode, 'dim', dim, 'chosen_labels', chosen_labels)
    print(data_d.keys())
    print('x_list.shape', x_list.shape)
    print('y_list.shape', y_list.shape)

    colors = ['black', 'pink', 'gray', 'lightgray', 'brown', 'red', 'salmon', 'orange', 'yellowgreen', 'green', 'blue', 'purple', 'magenta']

    unique_y_list = np.unique(y_list)
    num_per_subplot = int(np.ceil(len(y_list) / num_subplots))

    num_subplots_row_num = int(np.ceil(np.sqrt(num_subplots)))
    num_subplots_col_num = int(np.ceil(num_subplots / num_subplots_row_num))

    fig, axs = plt.subplots(num_subplots_col_num, num_subplots_row_num, figsize=(num_subplots_row_num*5, num_subplots_col_num*5))

    for i in range(num_subplots):
        cur_row = i // num_subplots_row_num
        cur_col = i % num_subplots_row_num
        ax = axs[cur_row, cur_col]

        left, right = i*num_per_subplot, (i+1)*num_per_subplot
        if i == num_subplots-1:
            right = x_list.shape[0]

        x_sublist = x_list[left:right]
        y_sublist = y_list[left:right]

        for j, y in enumerate(unique_y_list):
            color = colors[j]
            x_subset = x_sublist[y_sublist==y]
            x_subset = x_subset[:, inds_list]
            print('left', left, 'right', right, 'y', y, 'x_subset.shape', x_subset.shape)

            for k in range(x_subset.shape[0]):
                if i == num_subplots-1 and k == 0:
                    plot_arrow(ax, x_subset[k], y, color, legend=True)
                else:
                    plot_arrow(ax, x_subset[k], y, color, legend=False)

            # ax.scatter(x_subset[:, ind_0], x_subset[:, ind_1], s=10, c=color, label=str(y))
        ax.set_title('samples '+str(left)+' to '+str(right), fontsize=20)
        if i == num_subplots-1:
            ax.legend(loc='lower right', prop={'size': 16}, fancybox=True, framealpha=0.5)
    fig.suptitle(mode+' with '+str(dim)+' dimensions for '+str(x_list.shape[0])+' samples', fontsize=30)
    fig.savefig(os.path.join(folder_path, mode+'_'+str(dim)+'_'+str(x_list.shape[0])+'.png'))


if __name__ == '__main__':
    # The folder consists of the fuzzing data
    folder_path = 'no_simulation_function_script/run_results_no_simulation/nsga2/four_modes/2022_05_09_18_03_17,50_10_none_500_coeff_0_0.1_0.5_only_unique_0'
    # folder_path = 'carla_lbc/run_results/nsga2-un/town07_front_0/go_straight_town07_one_ped/lbc/2022_05_09_15_25_27,10_2_none_20_coeff_0.0_0.1_0.5_only_unique_1'

    # The number of subsets to split all the data during fuzzing. It needs to be an integer and less than or equal to (usually far less than) the number of data points.
    num_subplots = 10

    # The visualization method. ['plain', 'pca', 'tsne']
    mode = 'plain'

    # The number of dimensions to visualize. For 'plain', 2 to 4 are supported and dim must be equal to len(chosen_labels); For 'pca', 2 to 4 are supported; for 'tsne', only 2 is supported
    dim = 2
    # dim = 4

    # The labels used for visualization. It is used only if mode == 'plain' and every label in the chosen_labels must be in labels
    chosen_labels = ['x1', 'x2']
    # chosen_labels = ['pedestrian_x_0', 'pedestrian_y_0', 'pedestrian_yaw_0', 'pedestrian_speed_0']

    visualize_data(folder_path, num_subplots, mode, dim, chosen_labels)
