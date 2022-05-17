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





# TBD: visualize synthetic function bug distribution (2d)
def visualize_synthetic_function_bugs():
    pass

# -------------------- helper functions for visualize_data --------------------
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)




def plot_arrow(ax, values, label, color, plot_dim, legend=False, width=0.001, head_width=0.01):
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

        if plot_dim == 2:
            # since yaw will be represented by orientation, its value range is different from others
            yaw = yaw * 360
            yaw = np.deg2rad(yaw)

            dx = np.cos(yaw)*length*0.1
            dy = np.sin(yaw)*length*0.1

        if legend:
            label = str(label)
        else:
            label = None

        if plot_dim == 2:
            ax.arrow(x, y, dx, dy, color=color, head_width=head_width, alpha=0.5, width=width, label=label)
        elif plot_dim == 3:
            # ax.arrow3D(x,y,0.5, x+dx,y+dy,0.7, mutation_scale=20, arrowstyle="-|>", linestyle='dashed', color=color, label=label)
            ax.scatter(x, y, yaw, color=color, label=label)

    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-0.1, 1.1)
    if plot_dim == 3:
        ax.set_zlim(-0.1, 1.1)

def plot_subplot(ax, x_list, y_list, chosen_inds, unique_y_list, legend, mode, chosen_labels, plot_dim, split_label_v_pair=()):

    x_sublist = x_list[chosen_inds]
    y_sublist = y_list[chosen_inds]
    colors = ['black', 'red', 'gray', 'lightgray', 'brown', 'salmon', 'orange', 'yellowgreen', 'green', 'blue', 'purple', 'magenta', 'pink']
    for j, y in enumerate(unique_y_list):
        color = colors[j]
        x_subset = x_sublist[y_sublist==y]
        for k in range(x_subset.shape[0]):
            if legend and k == 0:
                plot_arrow(ax, x_subset[k], y, color, plot_dim, legend=True)
            else:
                plot_arrow(ax, x_subset[k], y, color, plot_dim, legend=False)

    if len(split_label_v_pair) > 0:
        subplot_split_label, v = split_label_v_pair
        ax.set_title(subplot_split_label+' = '+v, fontsize=18)
    else:
        ax.set_title('samples '+str(chosen_inds[0])+' to '+str(chosen_inds[1]), fontsize=18)

    if legend:
        ax.legend(loc='lower right', prop={'size': 16}, fancybox=True, framealpha=0.5)
    if mode == 'plain':
        ax.set_xlabel(chosen_labels[0])
        ax.set_ylabel(chosen_labels[1])
        if plot_dim == 3:
            ax.set_zlabel(chosen_labels[2])

# extract data from result folder
def extract_data_from_fuzzing(folder_path):
    data_path = os.path.join(folder_path, 'data.pickle')
    with open(data_path, 'rb') as f_in:
        data_d = pickle.load(f_in)

    x_list = data_d['x_list']
    y_list = data_d['y_list']
    x_labels = np.array(data_d['labels'])
    print('all x_labels', x_labels)

    xl = data_d['xl']
    xu = data_d['xu']
    used_labels_inds = xu - xl > 0

    x_list = x_list[:, used_labels_inds]
    x_labels = x_labels[used_labels_inds]
    print('used x_labels', x_labels)

    return x_list, y_list, x_labels

def extract_data_from_csv(folder_path, filename, x_labels, y_label):
    import pandas
    df = pandas.read_csv(os.path.join(folder_path, filename))

    x_list = df[x_labels].to_numpy()
    y_list = df[y_label].to_numpy()
    # print('x_list.shape', x_list.shape, 'y_list.shape', y_list.shape)

    return x_list, y_list, np.array(x_labels)


# -------------------- helper functions for visualize_data --------------------
def visualize_data(save_folder_path, initial_x_list, y_list, x_labels, num_subplots, mode, dim, chosen_labels, plot_dim, subplot_split_label=''):
    # normalize the data first
    from sklearn.preprocessing import MinMaxScaler
    transformer = MinMaxScaler().fit(initial_x_list)
    x_list_normalized = transformer.transform(initial_x_list)

    if plot_dim == 3:
        assert dim == plot_dim

    if subplot_split_label:
        split_ind = np.where(x_labels==subplot_split_label)[0][0]

    if mode == 'plain':
        assert len(chosen_labels) == dim
        inds_list = []
        for chosen_label in chosen_labels:
            assert chosen_label in x_labels
            inds_list.append(np.where(x_labels==chosen_label)[0][0])
    else:
        # TBD: when applying dimensionality reduction, exclude the split label if it is used.


        # dimensionality reduction is only used when input dimension is larger than the visualization dimension
        assert x_list_normalized.shape[1] > dim

        if mode == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=dim, svd_solver='full')
            pca.fit(x_list_normalized)
            print('dim', dim, 'pca.explained_variance_ratio_', pca.explained_variance_ratio_)
            x_list_normalized = pca.transform(x_list_normalized)
            inds_list = [i for i in range(dim)]
        elif mode == 'tsne':
            if plot_dim == 2:
                assert dim == 2
            elif plot_dim == 3:
                assert dim == 3
            from sklearn.manifold import TSNE
            print('x_list_normalized.shape', x_list_normalized.shape)
            x_list_normalized = TSNE(n_components=dim).fit_transform(x_list_normalized)
            inds_list = [i for i in range(dim)]
        else:
            print('mode', mode)
            raise

    print('mode', mode, 'dim', dim, 'chosen_labels', chosen_labels)
    print('x_list_normalized.shape', x_list_normalized.shape)
    print('y_list.shape', y_list.shape)

    x_list = x_list_normalized[:, inds_list]
    unique_y_list = np.unique(y_list)

    if subplot_split_label:
        v_list = np.unique(x_list_normalized[:, split_ind])
        num_subplots = len(v_list)

    assert num_subplots >= 1
    assert x_list_normalized.shape[0] >= num_subplots

    num_subplots_col_num = int(np.ceil(np.sqrt(num_subplots)))
    num_subplots_row_num = int(np.ceil(num_subplots / num_subplots_col_num))

    if num_subplots > 1:
        # for the overall plot at the first row
        num_subplots_row_num += 1
    if plot_dim == 2:
        projection = None
    else:
        projection = '3d'

    unit_size = 6
    fig = plt.figure(figsize=(num_subplots_col_num*unit_size, num_subplots_row_num*unit_size))

    # draw an overall plot
    ax = fig.add_subplot(num_subplots_col_num, num_subplots_row_num, 1, projection=projection)
    chosen_inds = np.arange(0, x_list.shape[0])

    plot_subplot(ax, x_list, y_list, chosen_inds, unique_y_list, True, mode, chosen_labels, plot_dim, split_label_v_pair=(subplot_split_label, 'any'))


    if subplot_split_label:
        for i, v in enumerate(v_list):
            ax = fig.add_subplot(num_subplots_col_num, num_subplots_row_num, i+num_subplots_col_num+1, projection=projection)

            chosen_inds = np.where(x_list_normalized[:, split_ind]==v)[0]
            print('v', v, 'len(chosen_inds)', len(chosen_inds), chosen_inds[:3])
            plot_subplot(ax, x_list, y_list, chosen_inds, unique_y_list, False, mode, chosen_labels, plot_dim, split_label_v_pair=(subplot_split_label, '{:.1f}'.format(v)))

        fig.suptitle(mode+' with '+str(dim)+' dimensions for different '+subplot_split_label, fontsize=25)

    else:
        num_per_subplot = int(np.ceil(len(y_list) / num_subplots))
        # draw subplots
        if num_subplots > 1:
            for i in range(num_subplots):
                ax = fig.add_subplot(num_subplots_col_num, num_subplots_row_num, i+num_subplots_col_num+1, projection=projection)

                left, right = i*num_per_subplot, (i+1)*num_per_subplot
                if i == num_subplots-1:
                    right = x_list.shape[0]
                chosen_inds = np.arange(left, right)
                plot_subplot(ax, x_list, y_list, chosen_inds, unique_y_list, False, mode, chosen_labels, plot_dim)

            fig.suptitle(mode+' with '+str(dim)+' dimensions for '+str(x_list.shape[0])+' samples', fontsize=25)

    fig.savefig(os.path.join(save_folder_path, mode+'_'+str(dim)+'_'+str(x_list.shape[0])+'.jpg'))


if __name__ == '__main__':
    # -------------------- Dataset Visualization Parameters--------------------
    folder_path = 'no_simulation_dataset_script'
    filename = 'grid.csv'
    # The values with these labels will be extracted
    x_labels = ['ego_pos', 'ego_init_speed', 'other_pos', 'other_init_speed', 'ped_delay', 'ped_init_speed']
    # The interested target's label
    y_label = 'oob'
    x_list, y_list, x_labels = extract_data_from_csv(folder_path, filename, x_labels, y_label)


    # -------------------- Fuzzing + Visualization Parameters --------------------
    # folder_path = 'no_simulation_function_script/run_results_no_simulation/nsga2/four_modes/2022_05_09_18_03_17,50_10_none_500_coeff_0_0.1_0.5_only_unique_0'
    # folder_path = 'carla_lbc/run_results/nsga2/town07_front_0/go_straight_town07_one_ped/lbc/2022_05_09_23_07_38,50_10_none_500_coeff_0.0_0.1_0.5_only_unique_0'
    # x_list, y_list, labels = extract_data_from_folder(folder_path)


    # -------------------- Common Parameters --------------------

    # The number of subsets to split all the data during fuzzing. It needs to be a positive integer and less than or equal to (usually far less than) the number of data points. When it is set to 1, only a plot with all the data points will be shown.
    num_subplots = 1

    # The visualization method. ['plain', 'pca', 'tsne']
    mode = 'plain'

    # The number of dimensions to visualize. For 'plain', 2 to 4 are supported and dim must be equal to len(chosen_labels); For 'pca', 2 to 4 are supported; for 'tsne', 2 to 3 are supported and plot_dim must be equal to dim
    dim = 4

    # The labels used for visualization. It is used only if mode == 'plain' and every label in the chosen_labels must be in labels
    chosen_labels = ['ego_pos', 'ego_init_speed', 'other_pos', 'other_init_speed']

    # The dimensionality for plotting. [2, 3]. Note if plot_dim == 3, currently only dim == 3 is supported.
    plot_dim = 2

    # The label used for splitting subplots (it is either None or an element in x_labels). When it is not None, num_subplots will be determined by the number of unique values of subplot_split_label in x_list. Usually this is set to be a categorical feature.
    subplot_split_label = 'ped_delay'

    visualize_data(folder_path, x_list, y_list, x_labels, num_subplots, mode, dim, chosen_labels, plot_dim, subplot_split_label=subplot_split_label)
