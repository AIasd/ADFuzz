import numpy as np
from pymoo.model.sampling import Sampling
from customized_utils import if_violate_constraints_vectorized, is_distinct_vectorized

def sample_one_feature(typ, lower, upper, dist, label, rng, size=1):
    assert lower <= upper, label+','+str(lower)+'>'+str(upper)
    if typ == 'int':
        val = rng.integers(lower, upper+1, size=size)
    elif typ == 'real':
        if dist[0] == 'normal':
            if dist[1] == None:
                mean = (lower+upper)/2
            else:
                mean = dist[1]
            val = rng.normal(mean, dist[2], size=size)
        else: # default is uniform
            val = rng.random(size=size) * (upper - lower) + lower
        val = np.clip(val, lower, upper)
    return val


# unique sampling / random sampling
class MySamplingVectorized(Sampling):
    def __init__(self, random_seed, use_unique_bugs, check_unique_coeff, sample_multiplier=500):
        self.rng = np.random.default_rng(random_seed)
        self.use_unique_bugs = use_unique_bugs
        self.check_unique_coeff = check_unique_coeff
        self.sample_multiplier = sample_multiplier
        assert len(self.check_unique_coeff) == 3
    def _do(self, problem, n_samples, **kwargs):
        p, c, th = self.check_unique_coeff
        xl = problem.xl
        xu = problem.xu
        mask = np.array(problem.mask)
        labels = problem.labels
        parameters_distributions = problem.parameters_distributions

        if self.sample_multiplier >= 50:
            max_sample_times = self.sample_multiplier // 50
            n_samples_sampling = n_samples * 50
        else:
            max_sample_times = self.sample_multiplier
            n_samples_sampling = n_samples

        algorithm = kwargs['algorithm']

        tmp_off = algorithm.tmp_off


        tmp_off_and_X = []
        if len(tmp_off) > 0:
            tmp_off = [off.X for off in tmp_off]
            tmp_off_and_X = tmp_off


        def subroutine(X, tmp_off_and_X):
            # TBD: temporary
            sample_time = 0
            while sample_time < max_sample_times and len(X) < n_samples:
                print('sample_time / max_sample_times', sample_time, '/', max_sample_times, 'len(X)', len(X))
                sample_time += 1
                cur_X = []
                for i, dist in enumerate(parameters_distributions):
                    typ = mask[i]
                    lower = xl[i]
                    upper = xu[i]
                    label = labels[i]
                    val = sample_one_feature(typ, lower, upper, dist, label, self.rng, size=n_samples_sampling)
                    cur_X.append(val)
                cur_X = np.swapaxes(np.stack(cur_X),0,1)


                remaining_inds = if_violate_constraints_vectorized(cur_X, problem.customized_constraints, problem.labels, problem.ego_start_position, verbose=False)
                if len(remaining_inds) == 0:
                    continue

                cur_X = cur_X[remaining_inds]

                if not self.use_unique_bugs:
                    X.extend(cur_X)
                    if len(X) > n_samples:
                        X = X[:n_samples]
                else:
                    if len(tmp_off_and_X) > 0 and len(problem.interested_unique_bugs) > 0:
                        prev_X = np.concatenate([problem.interested_unique_bugs, tmp_off_and_X])
                    elif len(tmp_off_and_X) > 0:
                        prev_X = tmp_off_and_X
                    else:
                        prev_X = problem.interested_unique_bugs

                    remaining_inds = is_distinct_vectorized(cur_X, prev_X, mask, xl, xu, p, c, th, verbose=False)

                    if len(remaining_inds) == 0:
                        continue
                    else:
                        cur_X = cur_X[remaining_inds]
                        X.extend(cur_X)
                        if len(X) > n_samples:
                            X = X[:n_samples]
                        if len(tmp_off) > 0:
                            tmp_off_and_X = tmp_off + X
                        else:
                            tmp_off_and_X = X
            return X, sample_time


        X = []
        X, sample_time_1 = subroutine(X, tmp_off_and_X)

        if len(X) > 0:
            X = np.stack(X)
        else:
            X = np.array([])
        print('\n'*3, 'We sampled', X.shape[0], '/', n_samples, 'samples', 'by sampling', sample_time_1, 'times' '\n'*3)

        return X

# grid sampling
class GridSampling(Sampling):
    def __init__(self, random_seed, grid_start_index, grid_dict):
        self.rng = np.random.default_rng(random_seed)
        self.grid_start_index = grid_start_index

        import itertools
        gird_value_list = list(itertools.product(*list(grid_dict.values())))
        print('total combinations:', len(gird_value_list))
        gird_value_list = list(zip(*gird_value_list))
        self.grid_value_dict = {k:gird_value_list[i] for i, k in enumerate(grid_dict.keys())}


    def _do(self, problem, n_samples, **kwargs):
        xl = problem.xl
        xu = problem.xu
        mask = np.array(problem.mask)
        labels = problem.labels
        parameters_distributions = problem.parameters_distributions
        print('\n', 'self.grid_start_index:', self.grid_start_index, '\n')
        def subroutine(X):
            n_samples_sampling = n_samples
            while len(X) < n_samples:
                cur_X = []
                for i, dist in enumerate(parameters_distributions):
                    typ = mask[i]
                    lower = xl[i]
                    upper = xu[i]
                    label = labels[i]
                    if label in self.grid_value_dict:
                        assert self.grid_start_index+n_samples_sampling <= len(self.grid_value_dict[label]), str(self.grid_start_index+n_samples_sampling)+'>'+str(len(self.grid_value_dict[label]))
                        val = self.grid_value_dict[label][self.grid_start_index:self.grid_start_index+n_samples_sampling]
                    else:
                        val = sample_one_feature(typ, lower, upper, dist, label, self.rng, size=n_samples_sampling)
                    cur_X.append(val)
                cur_X = np.swapaxes(np.stack(cur_X),0,1)

                remaining_inds = if_violate_constraints_vectorized(cur_X, problem.customized_constraints, problem.labels, problem.ego_start_position, verbose=False)
                if len(remaining_inds) == 0:
                    continue

                cur_X = cur_X[remaining_inds]
                X.extend(cur_X)

                self.grid_start_index += n_samples
                n_samples_sampling = n_samples - len(X)

            return X
        X = []
        X = subroutine(X)

        if len(X) > 0:
            X = np.stack(X)
        else:
            X = np.array([])

        return X




# 1.random search
# 2.for found bug -> build sphere (end when radius is smaller than a th or enough simulation)
# 3.if left bug -> go to 2; otherwise -> go to 1.
# end when enough simulation number reaches
class Sphere():
    def __init__(self, ind, x, max_radius=0.5):
        self.center = (ind, x)
        self.cur_radius = 0.1
        self.max_radius = max_radius
        self.members = []
        self.normal_members = []
        self.sampling_num = 0

        # only use for gss
        self.cur_dir = None

        # hyper-parameters
        self.stop_radius = 0.1
        self.max_num_points_per_center = 10

    def if_local_sampling(self):
        condition1 = self.sampling_num < self.max_num_points_per_center
        condition2 = self.max_radius > self.stop_radius
        return condition1 and condition2

    def cover(self, x):
        member_d_to_center = np.linalg.norm(self.center[1]-x)
        return member_d_to_center <= self.cur_radius

    def add_member(self, ind, x, y):
        print('x', x, 'y', y)
        if y > 0:
            self.members.append((ind, x))
            x_d_to_center = np.linalg.norm(self.center[1]-x)
            self.cur_radius = np.max([self.cur_radius, x_d_to_center])
            self.cur_radius = np.min([self.cur_radius, self.max_radius])
        else:
            self.normal_members.append((ind, x))
            x_d_to_center = np.linalg.norm(self.center[1]-x)
            print('x_d_to_center', x_d_to_center)
            self.max_radius = np.min([self.max_radius, x_d_to_center])
            self.cur_radius = np.min([self.cur_radius, x_d_to_center])

            # only keep members that are closer to center than the non-bug sample
            new_members = []
            for member in self.members:
                if self.cover(member[1]):
                    new_members.append(member)
            self.members = new_members
            # TBD: also consider the regions of other spheres when encountering a new non-bug sample

class RandomDirectionSampling(Sampling):
    def __init__(self, random_seed, chosen_labels):
        self.rng = np.random.default_rng(random_seed)
        self.chosen_labels = chosen_labels
        self.num_dim = len(self.chosen_labels)

        self.spheres = []
        self.normal_samples = []
        self.cur_ind = -1
        self.chosen_inds = []
        self.local_sampling = False
        self.max_raidus = 0.5

        self.sphere_center_d_th = 0.1

        # sphere local sampling ends when either:
        # (1) the sphere is smaller than a threshold
        # (2) the number of points sampled reaches a threshold
        self.min_sphere_radius = 0.01


    def direction_sampling(self, sphere, chosen_xl, chosen_xu):
        def search_lamb_bounds(center, dir, chosen_xl, chosen_xu, max_radius):
            lamb1 = (chosen_xl - center) / dir
            lamb2 = (chosen_xu - center) / dir
            lamb_all = np.concatenate([lamb1, lamb2])

            lamb_min = np.max(lamb_all[lamb_all <= 0])
            lamb_max = np.min(lamb_all[lamb_all >= 0])

            lamb_min = np.max([lamb_min, -max_radius])
            lamb_max = np.min([lamb_max, max_radius])

            print('lamb_min', lamb_min, 'lamb_max', lamb_max)

            return lamb_min, lamb_max

        def sample_direction():
            mean = np.zeros(self.num_dim)
            var = np.ones(self.num_dim)
            dir = self.rng.normal(mean, var)
            dir /= np.linalg.norm(dir)
            return dir

        strategy = 'random'
        if strategy == 'random':
            dir = sample_direction()
            lamb_min, lamb_max = search_lamb_bounds(sphere.center[1], dir, chosen_xl, chosen_xu, sphere.max_radius)
            lamb = lamb_min + self.rng.random() * (lamb_max-lamb_min)

        new_x_chosen_labels = sphere.center[1] + lamb * dir
        # print('sphere.center[1]', sphere.center[1], 'lamb', lamb, 'dir', dir)
        # print('new_x_chosen_labels', new_x_chosen_labels)

        return new_x_chosen_labels

    def d_to_spheres(self, x_list):
        if len(self.spheres) > 0:
            sphere_centers = [sphere.center[1] for sphere in self.spheres]
            sphere_radiuses = [sphere.cur_radius for sphere in self.spheres]
            d_mat = np.linalg.norm(np.expand_dims(x_list, 1) - np.expand_dims(sphere_centers, 0), axis=2) - np.expand_dims(sphere_radiuses, 0)
            d_list = np.min(d_mat, axis=1)
        else:
            d_list = np.array([self.sphere_center_d_th*2]*len(x_list))
        return d_list

    def find_all_uncovered_bug_inds(self, x_list, y_list):
        y_list_np = np.array(y_list)
        bug_inds = np.where(y_list_np > 0)[0].tolist()
        for sphere in self.spheres:
            center_ind = sphere.center[0]
            bug_inds.remove(center_ind)
            for ind, bug in sphere.members:
                bug_inds.remove(ind)

        x_list = np.array(x_list)
        bug_inds = np.array(bug_inds)
        bug_list = x_list[bug_inds]
        d_list = self.d_to_spheres(bug_list)
        return bug_inds, d_list

    def add_uncovered_coverable_bugs(self, x_list, y_list):
        bug_inds, _ = self.find_all_uncovered_bug_inds(x_list, y_list)
        for bug_ind in bug_inds:
            for i in range(len(self.spheres)):
                if self.spheres[i].cover(x_list[bug_ind]):
                    self.spheres[i].add_member(bug_ind, x_list[bug_ind], y_list[bug_ind])
                    break

    def find_an_uncovered_bug(self, x_list, y_list):
        bug_inds, d_list = self.find_all_uncovered_bug_inds(x_list, y_list)
        uncovered_bug = None
        if len(bug_inds) > 0:
            max_ind = np.argmax(d_list)
            max_d = d_list[max_ind]
            if max_d > self.sphere_center_d_th:
                uncovered_bug = (bug_inds[max_ind], x_list[bug_inds[max_ind]])
        return uncovered_bug

    def update_cur_sphere(self, latest_ind, latest_x, latest_y):
        self.spheres[self.cur_ind].add_member(latest_ind, latest_x, latest_y)
        self.spheres[self.cur_ind].sampling_num += 1


    def new_sphere(self, uncovered_bug, x_list, y_list):
        ind, center = uncovered_bug
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            if y == 0 and i != ind:
                x_d_to_center = np.linalg.norm(center-x)
                max_radius = np.min([self.max_raidus, x_d_to_center])
        sphere = Sphere(ind, center, self.max_raidus)
        self.spheres.append(sphere)
        self.cur_ind += 1

    def _do(self, problem, n_samples, **kwargs):

        sphere = self.spheres[self.cur_ind]

        xl = problem.xl
        xu = problem.xu
        mask = np.array(problem.mask)
        labels = problem.labels
        parameters_distributions = problem.parameters_distributions

        if len(self.chosen_inds) == 0:
            for i, label in enumerate(labels):
                if label in self.chosen_labels:
                    self.chosen_inds.append(i)

        chosen_xl = xl[self.chosen_inds]
        chosen_xu = xu[self.chosen_inds]


        X = []
        while len(X) < n_samples:
            cur_X = []

            new_x_chosen_labels = self.direction_sampling(sphere, chosen_xl, chosen_xu)

            for i, dist in enumerate(parameters_distributions):
                typ = mask[i]
                lower = xl[i]
                upper = xu[i]
                label = labels[i]
                # currently only support sampling one sample at a time
                if label in self.chosen_labels:
                    ind = self.chosen_labels.index(label)
                    val = [new_x_chosen_labels[ind]]
                else:
                    val = sample_one_feature(typ, lower, upper, dist, label, self.rng, size=n_samples_sampling)
                cur_X.append(val)

            cur_X = np.swapaxes(np.stack(cur_X),0,1)

            remaining_inds = if_violate_constraints_vectorized(cur_X, problem.customized_constraints, problem.labels, problem.ego_start_position, verbose=False)
            if len(remaining_inds) == 0:
                continue

            cur_X = cur_X[remaining_inds]
            X.extend(cur_X)

        return X
