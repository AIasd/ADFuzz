from itertools import combinations_with_replacement
import numpy as np
import torch
import torch.nn as nn
import os

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def class_combinations(c, n, m=np.inf):
    """Generates an array of n-element combinations where each element is one of
    the c classes (an integer). If m is provided and m < n^c, then instead of all
    n^c combinations, m combinations are randomly sampled.
    Arguments:
        c {int} -- the number of classes
        n {int} -- the number of elements in each combination
    Keyword Arguments:
        m {int} -- the number of desired combinations (default: {np.inf})
    Returns:
        np.ndarry -- An [m x n] or [n^c x n] array of integers in [0, c)
    """

    if m < c ** n:
        # randomly sample combinations
        return np.random.randint(c, size=(int(m), n))
    else:
        p_c = combinations_with_replacement(np.arange(c), n)
        return np.array(list(iter(p_c)), dtype=int)


def H(x, eps=1e-6):
    """Compute the element-wise entropy of x
    Arguments:
        x {torch.Tensor} -- array of probabilities in (0,1)
    Keyword Arguments:
        eps {float} -- prevent failure on x == 0
    Returns:
        torch.Tensor -- H(x)
    """
    return -(x + eps) * torch.log(x + eps)


def hasnan(x):
    return torch.isnan(x).any()


def remove_occurrences_from_list(l, items):
    print("l, items", l, items)
    return list(
        np.setdiff1d(
            np.array(l, dtype=int), np.array(items, dtype=int), assume_unique=True
        )
    )


def move_data(indices, from_subset, to_subset):
    from_subset.indices = remove_occurrences_from_list(from_subset.indices, indices)
    if isinstance(to_subset.indices, list):
        to_subset.indices.extend(indices)
    elif isinstance(to_subset.indices, np.ndarray):
        to_subset.indices = np.concatenate([to_subset.indices, np.array(indices)])


class Acquirer:
    """Base class for acquisition function"""

    def __init__(self, batch_size, device=None):
        self.batch_size = batch_size
        self.processing_batch_size = 100
        if not device:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = device

    @staticmethod
    def score(model, x):
        """Parallezied acquisition scoring function
        Arguments:
            model {nn.Module} -- the NN
            x {torch.Tensor} -- datapoints to evaluate
        Returns:
            [torch.Tensor] -- a vector of acquisition scores
        """
        return torch.zeros(len(x))

    def select_batch(self, model, pool_data, unique_len=0, uncertainty_conf=False):
        # score every datapoint in the pool under the model
        pool_loader = torch.utils.data.DataLoader(
            pool_data,
            batch_size=self.processing_batch_size,
            pin_memory=True,
            shuffle=False,
        )
        scores = torch.zeros(len(pool_data)).to(self.device)
        for batch_idx, (data, _) in enumerate(pool_loader):
            end_idx = batch_idx + data.shape[0]
            scores[batch_idx:end_idx] = self.score(
                model, data.to(self.device).float(), uncertainty_conf=uncertainty_conf
            )
        # print('select_batch scores:', scores)
        if unique_len > 0:
            scores[:unique_len] += torch.max(scores)
        best_local_indices = torch.argsort(scores)[-self.batch_size :]
        best_global_indices = np.array(pool_data.indices)[
            best_local_indices.cpu().numpy()
        ]
        return best_global_indices

    @staticmethod
    def calculate_conf(model, x, k=100):
        torch.manual_seed(0)
        with torch.no_grad():
            # take k monte-carlo samples of forward pass w/ dropout
            Y = torch.stack([model.predict_proba(x) for i in range(k)], dim=1)
            print(Y.shape)
            s = Y[:, :, 1]
            s = torch.mean(s, axis=1)
            print(s.shape)
            return s

    @staticmethod
    def norm_weighted_sum(s, s_conf, weights=None):
        if isinstance(s, np.ndarray) and isinstance(s_conf, np.ndarray):
            is_numpy = True
        else:
            is_numpy = False
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import MinMaxScaler

        stand = StandardScaler()
        scaler = MinMaxScaler()
        # print('s', s)
        # print('s_conf', s_conf)
        if is_numpy:
            s_combined = np.stack([s, s_conf])
        else:
            s_combined = torch.stack([s, s_conf]).cpu().detach().numpy()
        s_combined = np.swapaxes(s_combined, 0, 1)
        # print('s_combined before', s_combined)
        stand.fit(s_combined)
        s_combined = stand.transform(s_combined)
        # print('s_combined middle', s_combined)
        scaler.fit(s_combined)
        s_combined = scaler.transform(s_combined)
        # print('s_combined after', s_combined)
        if not weights:
            weights = np.expand_dims(np.array([1, 1]), axis=1)
        s = np.dot(s_combined, weights).squeeze()

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if is_numpy:
            return s
        else:
            return torch.from_numpy(s).to(device).float()


class BALD(Acquirer):
    def __init__(self, batch_size, device=None):
        super(BALD, self).__init__(batch_size, device)

    @staticmethod
    def score(model, x, k=100, uncertainty_conf=False):
        # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]
        torch.manual_seed(0)
        with torch.no_grad():
            # take k monte-carlo samples of forward pass w/ dropout
            Y = torch.stack([model.predict_proba(x) for i in range(k)], dim=1)
            H1 = H(Y.mean(axis=1)).sum(axis=1)
            H2 = H(Y).sum(axis=(1, 2)) / k

            s = H1 - H2

        if uncertainty_conf:
            s_conf = super(BALD, BALD).calculate_conf(model, x)
            s = super(BALD, BALD).norm_weighted_sum(s, s_conf)

        return s


class BUGCONF(Acquirer):
    def __init__(self, batch_size, device=None):
        super(BUGCONF, self).__init__(batch_size, device)

    @staticmethod
    def score(model, x, k=100, uncertainty_conf=False):
        # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]
        torch.manual_seed(0)
        with torch.no_grad():
            # take k monte-carlo samples of forward pass w/ dropout
            Y = torch.stack([model.predict_proba(x) for i in range(k)], dim=1)
            # print(Y.shape)
            s = Y[:, :, 1]
            s = torch.mean(s, axis=1)
            # print(s.shape)
            return s


class Random(Acquirer):
    def __init__(self, batch_size, device=None, uncertainty_conf=False):
        super(Random, self).__init__(batch_size, device)

    @staticmethod
    def score(model, _, uncertainty_conf=False):
        return np.random.rand()


class BatchBALD(Acquirer):
    def __init__(self, batch_size, device=None):
        super(BatchBALD, self).__init__(batch_size, device)
        self.m = 1e4  # number of MC samples for label combinations
        self.num_sub_pool = (
            500  # number of datapoints in the subpool from which we acquire
        )

    def select_batch(
        self, model, pool_data, k=100, unique_len=0, uncertainty_conf=False
    ):
        # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]

        c = 2  # number of classes

        # performing BatchBALD on the whole pool is very expensive, so we take
        # a random subset of the pool.
        num_extra = len(pool_data) - self.num_sub_pool
        if num_extra > 0:
            # sub_pool_data, _ = torch.utils.data.random_split(pool_data, [self.num_sub_pool, num_extra])
            sub_pool_data = torch.utils.data.Subset(
                pool_data, np.arange(self.num_sub_pool)
            )
        else:
            # even if we don't have enough data left to split, we still need to
            # call random_splot to avoid messing up the indexing later on
            # sub_pool_data, _ = torch.utils.data.random_split(pool_data, [len(pool_data), 0])
            sub_pool_data = torch.utils.data.Subset(
                pool_data, np.arange(len(pool_data))
            )
        # forward pass on the pool once to get class probabilities for each x
        data_list = []
        with torch.no_grad():
            pool_loader = torch.utils.data.DataLoader(
                sub_pool_data,
                batch_size=self.processing_batch_size,
                pin_memory=True,
                shuffle=False,
            )
            pool_p_y = torch.zeros(len(sub_pool_data), c, k)
            for batch_idx, (data, _) in enumerate(pool_loader):
                end_idx = batch_idx + data.shape[0]
                torch.set_deterministic(True)
                torch.manual_seed(0)
                pool_p_y[batch_idx:end_idx] = torch.stack(
                    [
                        model.predict_proba(data.to(self.device).float())
                        for i in range(k)
                    ],
                    dim=1,
                ).permute(0, 2, 1)

                data_list.append(data)

        data_list = torch.cat(data_list)
        print(data_list.size())
        # this only need to be calculated once so we pull it out of the loop
        H2 = (H(pool_p_y).sum(axis=(1, 2)) / k).to(self.device)

        # get all class combinations
        c_1_to_n = class_combinations(c, self.batch_size, self.m)

        # tensor of size [m x k]
        p_y_1_to_n_minus_1 = None

        # store the indices of the chosen datapoints in the subpool
        best_sub_local_indices = []
        # create a mask to keep track of which indices we've chosen
        remaining_indices = torch.ones(len(sub_pool_data), dtype=bool).to(self.device)

        if uncertainty_conf:
            s_conf = super(BatchBALD, BatchBALD).calculate_conf(
                model, data_list.to(self.device).float()
            )

        for n in range(self.batch_size):
            # tensor of size [N x m x l]
            p_y_n = pool_p_y[:, c_1_to_n[:, n], :].to(self.device)
            # tensor of size [N x m x k]
            p_y_1_to_n = (
                torch.einsum("mk,pmk->pmk", p_y_1_to_n_minus_1, p_y_n)
                if p_y_1_to_n_minus_1 is not None
                else p_y_n
            )

            # and compute the left entropy term
            H1 = H(p_y_1_to_n.mean(axis=2)).sum(axis=1)
            # scores is a vector of scores for each element in the pool.
            # mask by the remaining indices and find the highest scoring element
            scores = H1 - H2
            # print('scores1', scores)
            if uncertainty_conf:
                scores = super(BatchBALD, BatchBALD).norm_weighted_sum(scores, s_conf)
            # print('scores2', scores)
            if unique_len > 0:
                scores[:unique_len] += torch.max(scores)
            # print('scores3', scores)
            # print(scores)
            best_local_index = torch.argmax(
                scores - np.inf * (~remaining_indices)
            ).item()
            # print('remaining_indices', remaining_indices)
            # print('best_local_index', best_local_index)
            # print(f'Best idx {best_local_index}')
            best_sub_local_indices.append(best_local_index)
            # save the computation for the next batch
            p_y_1_to_n_minus_1 = p_y_1_to_n[best_local_index]
            # remove the chosen element from the remaining indices mask
            remaining_indices[best_local_index] = False

        # we've subset-ed our dataset twice, so we need to go back through
        # subset indices twice to recover the global indices of the chosen data
        best_local_indices = np.array(sub_pool_data.indices)[best_sub_local_indices]
        best_global_indices = np.array(pool_data.indices)[best_local_indices]
        return best_global_indices


def init_centers(X, s_conf, K, uncertainty_conf):
    from sklearn.metrics import pairwise_distances
    from scipy import stats

    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.0] * len(X)
    cent = 0
    # print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)

        # print('Ddist before', Ddist)
        # print("s_conf", s_conf)
        # print("uncertainty_conf", uncertainty_conf)
        if uncertainty_conf:
            Ddist = super(BADGE, BADGE).norm_weighted_sum(Ddist, s_conf)

        # print('Ddist after', Ddist)

        from sklearn.preprocessing import normalize

        Ddist = normalize([Ddist], norm="l1").squeeze()
        # print("Ddist normalize", Ddist)
        # Fix original bug in code
        while True:
            customDist = stats.rv_discrete(
                name="custm", values=(np.arange(len(D2)), Ddist)
            )
            ind = customDist.rvs(size=1)[0]
            if ind not in indsAll:
                break
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    # gram = np.matmul(X[indsAll], X[indsAll].T)
    # val, _ = np.linalg.eig(gram)
    # val = np.abs(val)
    # vgt = val[val > 1e-2]

    indsAll_np = np.array(indsAll)
    from scipy.stats import rankdata

    s_conf_rank = (rankdata(s_conf) - 1).astype(int)
    print("init_centers")
    print("s_conf", s_conf)
    print("s_conf[indsAll_np]", s_conf[indsAll_np])
    print(
        "chosen rank",
        s_conf_rank[indsAll_np],
        "mean, std",
        np.mean(s_conf_rank[indsAll_np]),
        np.std(s_conf_rank[indsAll_np]),
    )
    print(
        "s_conf[indsAll_np] mean std",
        np.mean(s_conf[indsAll_np]),
        np.std(s_conf[indsAll_np]),
    )
    print("s_conf mean std", np.mean(s_conf), np.std(s_conf))

    return indsAll


class BADGE(Acquirer):
    def __init__(self, batch_size, device=None):
        super(BADGE, self).__init__(batch_size, device)

    def select_batch(self, model, pool_data, unique_len=0, uncertainty_conf=False):

        original_pool_data = pool_data
        inds = []
        if unique_len > 0:
            if unique_len < self.batch_size:
                inds = np.arange(unique_len)
                pool_data = torch.utils.data.Subset(
                    original_pool_data, np.arange(len(unique_len), len(pool_data))
                )

            elif unique_len < len(pool_data):
                pool_data = torch.utils.data.Subset(
                    original_pool_data, np.arange(len(unique_len))
                )

        criterion = nn.BCELoss()
        device = torch.device("cuda")

        y_extra_pred_list = []
        grad_list = []

        pool_loader = torch.utils.data.DataLoader(
            pool_data, batch_size=1, pin_memory=True, shuffle=False
        )

        for batch_idx, (data, _) in enumerate(pool_loader):
            x_extra = data.to(self.device).float()
            y_extra_pred = model(x_extra).squeeze()
            y_extra_hat = torch.round(y_extra_pred.detach())

            loss = criterion(y_extra_pred, y_extra_hat)
            loss.backward()

            y_extra_pred_list.append(y_extra_pred.cpu().detach().numpy())
            grad_list.append(model.fc_end.weight.grad.cpu().detach().numpy())

        s_conf = np.array(y_extra_pred_list)
        grad_np = np.array(grad_list).squeeze()

        # print(grad_np.shape, s_conf.shape)
        # print('grad_n', grad_np)
        # print('s_conf', s_conf)
        cur_inds = init_centers(grad_np, s_conf, self.batch_size, uncertainty_conf)
        # if unique_len > 0:
        #     scores[:unique_len] += torch.max(scores)

        if len(inds) > 0:
            print("cur_inds", inds)
            cur_inds += len(inds)
            inds = np.concatenate([inds, cur_inds])
        else:
            inds = cur_inds
        # print(len(inds), 'inds before', sorted(inds))
        print("len(np.unique(inds))", len(np.unique(inds)))
        inds = np.array(original_pool_data.indices)[inds]
        # print(len(inds), 'inds after', sorted(inds))
        print("len(np.unique(inds))", len(np.unique(inds)))
        return inds


def map_acquisition(acquisition_name):
    uncertainty_map = {
        "BUGCONF": BUGCONF,
        "Random": Random,
        "BALD": BALD,
        "BatchBALD": BatchBALD,
        "BADGE": BADGE,
    }
    return uncertainty_map[acquisition_name]
