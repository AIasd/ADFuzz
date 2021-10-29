import sys
# TBD: need to be made more portable
sys.path.insert(0, 'carla_project/src')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from dataset import get_dataset


data = get_dataset(sys.argv[1])
bank = list()

for rgb, topdown, points, target, actions, meta in data:
    bank.extend(target.numpy())

    if len(bank) > 4096:
        break

    print(len(bank))

all_points = np.stack(bank)

plt.plot(all_points[::10, 0], all_points[::10, 1], '.')

for k in [4, 8, 16]:
    clf = KMeans(k).fit(all_points)
    x = clf.cluster_centers_

    plt.plot(x[:, 0], x[:, 1], 'o', label='%d' % k)

plt.legend()
plt.show()

import pdb; pdb.set_trace()
