import numpy as np
def four_modes(x):
    centers = np.array([(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)])
    for i in range(centers.shape[0]):
        if np.linalg.norm(x-centers[i]) < 0.3:
            return [-1]
    return [1]

def four_modes_gmm(x):
    from scipy.stats import multivariate_normal
    centers = np.array([(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)])
    sum = 0
    for i, center in enumerate(centers):
        vi = multivariate_normal.pdf(x, mean=center, cov=0.05*np.eye(2))
        sum += vi
    return [sum]

def himmelblau(x):
    x0 = x[0]*5
    x1 = x[1]*5
    f = (x0**2+x1-11)**2 + (x0+x1**2-7)**2
    f = -f+10
    print('x0', x0, 'x1', x1, 'f', f)
    return [f]

def rectangle(x):
    from scipy.stats import multivariate_normal
    f = multivariate_normal.pdf(x[:1], mean=(0), cov=0.05*np.eye(1))

    # print('x', x, 'f', f)
    return [f]

synthetic_function_dict = {'four_modes': four_modes, 'four_modes_gmm': four_modes_gmm, 'himmelblau': himmelblau, 'rectangle': rectangle}
