from numpy import std, mean, sqrt
import itertools as it

from bisect import bisect_left
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as ss

from scipy.stats import ranksums

#correct if the population S.D. is expected to be equal for the two groups.
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)





def VD_A(treatment: List[float], control: List[float]):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    :param treatment: a numeric list
    :param control: another numeric list
    :returns the value estimate and the magnitude
    """
    m = len(treatment)
    n = len(control)

    if m != n:
        raise ValueError("Data d and f must have the same length")

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude




if __name__ == '__main__':
    # town03 3.75
    x3 = [156,138,148,167,154,173]
    y3 = [111,110,96,123,107,125]
    # town05 3.67
    x5 = [555,557,546,559,551,549]
    y5 = [517,506,479,523,521,506]
    # town01 10.52
    x1 = [669,675,673,676,679,681]
    y1 = [555,546,511,527,542,510]
    # town07 7.08
    x7 = [409,390,386,368,424,391]
    y7 = [286,281,284,293,292,263]

    x = x3
    y = y3
    print(x, y)
    print ("cohen d = ", cohen_d(x, y))
    print('Wilcoxon rank-sum statistics p value', ranksums(x, y))
    print('Vargha-Delaney effect size', VD_A(x, y))
