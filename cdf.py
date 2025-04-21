import numpy as np
from scipy import stats
import torch


class CDF:
    def __init__(self, x):
        self.cdf = stats.ecdf(np.array(x).flatten()).cdf
        self.q = self.cdf.quantiles
        self.p = self.cdf.probabilities

    def __call__(self, val):
        return np.interp(val, self.q, self.p)


    def inverse(self, val):
        return np.interp(val, self.p, self.q)
