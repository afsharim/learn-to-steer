import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tqdm import trange
from datetime import datetime
import os
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import random


def set_seed(seed_value=0):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_to_onehot(data):
    if isinstance(data, torch.Tensor):
        data = data.long()
        n_train = data.shape[0]
        n_class = int(data.max()) + 1
        data_onehot = torch.zeros(n_train, n_class, device=data.device)
        data_onehot[torch.arange(n_train, device=data.device), data] = 1
        return data_onehot
    else:
        data = data.astype(int)
        n_train = data.shape[0]
        n_class = int(data.max()) + 1
        data_onehot = np.zeros((n_train, n_class))
        data_onehot[np.arange(n_train), data] = 1
        return data_onehot


def cross_cov(x, y):
    return 1 / x.shape[0] * x.T @ y



def get_gamma(x_train):
    if x_train.shape[0] > 10_000:
        idx = torch.randperm(x_train.shape[0])[:10_000]
        x_train = x_train[idx]
    dist_mat = torch.cdist(x_train, x_train, p=2)
    row, col = torch.triu_indices(dist_mat.shape[0], dist_mat.shape[1], offset=1)
    dist_mat = dist_mat[row, col]
    return 0.5 / dist_mat.median().square()



class RandomFourierFeatures:
    def __init__(self, dx, gamma=0.25, drff=1000, use_sine=True, device='cpu', resample=True):
        gamma = torch.scalar_tensor(gamma)
        self.dx = dx
        self.device = device
        self.omega = torch.sqrt(2 * gamma)
        self.const = torch.sqrt(2 / torch.scalar_tensor(drff)).to(device)

        if use_sine:
            self.drff = drff // 2
            if resample:
                self.apply_rff = self._resample_sine
            else:
                self._sine_sampler()
                self.apply_rff = self._sine_cosine_features
            return

        self.drff = drff
        if resample:
            self.apply_rff = self._resample_bias
        else:
            self._bias_sampler()
            self.apply_rff = self._cosine_bias_features

    def _sine_sampler(self):
        self.w = (torch.randn(self.dx, self.drff) * self.omega).to(self.device)

    def _bias_sampler(self):
        self.w = (torch.randn(self.dx, self.drff) * self.omega).to(self.device)
        self.b = (torch.rand(self.drff) * torch.pi * 2).to(self.device)

    def _sine_cosine_features(self, x):
        x_rff = x @ self.w
        return self.const * torch.hstack([torch.sin(x_rff), torch.cos(x_rff)])

    def _cosine_bias_features(self, x):
        x_rff = x @ self.w
        return self.const * torch.cos(x_rff + self.b)

    def _resample_sine(self, x):
        self._sine_sampler()
        return self._sine_cosine_features(x)

    def _resample_bias(self, x):
        self._bias_sampler()
        return self._cosine_bias_features(x)

    def __call__(self, x):
        return self.apply_rff(x)