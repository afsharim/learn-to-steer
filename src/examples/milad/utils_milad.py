import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def cross_cov(x, y):
    return 1 / x.shape[0] * x.T @ y

def convert_to_onehot(data):
    data = data.astype(int)
    n_train = data.shape[0]
    n_class = int(data.max()) + 1
    data_onehot = np.zeros((n_train, n_class))
    data_onehot[np.arange(n_train), data] = 1
    return data_onehot



class RandomFourierFeatures:
    def __init__(
        self, dx, gamma=0.25, drff=1000, use_sine=True, device="cpu", resample=True
    ):
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
    
    
def get_gamma(x_train):
    if x_train.shape[0] > 10_000:
        idx = torch.randperm(x_train.shape[0])[:10_000]
        x_train = x_train[idx]
    dist_mat = torch.cdist(x_train, x_train, p=2)
    row, col = torch.triu_indices(dist_mat.shape[0], dist_mat.shape[1], offset=1)
    dist_mat = dist_mat[row, col]
    return 0.5 / dist_mat.median().square()




import torch
import torch.linalg as LA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def centering_matrix(n):
    return torch.eye(n, device=DEVICE) - torch.ones(n, n, device=DEVICE) / n


def label_kernel(y):
    y = torch.tensor(y, device=DEVICE)
    L = (y.unsqueeze(1) == y.unsqueeze(0)).float()
    L += torch.eye(len(y), device=DEVICE)   # regularisation
    return L


def rbf_kernel(X, gamma=1.0):
    """X shape: (n, p)"""
    sq = (X ** 2).sum(dim=1)
    dist_sq = sq.unsqueeze(1) + sq.unsqueeze(0) - 2 * X @ X.T
    return torch.exp(-gamma * dist_sq.clamp(min=0))


def rbf_kernel_cross(X_train, X_test, gamma=1.0):
    """Returns shape (m, n)"""
    sq_tr = (X_train ** 2).sum(dim=1)
    sq_te = (X_test  ** 2).sum(dim=1)
    dist_sq = sq_te.unsqueeze(1) + sq_tr.unsqueeze(0) - 2 * X_test @ X_train.T
    return torch.exp(-gamma * dist_sq.clamp(min=0))


def supervised_pca(X_train, L, X_test=None, n_components=2):
    """
    X_train : (p, n)
    L       : (n, n)  — use label_kernel(y) for classification
    X_test  : (p, m)  — optional
    returns Z_train (d, n), Z_test (d, m) or None, U (p, d)
    """
    X_train = X_train.to(DEVICE).float()
    # X_train_mean = X_train.mean(dim=0)
    # X_train = X_train - X_train_mean
    # if X_test is not None:
    #     X_test = X_test - X_train_mean
    L       = L.to(DEVICE).float()
    p, n    = X_train.shape

    H = centering_matrix(n)
    Q = X_train @ H @ L @ H @ X_train.T
    Q = (Q + Q.T) / 2                          # enforce symmetry

    _, U = LA.eigh(Q)                           # ascending order
    U    = U[:, -n_components:].flip(dims=[1])  # take top-d, descending

    Z_train = U.T @ X_train
    Z_test  = U.T @ X_test.to(DEVICE).float() if X_test is not None else None

    return Z_train, Z_test, U


def kernel_supervised_pca(K_train, L, K_test=None, n_components=2):
    """
    K_train : (n, n)  — use rbf_kernel(X_train)
    L       : (n, n)  — use label_kernel(y)
    K_test  : (m, n)  — use rbf_kernel_cross(X_train, X_test)
    returns Z_train (d, n), Z_test (d, m) or None, b (n, d)
    """
    K_train = K_train.to(DEVICE).float()
    L       = L.to(DEVICE).float()
    n       = K_train.shape[0]

    H = centering_matrix(n)
    Q = K_train @ H @ L @ H @ K_train
    Q = (Q + Q.T) / 2

    # Solve generalised eigenproblem Q b = λ K b via whitening
    K_reg = (K_train + 1e-6 * torch.eye(n, device=DEVICE))
    D, V  = LA.eigh(K_reg)
    W     = V * (1.0 / D.sqrt())               # whitening matrix (n, n)

    Q_white        = (W.T @ Q @ W)
    Q_white        = (Q_white + Q_white.T) / 2
    _, b_white     = LA.eigh(Q_white)
    b_white        = b_white[:, -n_components:].flip(dims=[1])

    b = W @ b_white                             # transform back (n, d)

    Z_train = b.T @ K_train
    Z_test  = b.T @ K_test.to(DEVICE).float().T if K_test is not None else None

    return Z_train, Z_test, b




def make_rff_projector(p, rff_dim=512, gamma=1.0):
    
    W = torch.randn(p, rff_dim, device=DEVICE) * (2 * gamma) ** 0.5
    b = torch.rand(rff_dim, device=DEVICE) * 2 * torch.pi
    def project(X):
        X = X.to(DEVICE).float()   # ← this line fixes it
        return torch.cos(X @ W + b) * (2 / rff_dim) ** 0.5
    return project