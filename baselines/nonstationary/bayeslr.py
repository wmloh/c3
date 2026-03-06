import numpy as np

from numpy.linalg import inv
from scipy.stats import norm as univariate_normal


class BayesLR:
    def __init__(self, feat_dim: int, ci_width: float = 1.96, noise_var: float = 1, lamb_var: float = 1,
                 projection_func=None):
        self.feat_dim = feat_dim

        if projection_func is None:
            self.projection_func = self.to_design_mat
            self.feat_dim += 1
        else:
            self.projection_func = projection_func

        self.sigma = np.eye(self.feat_dim) * lamb_var
        self.mu = np.zeros(self.feat_dim)

        self.prior_sigma = self.sigma.copy()
        self.prior_mu = self.mu.copy()

        self.ci_width = ci_width
        self.noise_var = noise_var
        self.lamb_var = lamb_var

    def __call__(self, feat: np.ndarray):
        feat_mat = self.projection_func(feat)

        pred_mean = feat_mat.dot(self.mu)
        pred_cov = feat_mat.dot(self.sigma.dot(feat_mat.T)) + self.noise_var

        pred_sd = np.sqrt(np.diag(pred_cov))

        return pred_mean, pred_sd, univariate_normal(loc=pred_mean.flatten(), scale=pred_cov)

    def to_design_mat(self, feat_vec: np.ndarray) -> np.ndarray:
        feat_mat = np.ones((len(feat_vec), self.feat_dim))
        feat_mat[:, 1:] = feat_vec
        return feat_mat

    def update_posterior(self, feat_vec: np.ndarray, y: np.ndarray):
        X = self.projection_func(feat_vec)

        self.sigma = inv(X.T.dot(X) / self.noise_var + inv(self.prior_sigma))
        self.mu = self.sigma.dot(inv(self.prior_sigma).dot(self.prior_mu) + X.T.dot(y) / self.noise_var)

    def replace_prior(self):
        self.prior_mu = self.mu.copy()
        self.prior_sigma = self.sigma.copy()

    def reset(self):
        self.sigma = np.eye(self.feat_dim) * self.lamb_var
        self.mu = np.zeros(self.feat_dim)


class BayesMultiLR:
    def __init__(self, feat_dim: int, out_dim: int = 1, ci_width: float = 1.96, noise_var: float = 1, lamb_var: float = 1,
                 projection_func=None):
        self.feat_dim = feat_dim

        if projection_func is None:
            self.projection_func = self.to_design_mat
            self.feat_dim += 1
        else:
            self.projection_func = projection_func

        self.sigma = np.eye(self.feat_dim) * lamb_var
        self.mu = np.zeros((self.feat_dim, out_dim))

        self.prior_sigma = self.sigma.copy()
        self.prior_mu = self.mu.copy()

        self.ci_width = ci_width
        self.noise_var = noise_var
        self.lamb_var = lamb_var

    def __call__(self, feat: np.ndarray):
        feat_mat = self.projection_func(feat)

        pred_mean = feat_mat.dot(self.mu)
        pred_cov = feat_mat.dot(self.sigma.dot(feat_mat.T)) + self.noise_var

        pred_sd = np.sqrt(np.diag(pred_cov))

        return pred_mean, pred_sd, None

    def to_design_mat(self, feat_vec: np.ndarray) -> np.ndarray:
        feat_mat = np.ones((len(feat_vec), self.feat_dim))
        feat_mat[:, 1:] = feat_vec
        return feat_mat

    def update_posterior(self, feat_vec: np.ndarray, y: np.ndarray):
        X = self.projection_func(feat_vec)

        self.sigma = inv(X.T.dot(X) / self.noise_var + inv(self.prior_sigma))
        self.mu = self.sigma.dot(inv(self.prior_sigma).dot(self.prior_mu) + X.T.dot(y) / self.noise_var)

    def replace_prior(self):
        self.prior_mu = self.mu.copy()
        self.prior_sigma = self.sigma.copy()

    def reset(self):
        self.sigma = np.eye(self.feat_dim) * self.lamb_var
        self.mu = np.zeros(self.feat_dim)
