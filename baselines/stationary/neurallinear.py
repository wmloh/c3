import numpy as np
import torch
import torch.nn as nn

from numpy.linalg import inv
from itertools import chain
from scipy.stats import multivariate_normal


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

    def to_design_mat(self, feat_vec: np.ndarray) -> np.ndarray:
        feat_mat = np.ones((len(feat_vec), self.feat_dim))
        feat_mat[:, 1:] = feat_vec
        return feat_mat

    def update_posterior(self, feat_vec: np.ndarray, y: np.ndarray):
        X = self.projection_func(feat_vec)

        self.sigma = inv(X.T.dot(X) / self.noise_var + inv(self.prior_sigma))
        self.mu = self.sigma.dot(inv(self.prior_sigma).dot(self.prior_mu) + X.T.dot(y) / self.noise_var)

    def sample_prediction(self, input_feat):
        rv = multivariate_normal(mean=self.mu, cov=self.sigma)
        sampled_weights = rv.rvs(1)

        return self.to_design_mat(input_feat) @ sampled_weights

    def replace_prior(self):
        self.prior_mu = self.mu.copy()
        self.prior_sigma = self.sigma.copy()

    def reset(self):
        self.sigma = np.eye(self.feat_dim) * self.lamb_var
        self.mu = np.zeros(self.feat_dim)


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)

    def fit(self, train_dl, num_epochs=10):
        model = self.model
        regressor = self.regressor

        optim = torch.optim.Adam(chain(model.parameters(), regressor.parameters()))
        loss_fn = nn.MSELoss()

        train_losses = list()

        for e in range(num_epochs):
            cur_loss = 0
            for data, target in train_dl:
                pred = regressor(model(data))
                loss = loss_fn(pred, target)

                optim.zero_grad()
                loss.backward()
                optim.step()

                cur_loss += loss.item() * len(data)
            train_losses.append(cur_loss / len(train_dl.dataset))

        return train_losses


class NeuralLinear:
    def __init__(self, input_dim, embed_dim):
        self.mlp = MLP(input_dim, embed_dim)
        self.lr = BayesLR(embed_dim)

    def fit(self, *args, **kwargs):
        return self.mlp.fit(*args, **kwargs)

    def update(self, X, y):
        with torch.no_grad():
            embed = self.mlp(torch.tensor(X, dtype=torch.float32)).numpy()
        self.lr.update_posterior(embed, y)
        self.lr.replace_prior()

    def pick_action(self, C_arms):
        C_arms = torch.tensor(C_arms, dtype=torch.float32)
        with torch.no_grad():
            embed = self.mlp(C_arms)
        pred = self.lr.sample_prediction(embed.numpy())

        return pred.argmax()
