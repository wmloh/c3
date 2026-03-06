import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class DataConverter:
    def __init__(self, X, y, num_arms=None):
        self.X_base = X
        self.y_base = y
        if num_arms is None:
            self.num_arms = int(y.max()) + 1
        else:
            self.num_arms = num_arms

        self.X = None
        self.y = None
        self.i = None
        self.max_iter = None

    def get_pretraining_split(self, proportion=0.2, reward_prop=None, seed=None):
        if proportion >= 1:
            X_out = self.X_base
            y_out = self.y_base
        else:
            X, X_out, y, y_out = train_test_split(self.X_base, self.y_base, test_size=proportion, random_state=seed)
            self.X_base = X
            self.y_base = y

        rng = np.random.RandomState(seed=seed)
        if reward_prop is None:
            sampled_actions = rng.choice(self.num_arms, size=len(y_out))
        else:
            assert isinstance(reward_prop, float) and 0 < reward_prop < 1
            sampled_actions = y_out[:, 0].copy()
            num_incorrect = int((1 - reward_prop) * len(sampled_actions))
            error_shifts = rng.choice(self.num_arms - 1, size=num_incorrect) + 1
            index_shifts = rng.choice(len(y_out), size=num_incorrect, replace=False)
            sampled_actions[index_shifts] = (sampled_actions[index_shifts] + error_shifts) % self.num_arms

        rewards = (y_out[:, 0] == sampled_actions).astype(np.float32)

        return X_out, sampled_actions.astype(int), rewards, y_out[:, 0]

    def reset(self, max_iter=np.inf, seed=None):
        self.max_iter = max_iter

        X, y = shuffle(self.X_base, self.y_base, random_state=seed)
        self.X = X
        self.y = y
        self.i = 0

        return self.X[0], np.hstack([np.tile(self.X[0], (self.num_arms, 1)), np.eye(self.num_arms)])

    def take_action(self, action):
        reward = int(self.y[self.i, 0] == action)
        return reward, 1 - reward

    def next(self):
        self.i += 1
        return self.X[self.i], np.hstack([np.tile(self.X[self.i], (self.num_arms, 1)), np.eye(self.num_arms)])

    def __len__(self):
        if self.X_base is not None:
            return len(self.X_base)
        else:
            return -1
