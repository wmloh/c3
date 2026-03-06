import numpy as np

from typing import Dict, List, Tuple


class LinearIndexModel:
    def __init__(self, dim: int, lr: float = 0.01):
        self.theta = np.zeros(dim)
        self.lr = lr

    def predict(self, features: np.ndarray) -> float:
        return float(self.theta @ features)

    def update(self, features: np.ndarray, target: float):
        pred = self.predict(features)
        error = pred - target
        self.theta -= self.lr * error * features


class DynamicFeatureCRB:
    def __init__(
        self,
        dim,
        epsilon: float = 0.1,
        lambda_lr: float = 0.05,
        model_lr: float = 0.01,
    ):
        self.dim = dim

        self.epsilon = epsilon
        self.lambda_lr = lambda_lr

        self.model = LinearIndexModel(dim, lr=model_lr)
        self.lambdas = 0.0

        self.arms: Dict[int, np.ndarray] = {}

    def _joint_feature(self, context: int, arm_feature: np.ndarray) -> np.ndarray:
        return np.concatenate([context, arm_feature])

    def select_action(self, D_t: int) -> List[int]:
        scores = []

        for vec in D_t:
            score = self.model.predict(vec) - self.lambdas
            scores.append(score)

        if np.random.rand() < self.epsilon:
            idx = np.random.choice(len(scores))
        else:
            idx = [
                arm for arm, _ in sorted(
                    enumerate(scores), key=lambda x: x[1], reverse=True
                )
            ][0]

        return idx, D_t[idx]

    def update(
        self,
        context: int,
        reward: float,
        baseline: float = 0.0,
    ):
        target = reward - baseline
        self.model.update(context, target)
