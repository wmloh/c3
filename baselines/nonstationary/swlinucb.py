import numpy as np

from collections import deque


class SWUCB:
    def __init__(self, d, w, R, L, S, lam=1.0, delta=1.0):
        self.d = d
        self.w = w
        self.R = R
        self.L = L
        self.S = S
        self.lam = lam

        self.V = lam * np.eye(d)    
        self.V_inv = np.linalg.inv(self.V)
        self.window_X = deque(maxlen=w)
        self.window_Y = deque(maxlen=w) 
        self.t = 0

        self.conf_const = R * np.sqrt(d * np.log((1 + (w * L * L)/lam) / delta)) + np.sqrt(lam) * S

    def select_action(self, D_t):
        self.t += 1

        if len(self.window_X) == 0:
            theta_hat = np.zeros(self.d)
        else:
            X_mat = np.vstack(self.window_X)  
            Y_vec = np.array(self.window_Y) 

            theta_hat = self.V_inv.dot(X_mat.T.dot(Y_vec))

        best_idx = None
        best_val = -np.inf
        for i, x in enumerate(D_t):
            x = x.reshape(-1)
            mean_est = x.dot(theta_hat)
            bonus = np.sqrt(x.dot(self.V_inv).dot(x)) * self.conf_const
            val = mean_est + bonus
            if val > best_val:
                best_val = val
                best_idx = i
        return best_idx, D_t[best_idx]

    def update(self, x_t, y_t, batched=False):
        if batched:
            self.window_X.extend(x_t)
            self.window_Y.extend(y_t)
        else:
            self.window_X.append(x_t)
            self.window_Y.append(y_t)

        X_mat = np.vstack(self.window_X)
        self.V = self.lam * np.eye(self.d) + X_mat.T.dot(X_mat)
        self.V_inv = np.linalg.inv(self.V)  

