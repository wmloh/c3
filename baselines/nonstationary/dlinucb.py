import numpy as np


class DLinUCB:
    def __init__(self, d, lam=1.0, gamma=0.99, delta=0.1, L=1.0, S=1.0, sigma=1.0):
        self.d = d
        self.lam = lam
        self.gamma = gamma
        self.delta = delta
        self.L = L
        self.S = S
        self.sigma = sigma

        self.b = np.zeros((d, 1)) 
        self.V = lam * np.eye(d)  
        self.Ve = lam * np.eye(d)        
        self.theta_hat = np.zeros((d, 1))   

        self.t = 0

    def _compute_beta(self):
        t = self.t
        term1 = np.sqrt(self.lam * self.S)
        term2 = self.sigma * np.sqrt(2 * np.log(1.0/self.delta) +
                                     self.d * np.log(1 + (self.L**2 * (1 - self.gamma**(2*(t)))) / (self.lam * self.d * (1 - self.gamma**2))))
        beta = term1 + term2
        return beta

    def select_action(self, A_t):
        self.t += 1
        beta = self._compute_beta()

        V_inv = np.linalg.inv(self.V)

        Ve_inv = np.linalg.inv(self.Ve)

        ucb_values = []
        for i, a in enumerate(A_t):
            a = a.reshape((self.d, 1))
            mean_est = np.squeeze(a.T.dot(self.theta_hat))
            bonus = beta * np.sqrt(a.T.dot(V_inv).dot(self.V).dot(Ve_inv).dot(a))
            ucb = mean_est + bonus
            ucb_values.append(ucb)
        ucb_values = np.array(ucb_values).flatten()

        idx = int(np.argmax(ucb_values))
        return idx, A_t[idx].reshape((self.d, 1))

    def update(self, a_chosen, x_reward):

        a = a_chosen.reshape((self.d, 1))
        x = float(x_reward)

        self.V = self.gamma * self.V + a.dot(a.T) + (1 - self.gamma) * self.lam * np.eye(self.d)

        self.Ve = (self.gamma**2) * self.Ve + a.dot(a.T) + (1 - self.gamma**2) * self.lam * np.eye(self.d)

        self.b = self.gamma * self.b + x * a

        self.theta_hat = np.linalg.inv(self.V).dot(self.b)
