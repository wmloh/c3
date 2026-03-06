import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Obtained from: https://github.com/uclaml/NeuralUCB

class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class OnlineNeuralUCB:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100, t_threshold=2000, t_freq=100,
                 sliding_window=2000):
        self.func = Network(dim, hidden_size=hidden)#.cuda()
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,))#.cuda()
        self.nu = nu
        self.t = 0
        self.t_threshold = t_threshold
        self.t_freq = t_freq
        self.sliding_window = sliding_window

    def reset(self):
        self.t = 0

    def select_action(self, context):
        tensor = torch.from_numpy(context).float()#.cuda()
        mu = self.func(tensor)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))

            sample_r = fx.item() + sigma.item()

            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm, tensor[arm]

    def insert_data(self, context, reward, trim_excess):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)

        if trim_excess and len(self.context_list) > self.sliding_window:
            self.context_list = self.context_list[-self.sliding_window:]
            self.reward = self.reward[-self.sliding_window:]

    def update(self):
        t = self.t
        self.t += 1
        if (t >= self.t_threshold) and ((t % self.t_freq) != 0):
            return -1
        
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0

        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                delta = self.func(c) - r  # .cuda()
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length
