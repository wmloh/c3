import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tqdm import trange


class TwoTower(nn.Module):
    def __init__(self, user_encoder_layers, arm_encoder_layers):
        super().__init__()

        ulayers = list()
        for l1, l2 in zip(user_encoder_layers[:-1], user_encoder_layers[1:]):
            ulayers.append(nn.Linear(l1, l2))
            ulayers.append(nn.ReLU())
        ulayers.pop()

        alayers = list()
        for l1, l2 in zip(arm_encoder_layers[:-1], arm_encoder_layers[1:]):
            alayers.append(nn.Linear(l1, l2))
            alayers.append(nn.ReLU())
        alayers.pop()

        self.user_encoder = nn.Sequential(*ulayers)
        self.arm_encoder = nn.Sequential(*alayers)

    def forward(self, user, arms):
        user_emb = self.user_encoder(user)
        arms_emb = self.arm_encoder(arms)

        logits = torch.einsum("bd,bad->ba", user_emb, arms_emb)
        prob = torch.softmax(logits, dim=1)
        return prob

    def fit(self, train_dl, val_dl, device, epochs=25, lr=8e-4, plot=True, pick_best_val=True, save_plot_name=None):
        self.to(device)
        self.train()

        optim = torch.optim.Adam(self.parameters(), lr=lr)
        bce_loss_fn = nn.CrossEntropyLoss()

        best_loss = torch.inf
        best_state_dict = self.state_dict()

        train_bce = list()
        val_bce = list()

        for e in trange(epochs):
            cur_loss = 0

            self.train()
            for users, arms_list, labels in train_dl:
                loss = torch.Tensor([0.]).to(device)
                for user, arms, label in zip(users, arms_list, labels):
                    user = user.to(device)
                    arms = arms.to(device)
                    label = label.to(device)

                    pred = self(user, arms)
                    loss += bce_loss_fn(pred, label)

                loss /= len(users)

                optim.zero_grad()
                loss.backward()
                optim.step()

                cur_loss += loss.item() * len(users)

            train_bce.append(cur_loss / len(train_dl.dataset))

            cur_val_loss = 0

            self.eval()
            with torch.no_grad():
                for users, arms_list, labels in val_dl:
                    loss = torch.Tensor([0.]).to(device)
                    for user, arms, label in zip(users, arms_list, labels):
                        user = user.to(device)
                        arms = arms.to(device)
                        label = label.to(device)

                        pred = self(user, arms)
                        loss += bce_loss_fn(pred, label)

                    cur_val_loss += loss.item()

            if cur_val_loss < best_loss:
                best_loss = cur_val_loss
                best_state_dict = self.state_dict()

            val_bce.append(cur_val_loss / len(val_dl.dataset))

        if pick_best_val:
            self.load_state_dict(best_state_dict)

        if plot:
            plt.plot(train_bce, label="train")
            plt.plot(val_bce, label="val")
            plt.xlabel("Number of epochs")
            plt.ylabel("Cross entropy loss")
            plt.legend()

            plt.tight_layout()
            if save_plot_name is not None:
                plt.savefig(save_plot_name, dpi=200)
                plt.close()

        self.cpu()
        self.eval()
