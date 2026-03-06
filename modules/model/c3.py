import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from tqdm import trange
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from modules.data.bootstrapper import Bootstrapper, MultiBootstrapper
from modules.model.losses import ECELoss, EPS


class C3(nn.Module):

    def __init__(self,
                 layer_nums: list[int],
                 X_init: Optional[torch.Tensor] = None,
                 y_init: Optional[torch.Tensor] = None,
                 sigma: float = 0.5,
                 weight_factor: float = 1.,
                 clear_buffer: bool = False,
                 min_alpha_beta: float = 0.1,
                 seed: Optional[int] = None):
        """
        Args:
            layer_nums: List of hidden dimensions in the MLP
            X_init: Initial features to condition on
            y_init: Initial labels to condition
            sigma: RBF bandwidth
            weight_factor: Multiplicative factor on alpha and beta parameters
            clear_buffer: Computes importance weights upon construction
            min_alpha_beta: Minimum alpha and beta values
            seed: Random seed
        """
        super().__init__()

        assert sigma > 0., "sigma must be positive"
        assert weight_factor > 0., "weight_factor must be positive"
        assert min_alpha_beta >= 0, "Minimum alpha/beta values must be non-negative"

        layers = list()
        for l1, l2 in zip(layer_nums[:-1], layer_nums[1:]):
            layers.append(nn.Linear(l1, l2))
            layers.append(nn.Softplus())
        layers.pop()

        self.project = nn.Sequential(*layers)
        self.X_init = torch.zeros((1, layer_nums[0])) if X_init is None else X_init
        self.y_init = torch.zeros((1, 1)) if y_init is None else y_init

        self.X_buff = None
        self.y_buff = None
        self.w_buff = None
        self.rng = None

        self.sigma = sigma
        self.weight_factor = weight_factor
        self.min_alpha_beta = min_alpha_beta

        if clear_buffer:  # computes importance weights
            self.clear_buffer(seed=seed)

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Module forward function. Predicts an IWKR estimate.
            For online bandits, you should use `infer_batch` instead.

        Args:
            Q: Raw vectors of queries (B, 1, D_in)
            K: Raw vector of reference data  # (N, D_in)
            V: Labels of reference data  # (N, 1)

        Returns:
            A tuple of:
            * Mean estimate of each query  (B,)
            * Standard deviation of each query from the Beta posterior distribution  (B,)
        """
        Q_proj = self.project(Q)  # (B, 1, D)
        K_proj = self.project(K)  # (N, D)

        # RBF kernel weight
        dist = F.pairwise_distance(Q_proj, K_proj)  # (B, N)
        S = torch.exp(-dist ** 2 / (2 * self.sigma ** 2 + EPS))  # (B, N)
        eta = S.sum(dim=-1)  # (B,)

        # importance weight computation
        L = torch.exp(-torch.cdist(K_proj, K_proj) ** 2 / (2 * self.sigma ** 2)).sum(dim=-1)  # (N,)
        w = 1 / L  # (N,)

        # Beta parameter estimation
        joint_score = S * w  # (B, N)
        mu_hat = (joint_score * V.T).sum(dim=-1) / joint_score.sum(dim=-1)  # (B,)
        pos = eta * mu_hat + self.min_alpha_beta  # (B,)
        neg = eta * (1 - mu_hat) + self.min_alpha_beta  # (B,)

        mean = mu_hat  # (B,)
        variance = (pos * neg) / ((pos + neg) ** 2 * (pos + neg + 1) + EPS)  # (B,)

        return mean, torch.sqrt(variance)

    def infer_batch(self,
                    Q: torch.Tensor,
                    K: Optional[torch.Tensor] = None,
                    V: Optional[torch.Tensor] = None,
                    w: Optional[torch.Tensor] = None,
                    sigma: float = None,
                    weight_factor: float = None,
                    matrix_size: int | float = 1e6) \
            -> tuple[np.array, np.array, np.array, np.array, np.array]:
        """
        Primary inference function - constructs a posterior Beta distribution and samples the best action.
            If K is defined, V and w must also be defined. 

        `self.clear_buffer` should be called prior to calling this function, 
            as the importance weights may not have been precomputed yet.

        Args:
            Q: Query feature embeddings (B, D)
            K: If defined, reference feature embeddings to condition on (N, D)
            V: If defined, reference labels to condition on (N, 1)
            w: If defined, importance weight to use for reference data
            sigma: If defined, overrides the default RBF bandwidth
            weight_factor: If defined, overrides the default multiplicative factor on alpha and beta parameters
            matrix_size: Size of matrix that can be fit into memory

        Returns:
            A tuple of 5 elements:
            * Best sampled action based on the Beta posterior
            * Mean estimate from the Beta posterior
            * Updated importance weight vector based on sampled action
            * Standard deviation from the Beta posterior
            * Eta of the queries (sum of kernel weights)
        """
        # overriding default values
        if sigma is None:
            sigma = self.sigma
        if weight_factor is None:
            weight_factor = self.weight_factor

        # load buffer data if not overriden
        if K is None:
            K = self.X_buff
            V = self.y_buff
            w = self.w_buff

        # embeds query features
        with torch.no_grad():
            Q = self.project(Q).unsqueeze(dim=1).numpy()

        X_ref = K.numpy()
        y_ref = V.numpy().reshape(-1)
        w_ref = w.numpy()

        # forming batches (if necessary) to avoid OOM error
        N = X_ref.shape[1]
        B = len(Q)
        batch_size = int(np.ceil(min(matrix_size / B, N)))

        X_batches = np.split(X_ref, np.arange(batch_size, len(X_ref), batch_size))
        y_batches = np.split(y_ref, np.arange(batch_size, len(y_ref), batch_size))
        w_batches = np.split(w_ref, np.arange(batch_size, len(w_ref), batch_size))

        # computing Beta parameters
        pos = 0.
        neg = 0.
        eta = 0.

        w_new = list()

        for X_batch, y_batch, w_batch in zip(X_batches, y_batches, w_batches):  # (M, D), (M,), (M,)
            score_batch = np.exp(-np.linalg.norm(X_batch - Q, axis=-1) ** 2 / (2 * sigma ** 2 + EPS))  # (B, M)

            w_new.append((1 / (1 / w_batch + score_batch.sum(axis=0))))

            eta += score_batch.sum(axis=-1)  # (B,)
            pos += (score_batch * w_batch * y_batch).sum(axis=-1)  # (B,)
            neg += (score_batch * w_batch * (1. - y_batch)).sum(axis=-1)  # (B,)

        total = pos + neg  # (B,)
        pos = eta * pos / (total + EPS) + self.min_alpha_beta  # (B,)
        neg = eta * neg / (total + EPS) + self.min_alpha_beta  # (B,)

        pos *= weight_factor
        neg *= weight_factor

        # computing mean, standard deviation, Beta sample, action and new importance weight
        mean = pos / (pos + neg + EPS)  # (B,)
        variance = (pos * neg) / ((pos + neg) ** 2 * (pos + neg + 1) + EPS)  # (B,)

        sampled_reward = self.rng.beta(pos + 1, neg + 1, size=mean.shape)  # (B,)
        action = np.argmax(sampled_reward)  # scalar

        w_new = np.concatenate(w_new + [1 / (eta + 1)[[action]]])  # (N+1,)

        return action, mean, w_new, np.sqrt(variance), eta

    def store_buffer(self,
                     X_query: torch.Tensor,
                     y_query: torch.Tensor,
                     importance_weights: Optional[torch.Tensor] = None) -> None:
        """
        Append new data into the reference data buffer.
            `importance_weights` should be passed in, otherwise it will
             have to recompute based on existing reference dataset.
            The updated importance weight output from `infer_batch` should be 
            used as an argument to `importance_weights`.

        Args:
            X_query: New raw features (not embedded)
            y_query: New labels
            importance_weights: If defined, stores new importance weights
        """
        # embeds new features and stores in buffer
        with torch.no_grad():
            self.X_buff = torch.cat((self.X_buff, self.project(X_query)))

        self.y_buff = torch.cat((self.y_buff, y_query))

        # recomputes importance weights if necessary
        if importance_weights is None:
            self.w_buff = 1 / torch.concat(
                [torch.exp(-torch.cdist(self.X_buff, X_batch) ** 2 / (2 * self.sigma ** 2 + EPS)).sum(dim=0)
                 for X_batch in torch.split(self.X_buff, 512)])
        else:
            self.w_buff = importance_weights

    def clear_buffer(self,
                     seed: Optional[int] = None,
                     chuck_size: int = 512):
        """
        Reset buffer to the original state, i.e. the X_init and y_init passed in during initialization.
            Recomputes importance weight stored in the buffer based on the original state.
            Also, re-creates a NumPy random generator state.

        Args:
            seed: NumPy random generator state
            chuck_size: Chunk size used when recomputing importance weights to avoid OOM error
        """
        # embeds initial feature set
        with torch.no_grad():
            self.X_buff = self.project(self.X_init.clone())
        self.y_buff = self.y_init.clone()

        # computes importance weights
        self.w_buff = 1 / torch.concat(
            [torch.exp(-torch.cdist(self.X_buff, X_batch) ** 2 / (2 * self.sigma ** 2 + EPS)).sum(dim=0)
             for X_batch in torch.split(self.X_buff, chuck_size)])

        # re-initializes the NumPy random generator
        self.rng = np.random.RandomState(seed=seed)

    def pop_buffer(self,
                   indices: np.array,
                   chunk_size: int = 512):
        """
        Remove samples from the reference data buffer.

        Args:
            indices: Indices of samples to remove
            chunk_size: Chunk size used when recomputing importance weights to avoid OOM error
        """
        # simply deletion of features and label in the reference data
        indices = [indices] if isinstance(indices, int) else indices
        X_out = np.delete(self.X_buff, indices, axis=0)
        y_out = np.delete(self.y_buff, indices, axis=0)

        # efficiently recomputes importance weights
        score = torch.concat([
            torch.exp(-torch.linalg.norm(X - self.X_buff[indices].unsqueeze(dim=1), dim=-1) ** 2 / (2 * self.sigma ** 2)) for X in
            torch.split(X_out, chunk_size)
        ], dim=1)

        w = 1 / (1 / np.delete(self.w_buff, indices) - score.sum(dim=0))

        # updates object variables
        self.X_buff = X_out
        self.y_buff = y_out
        self.w_buff = w

    def fit(self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            device: str | torch.device,
            loss_coef: dict[str, float | int],
            X_val: torch.Tensor,
            y_val: torch.Tensor,
            epochs: int = 10,
            batch_size: int = 16,
            M_ece: int = 5,
            base_ratio: float = 0.6,
            usage_ratio: float = 1.0,
            lr: float = 8e-3,
            explr_gamma: float = 0.99,
            val_base_ratio: float = 0.8,
            val_seed: Optional[int] = None,
            pick_best_val: bool = True,
            plot: bool = True,
            save_plot_name: Optional[str] = None,
            tqdm_pbar: bool = True) -> None:
        """
        Train phi (embedding model) based on the data in a contrastive self-supervised fashion.

        Args:
            X_train: Raw training features
            y_train: Training labels
            device: PyTorch device
            loss_coef: A dictionary containing the loss coefficients for "bce" and "ece"
            X_val: Raw validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Training batch size
            M_ece: Number of bins to use when computing ECE loss
            base_ratio: Proportion of training data to be used as reference data during self-supervised learning
            usage_ratio: Proportion of training to be used as a whole (resampled per epoch)
            lr: Adam learning rate
            explr_gamma: Gamma parameter of exponential learning rate scheduler
            val_base_ratio: Proportion of validation data to be used as reference data during self-supervised learning
            val_seed: If given, make reference-query splits deterministic during validation
            pick_best_val: Keep on the best embedding model based on validation score
            plot: If true, plot the training results 
            save_plot_name: Path to where the training plot will be saved to
            tqdm_pbar: Uses tqdm progress bar to show progress
        """
        self.to(device)
        self.train()

        # neural network training objects
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = ExponentialLR(optim, gamma=explr_gamma)
        ece_loss_fn = ECELoss(M_ece)

        coef_bce = loss_coef["bce"]
        coef_ece = loss_coef["ece"]

        # organizing validation data
        val_dataset = Bootstrapper(X_val, y_val, usage_ratio=1, base_ratio=val_base_ratio, seed=val_seed)
        val_dl = DataLoader(val_dataset, batch_size=2 * batch_size, shuffle=False)
        X_val_ref, y_val_ref = val_dataset.get_ref_data()
        X_val_ref = X_val_ref.to(device)
        y_val_ref = y_val_ref.to(device).view(-1)

        # collectors of training statistics
        train_bce = list()
        train_ece = list()

        val_bce = list()
        val_ece = list()

        best_loss = torch.inf
        best_state_dict = self.state_dict()

        # training loop
        pbar = trange(epochs) if tqdm_pbar else range(epochs)
        for e in pbar:
            cur_bce_loss = 0
            cur_ece_loss = 0

            # create a new reference-query train split each epoch
            dataset = Bootstrapper(X_train, y_train, base_ratio=base_ratio, usage_ratio=usage_ratio, seed=e)
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # store the reference data (unchanged within a single epoch)
            X_ref, y_ref = dataset.get_ref_data()
            X_ref = X_ref.to(device)
            y_ref = y_ref.to(device).view(-1)

            # batch-wise training
            self.train()
            for X_query, y_query in dl:
                X_query = X_query.to(device)
                y_query = y_query.to(device)

                # training with IWKR predictions
                mean, stddev = self(X_query, X_ref, y_ref.view(-1, 1))
                mean = mean.view(-1, 1)

                # loss, backprop and gradient descent
                loss_bce = F.binary_cross_entropy(mean, y_query)
                loss_ece = ece_loss_fn(mean, y_query)

                loss = coef_bce * loss_bce + coef_ece * loss_ece

                cur_bce_loss += loss_bce.item() * len(y_query)
                cur_ece_loss += loss_ece.item() * len(y_query)

                optim.zero_grad()
                loss.backward()
                optim.step()
            scheduler.step()

            cur_val_bce_loss = 0
            cur_val_ece_loss = 0

            # batch-wise validation
            self.eval()
            with torch.no_grad():
                for X_query, y_query in val_dl:
                    X_query = X_query.to(device)
                    y_query = y_query.to(device)

                    # IWKR predictions
                    mean, stddev = self(X_query, X_val_ref, y_val_ref.view(-1, 1))
                    mean = mean.view(-1, 1)

                    loss_bce = F.binary_cross_entropy(mean, y_query)
                    loss_ece = ece_loss_fn(mean, y_query)

                    cur_val_bce_loss += loss_bce.item() * len(y_query)
                    cur_val_ece_loss += loss_ece.item() * len(y_query)

            # keep the best
            if cur_val_bce_loss < best_loss:
                best_loss = cur_val_bce_loss
                best_state_dict = self.state_dict()

            # storing training and validation statistics
            train_bce.append(cur_bce_loss / len(dataset))
            train_ece.append(cur_ece_loss / len(dataset))

            val_bce.append(cur_val_bce_loss / len(val_dataset))
            val_ece.append(cur_val_ece_loss / len(val_dataset))

            # update progress bar message
            if tqdm_pbar:
                pbar.set_description(f"avg loss = {train_bce[-1]:.4f} | avg val loss = {val_bce[-1]:.4f}")

        if tqdm_pbar:
            pbar.close()

        # replaces latest model state with the best model state
        if pick_best_val:
            self.load_state_dict(best_state_dict)

        # plot statistics
        if plot:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            ax1.plot(train_bce, label="transductive CE", color="tab:blue")
            ax1.plot(train_ece, label="calibration error", alpha=0.4, color="tab:orange")
            ax1.plot(val_bce, "--", label="[V] transductive CE", color="tab:blue")
            ax1.plot(val_ece, "--", label="[V] calibration error", alpha=0.4, color="tab:orange")
            ax1.set_title("Unweighted loss")
            ax1.legend()

            ax2.plot(np.array(train_bce) * coef_bce, label="transductive CE", color="tab:blue")
            ax2.plot(np.array(train_ece) * coef_ece, label="calibration error", alpha=0.4, color="tab:orange")
            ax2.plot(np.array(val_bce) * coef_bce, "--", label="[V] transductive CE", color="tab:blue")
            ax2.plot(np.array(val_ece) * coef_ece, "--", label="[V] calibration error", alpha=0.4, color="tab:orange")
            ax2.set_title("Effective loss (weighted by coefficients)")
            ax2.legend()

            plt.tight_layout()
            if save_plot_name is not None:
                plt.savefig(save_plot_name, dpi=250)
                plt.close()

        self.cpu()
        self.eval()

    def fit_time_partitioned(self,
                             X_train: list[torch.Tensor],
                             y_train: list[torch.Tensor],
                             device: str | torch.device,
                             loss_coef: dict[str, float | int],
                             X_val: list[torch.Tensor],
                             y_val: list[torch.Tensor],
                             epochs: int = 10,
                             batch_size: int = 16,
                             M_ece: int = 5,
                             base_ratio: float = 0.6,
                             usage_ratio: float = 1,
                             lr: float = 8e-3,
                             explr_gamma: float = 0.99,
                             val_base_ratio: float = 0.8,
                             val_seed: Optional[int] = None,
                             pick_best_val: bool = False,
                             plot: bool = True,
                             save_plot_name: Optional[str] = None,
                             tqdm_pbar: bool = True):
        """
        Train phi (embedding model) based on the data in a contrastive self-supervised fashion
            where the data is partitioned by time interval.

        Args:
            X_train: Time-partitioned raw training features
            y_train: Time-partitioned training labels
            device: PyTorch device
            loss_coef: A dictionary containing the loss coefficients for "bce" and "ece"
            X_val: Time-partitioned raw validation features
            y_val: Time-partitioned validation labels
            epochs: Number of training epochs
            batch_size: Training batch size
            M_ece: Number of bins to use when computing ECE loss
            base_ratio: Proportion of training data to be used as reference data during self-supervised learning
            usage_ratio: Proportion of training to be used as a whole (resampled per epoch)
            lr: Adam learning rate
            explr_gamma: Gamma parameter of exponential learning rate scheduler
            val_base_ratio: Proportion of validation data to be used as reference data during self-supervised learning
            val_seed: If given, make reference-query splits deterministic during validation
            pick_best_val: Keep on the best embedding model based on validation score
            plot: If true, plot the training results 
            save_plot_name: Path to where the training plot will be saved to
            tqdm_pbar: Uses tqdm progress bar to show progress
        """
        self.to(device)
        self.train()

        # neural network training objects
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = ExponentialLR(optim, gamma=explr_gamma)
        ece_loss_fn = ECELoss(M_ece)

        coef_bce = loss_coef["bce"]
        coef_ece = loss_coef["ece"]

        # organizing validation data
        mboot_val = MultiBootstrapper(X_val, y_val, base_ratio=val_base_ratio, seed=val_seed, batch_size=2 * batch_size, shuffle=False)
        boot_val_dl = mboot_val.get_dl()

        # collectors of training statistics
        train_bce = list()
        train_ece = list()

        val_bce = list()
        val_ece = list()

        best_loss = torch.inf
        best_state_dict = self.state_dict()

        # training loop
        pbar = trange(epochs) if tqdm_pbar else range(epochs)
        for e in pbar:
            cur_bce_loss = 0
            cur_ece_loss = 0

            # create a new reference-query train split each epoch
            mboot = MultiBootstrapper(X_train, y_train, usage_ratio=usage_ratio, base_ratio=base_ratio,
                                      seed=e, batch_size=batch_size, shuffle=True)
            boot_dl = mboot.get_dl()

            # batch-wise training
            self.train()
            for dl, (X_ref, y_ref) in boot_dl:
                X_ref = X_ref.to(device)
                y_ref = y_ref.to(device).view(-1)

                for X_query, y_query in dl:
                    X_query = X_query.to(device)
                    y_query = y_query.to(device)

                    # training with IWKR predictions
                    mean, _ = self(X_query, X_ref, y_ref.view(-1, 1))
                    mean = mean.view(-1, 1)

                    # loss, backprop and gradient descent
                    loss_bce = F.binary_cross_entropy(mean, y_query)
                    loss_ece = ece_loss_fn(mean, y_query)

                    loss = coef_bce * loss_bce + coef_ece * loss_ece

                    cur_bce_loss += loss_bce.item() * len(y_query)
                    cur_ece_loss += loss_ece.item() * len(y_query)

                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                scheduler.step()

            cur_val_bce_loss = 0
            cur_val_ece_loss = 0

            # batch-wise validation
            self.eval()
            with torch.no_grad():
                for dl, (X_val_ref, y_val_ref) in boot_val_dl:
                    X_val_ref = X_val_ref.to(device)
                    y_val_ref = y_val_ref.to(device).view(-1)

                    for X_query, y_query in dl:
                        X_query = X_query.to(device)
                        y_query = y_query.to(device)

                        # IWKR predictions
                        mean, _ = self(X_query, X_val_ref, y_val_ref.view(-1, 1))
                        mean = mean.view(-1, 1)

                        loss_bce = F.binary_cross_entropy(mean, y_query)
                        loss_ece = ece_loss_fn(mean, y_query)

                        cur_val_bce_loss += loss_bce.item() * len(y_query)
                        cur_val_ece_loss += loss_ece.item() * len(y_query)

            # keep the best
            if cur_val_bce_loss < best_loss:
                best_loss = cur_val_bce_loss
                best_state_dict = self.state_dict()

            # storing training and validation statistics
            train_bce.append(cur_bce_loss / mboot.total_length)
            train_ece.append(cur_ece_loss / mboot.total_length)

            val_bce.append(cur_val_bce_loss / mboot_val.total_length)
            val_ece.append(cur_val_ece_loss / mboot_val.total_length)

            # update progress bar message
            if tqdm_pbar:
                pbar.set_description(f"avg loss = {train_bce[-1]:.4f} | avg val loss = {val_bce[-1]:.4f}")

        if tqdm_pbar:
            pbar.close()

        # replaces latest model state with the best model state
        if pick_best_val:
            self.load_state_dict(best_state_dict)

        # plot statistics
        if plot:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            ax1.plot(train_bce, label="transductive CE", color="tab:blue")
            ax1.plot(train_ece, label="calibration error", alpha=0.4, color="tab:orange")
            ax1.plot(val_bce, "--", label="[V] transductive CE", color="tab:blue")
            ax1.plot(val_ece, "--", label="[V] calibration error", alpha=0.4, color="tab:orange")
            ax1.set_title("Unweighted loss")
            ax1.legend()

            ax2.plot(np.array(train_bce) * coef_bce, label="transductive CE", color="tab:blue")
            ax2.plot(np.array(train_ece) * coef_ece, label="calibration error", alpha=0.4, color="tab:orange")
            ax2.plot(np.array(val_bce) * coef_bce, "--", label="[V] transductive CE", color="tab:blue")
            ax2.plot(np.array(val_ece) * coef_ece, "--", label="[V] calibration error", alpha=0.4, color="tab:orange")
            ax2.set_title("Effective loss (weighted by coefficients)")
            ax2.legend()

            plt.tight_layout()
            if save_plot_name is not None:
                plt.savefig(save_plot_name, dpi=250)

        self.cpu()
        self.eval()
