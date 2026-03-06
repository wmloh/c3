import torch
import gpytorch


class SimpleGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ContextualGPBandit:
    def __init__(self, X_buffer, y_buffer, num_refit_epochs=1, num_init_fit_epochs=10, noise_std=1.0, ucb_beta=2.0, device='cpu'):
        self.device = device
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise=torch.tensor(noise_std**2, device=device)
        ).to(device)
        self.gp = SimpleGPRegressionModel(X_buffer, y_buffer, self.likelihood).to(self.device)
        self.ucb_beta = ucb_beta
        self.num_refit_epochs = num_refit_epochs

        self.X_buffer = X_buffer
        self.y_buffer = y_buffer

        self.fit(num_init_fit_epochs)

    def fit(self, num_epochs):
        likelihood = self.likelihood
        model = self.gp

        X_train = self.X_buffer
        y_train = self.y_buffer

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(num_epochs):
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train).sum()
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

    def select_action(self, candidate_xs):
        xs = torch.tensor(candidate_xs)
        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.gp(xs))
            mean = preds.mean
            std = preds.stddev
            ucb = mean + (self.ucb_beta ** 0.5) * std
        best_arm = int(torch.argmax(ucb).item())
        return best_arm, xs[best_arm].numpy()

    def update(self, chosen_x, reward, remove_factor=1):
        x = torch.tensor(chosen_x, dtype=torch.float32).to(self.device).unsqueeze(0)
        y = torch.tensor([reward], dtype=torch.float32).to(self.device)

        self.X_buffer = torch.cat([self.X_buffer[remove_factor:], x], dim=0)
        self.y_buffer = torch.cat([self.y_buffer[remove_factor:], y], dim=0)

        self.gp.set_train_data(self.X_buffer, self.y_buffer, strict=False)

        self.fit(self.num_refit_epochs)
