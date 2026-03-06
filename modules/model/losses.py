import torch
import torch.nn as nn

EPS = 1e-5


class ECELoss(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M

    def forward(self, pred, label):
        bin_boundaries = torch.linspace(0, 1, self.M + 1).to(pred.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        confidences = torch.abs(pred - 0.5) + 0.5
        predicted_label = (pred >= 0.5).float()
        accuracies = (predicted_label == label).float()

        ece = torch.zeros(1).to(pred.device).float()
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = torch.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prob_in_bin = in_bin.float().mean()

            if prob_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
        return ece
