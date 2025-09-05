# models/gating_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingModule(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3, mc_passes=5):
        super(GatingModule, self).__init__()

        self.conf_fc = nn.Linear(input_dim, 1)
        self.uncertainty_fc = nn.Linear(input_dim, 1)

        self.dropout = nn.Dropout(dropout_rate)
        self.mc_passes = mc_passes

    def forward(self, features_list):
        """
        features_list: list of [B, T, D] tensors for each modality
        returns: list of gated [B, D] tensors (averaged across time)
        """
        gated_features = []
        conf_list, unc_list = [], []

        for x in features_list:
            # Average across time: [B, D]
            x_avg = x.mean(dim=1)

            # --- Confidence ---
            conf = torch.sigmoid(self.conf_fc(x_avg))  # [B, 1]
            conf_list.append(conf)

            # --- Uncertainty (MC Dropout) ---
            preds = []
            for _ in range(self.mc_passes):
                pred = self.dropout(self.uncertainty_fc(x_avg))  # [B, 1]
                preds.append(pred)

            preds = torch.stack(preds, dim=0)  # [T, B, 1]
            mean_pred = preds.mean(dim=0)     # [B, 1]
            var_pred = ((preds - mean_pred) ** 2).mean(dim=0)  # [B, 1]
            unc_list.append(var_pred)

        # Stack and compute weights
        conf_tensor = torch.stack(conf_list, dim=0)  # [M, B, 1]
        unc_tensor = torch.stack(unc_list, dim=0)    # [M, B, 1]

        alpha = torch.exp(conf_tensor - unc_tensor)
        alpha = alpha / torch.sum(alpha, dim=0, keepdim=True)  # [M, B, 1]

        # Apply gating
        for i in range(len(features_list)):
            x_avg = features_list[i].mean(dim=1)  # [B, D]
            x_gated = alpha[i] * x_avg            # [B, D]
            gated_features.append(x_gated)

        return gated_features  # List of [B, D]
