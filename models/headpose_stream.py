# models/headpose_stream.py

import torch
import torch.nn as nn

class HeadPoseEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, embed_dim=256, num_layers=1):
        super(HeadPoseEncoder, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=False)

        self.fc = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        """
        x: [B, 60, 3] → yaw, pitch, roll per frame
        """
        lstm_out, _ = self.lstm(x)  # [B, 60, hidden_dim]
        out = self.fc(lstm_out)     # [B, 60, embed_dim]
        return out
