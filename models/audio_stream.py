# models/audio_stream.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    def __init__(self, input_shape=(64, 96), embed_dim=256, lstm_hidden=256, lstm_layers=1):
        super(AudioEncoder, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 32, 64, 96]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),                        # [B, 32, 32, 48]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 32, 48]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),                        # [B, 64, 16, 24]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# [B, 128, 16, 24]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 24))                # [B, 128, 1, 24]
        )

        self.fc = nn.Linear(128, embed_dim)             # Project channel dim to embedding dim
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, lstm_layers, batch_first=True)

    def forward(self, x):
        """
        x: [B, 1, 64, 96] - Log-Mel spectrogram
        """
        features = self.cnn(x)         # [B, 128, 1, 24]
        features = features.squeeze(2) # [B, 128, 24]
        features = features.permute(0, 2, 1)  # [B, 24, 128]

        features = self.fc(features)   # [B, 24, embed_dim]
        out, _ = self.lstm(features)  # [B, 24, lstm_hidden]
        return out  # sequence of audio embeddings
