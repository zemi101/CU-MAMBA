# models/visual_stream.py

import torch
import torch.nn as nn
import torchvision.models as models

class VisualEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_layers=2, num_heads=8):
        super(VisualEncoder, self).__init__()

        # Pretrained ResNet-50 (remove last fc layer)
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output: [B, 2048, 7, 7]

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce to [B, 2048, 1, 1] → [B, 2048]
        self.feature_proj = nn.Linear(2048, embed_dim)  # Project to desired embedding size

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [B, 60, 3, 224, 224] → B = batch size
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # Flatten time

        feats = self.backbone(x)        # [B*T, 2048, 7, 7]
        feats = self.pool(feats).squeeze(-1).squeeze(-1)  # [B*T, 2048]
        feats = self.feature_proj(feats)  # [B*T, embed_dim]
        feats = feats.view(B, T, -1)      # [B, 60, embed_dim]

        visual_embeddings = self.transformer(feats)  # [B, 60, embed_dim]
        return visual_embeddings
