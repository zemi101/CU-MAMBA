# models/fusion_module.py

import torch
import torch.nn as nn
from .mamba_block import MambaBlock   # uses settings inside MambaBlock

class FusionModule(nn.Module):
    """
    Cross-Mamba fusion over modality embeddings
    -------------------------------------------
    • Input : list[Tensor] → 4 modality vectors  [B, D] each
    • Stack : [B, M=4, D]  (treat modalities like a short sequence)
    • Mamba : stacked MambaBlocks capture interactions (lip↔audio, face↔head-pose …)
    • Pool  : mean-pool across modalities  → fused vector [B, D]
    """
    def __init__(self, input_dim: int = 256, num_layers: int = 2):
        super().__init__()
        # Each MambaBlock handles its own dim projection & (optional) local pretrained loading
        self.layers = nn.ModuleList(
            [MambaBlock(input_dim=input_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(input_dim)
        # Optional small projection after pooling (keeps dim = input_dim)
        self.proj_out = nn.Linear(input_dim, input_dim)

    def forward(self, embeddings_list):
        """
        embeddings_list : list of 4 tensors, each [B, D]
        returns         : fused vector        [B, D]
        """
        # Stack modalities as a length-4 sequence
        x = torch.stack(embeddings_list, dim=1)   # [B, 4, D]

        # Cross-modal interaction via Mamba layers
        for layer in self.layers:
            x = layer(x)                          # [B, 4, D]

        # Normalize & pool across modalities
        x = self.norm(x)                          # [B, 4, D]
        fused = x.mean(dim=1)                     # [B, D]

        # Light projection for mixing
        fused = self.proj_out(fused)              # [B, D]
        return fused
