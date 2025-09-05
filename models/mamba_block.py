# models/mamba_block.py

import os
import torch
import torch.nn as nn

from models.mamba_setting import (
    USE_LOCAL_PRETRAINED,
    LOCAL_MAMBA,
    FREEZE_BACKBONE,
    CUSTOM_HIDDEN_SIZE,
)

try:
    from transformers import MambaModel, MambaConfig
except Exception as e:
    raise ImportError(
        "transformers is required for Mamba. Install with:\n"
        "  pip install transformers accelerate\n"
        f"Underlying import error: {e}"
    )


class MambaBlock(nn.Module):
    """
    Wrapper around HuggingFace Mamba that:
      - loads weights from a LOCAL folder (no internet) when USE_LOCAL_PRETRAINED is True
      - otherwise builds a small random-initialized Mamba (fast, no weights)
      - projects from input_dim -> hidden_size -> input_dim
      - returns the SAME shape as input: [B, T, input_dim]
    """

    def __init__(self, input_dim: int = 256):
        super().__init__()

        self.input_dim = input_dim

        if USE_LOCAL_PRETRAINED:
            local_path = LOCAL_MAMBA
            if not os.path.isdir(local_path):
                raise FileNotFoundError(
                    f"Mamba local_path not found: {local_path}\n"
                    f"Expected files: config.json + model.safetensors (or pytorch_model.bin)"
                )
            self.mamba = MambaModel.from_pretrained(local_path, local_files_only=True)
            hidden_size = self.mamba.config.hidden_size
        else:
            # Build a small random-initialized Mamba (fast)
            hs = CUSTOM_HIDDEN_SIZE if CUSTOM_HIDDEN_SIZE is not None else max(256, input_dim)
            cfg = MambaConfig(
                hidden_size=hs,
                num_hidden_layers=2,
                intermediate_size=4 * hs,
                max_position_embeddings=1024,
                use_cache=False,
            )
            self.mamba = MambaModel(cfg)
            hidden_size = hs

        # Project in/out so the rest of your model can stay at input_dim (e.g., 256)
        self.in_proj  = nn.Linear(input_dim, hidden_size) if hidden_size != input_dim else nn.Identity()
        self.out_proj = nn.Linear(hidden_size, input_dim) if hidden_size != input_dim else nn.Identity()

        if FREEZE_BACKBONE:
            for p in self.mamba.parameters():
                p.requires_grad = False

        self.pre_norm  = nn.LayerNorm(input_dim)
        self.post_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        x: [B, T, D_in]  where D_in == input_dim
        returns: [B, T, D_in]
        """
        x = self.pre_norm(x)
        x_h = self.in_proj(x)                                   # [B, T, hidden_size]
        y_h = self.mamba(inputs_embeds=x_h).last_hidden_state   # [B, T, hidden_size]
        y = self.out_proj(y_h)                                  # [B, T, input_dim]
        y = self.post_norm(y)
        return y
