# models/assemble_model.py

# models/assemble_model.py

import torch.nn as nn

# ✅ use package-relative imports (safer from inside the package)
from .visual_stream   import VisualEncoder
from .audio_stream    import AudioEncoder
from .lipsync_stream  import LipSyncStream
from .headpose_stream import HeadPoseEncoder
from .mamba_block     import MambaBlock
from .gating_module   import GatingModule
from .fusion_module   import FusionModule
from .classifier      import Classifier


class MultiModalDeepfakeModel(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # 1) Encoders
        self.face_enc   = VisualEncoder(embed_dim=embed_dim)                  # [B, 60, D]
        self.audio_enc  = AudioEncoder(embed_dim=embed_dim)                   # [B, 24, D]
        self.lip_enc    = LipSyncStream(img_embed_dim=embed_dim,
                                        audio_embed_dim=embed_dim,
                                        fusion_dim=embed_dim)                 # [B, 60, D]
        self.head_enc   = HeadPoseEncoder(embed_dim=embed_dim)                # [B, 60, D]

        # 2) Sequence modeling via Mamba (all at D=embed_dim)
        self.face_mamba  = MambaBlock(input_dim=embed_dim)                    # [B, 60, D]
        self.audio_mamba = MambaBlock(input_dim=embed_dim)                    # [B, 24, D]
        self.lip_mamba   = MambaBlock(input_dim=embed_dim)                    # [B, 60, D]
        self.head_mamba  = MambaBlock(input_dim=embed_dim)                    # [B, 60, D]

        # 3) Gating / Fusion / Classifier
        self.gate   = GatingModule(input_dim=embed_dim)                       # -> 4x [B, D]
        self.fusion = FusionModule(input_dim=embed_dim, num_layers=2)         # [B, D]
        self.cls    = Classifier(input_dim=embed_dim)                         # [B, 2]

    def forward(self, face, audio_mel, lips, audio_mel_lip, head_pose):
        """
        face           : [B, 60, 3, 224, 224]
        audio_mel      : [B, 1, 64, 96]
        lips           : [B, 60, 3, 112, 112]
        audio_mel_lip  : [B, 1, 64, 96]  (reused)
        head_pose      : [B, 60, 3]
        """
        # Encoders
        v = self.face_enc(face)               # [B, 60, D]
        a = self.audio_enc(audio_mel)         # [B, 24, D]
        l = self.lip_enc(lips, audio_mel_lip) # [B, 60, D]
        h = self.head_enc(head_pose)          # [B, 60, D]

        # Per-modality sequence modeling
        v = self.face_mamba(v)                # [B, 60, D]
        a = self.audio_mamba(a)               # [B, 24, D]
        l = self.lip_mamba(l)                 # [B, 60, D]
        h = self.head_mamba(h)                # [B, 60, D]

        # Confidence gating → vectors [B, D]
        v_g, a_g, l_g, h_g = self.gate([v, a, l, h])

        # Cross-modality fusion
        fused = self.fusion([v_g, a_g, l_g, h_g])  # [B, D]

        # Classification
        logits = self.cls(fused)                    # [B, 2]
        return logits
