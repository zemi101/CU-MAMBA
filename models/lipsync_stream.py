import torch
import torch.nn as nn
from torchvision import models

class LipSyncStream(nn.Module):
    def __init__(self, img_embed_dim=256, audio_embed_dim=256, fusion_dim=256):
        super(LipSyncStream, self).__init__()

        # 🟠 Lip CNN: ResNet18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.lip_cnn = nn.Sequential(*list(resnet.children())[:-2])  # [B*T, 512, 7, 7]
        self.lip_pool = nn.AdaptiveAvgPool2d((1, 1))                 # [B*T, 512, 1, 1]
        self.lip_proj = nn.Linear(512, img_embed_dim)                # [B*T, 512] → [B*T, 256]

        # 🟠 Audio LSTM: Input [B, 64, 96] → Mean → [B, audio_embed_dim]
        self.audio_lstm = nn.LSTM(input_size=96, hidden_size=audio_embed_dim, batch_first=True)
        self.audio_fc = nn.Linear(audio_embed_dim, fusion_dim)  # [B, 256] → [B, 256]

        # 🟠 Final similarity projection
        self.lip_fc = nn.Linear(1, fusion_dim)  # [B, 60, 1] → [B, 60, 256]

        # 🟠 Cosine similarity
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, lips, audio):
        """
        lips:  [B, 60, 3, 112, 112]
        audio: [B, 1, 64, 96]
        """
        B, T, C, H, W = lips.shape

        # === Lip CNN ===
        lips = lips.view(B * T, C, H, W)                      # [B*T, 3, 112, 112]
        lip_feats = self.lip_cnn(lips)                        # [B*T, 512, 7, 7]
        lip_feats = self.lip_pool(lip_feats).squeeze()        # [B*T, 512]
        lip_feats = self.lip_proj(lip_feats)                  # [B*T, 256]
        lip_feats = lip_feats.view(B, T, -1)                  # [B, 60, 256]

        # === Audio LSTM ===
        audio = audio.squeeze(1)                              # [B, 64, 96]
        audio_out, _ = self.audio_lstm(audio)                 # [B, 64, 256]
        audio_feat = audio_out.mean(dim=1)                    # [B, 256]
        audio_feat = self.audio_fc(audio_feat)                # [B, 256]
        audio_feat = audio_feat.unsqueeze(1).repeat(1, T, 1)  # [B, 60, 256]

        # === Cosine similarity ===
        sync_scores = self.cos_sim(lip_feats, audio_feat)     # [B, 60]

        # === Project similarity score to 256-D
        sync_proj = self.lip_fc(sync_scores.unsqueeze(-1))    # [B, 60, 1] → [B, 60, 256]

        return sync_proj
