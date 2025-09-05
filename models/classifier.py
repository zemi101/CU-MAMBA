# models/classifier.py

import torch.nn as nn

class Classifier(nn.Module):
    """
    Simple MLP-based head: fused → 2-logit output (Real / Fake)
    """
    def __init__(self, input_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)          # logits [B, 2]
