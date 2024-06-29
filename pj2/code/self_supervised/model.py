import torch
import torch.nn as nn

# 定义SimCLR模型
class SimCLR(nn.Module):
    def __init__(self, base_encoder):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder
        in_features = base_encoder.fc.in_features
        out_features = base_encoder.fc.out_features
        
        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, 128)
        )
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z