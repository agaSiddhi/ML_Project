"""Models for Text and Image Composition."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ConCatModule(nn.Module):
    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return x


class CompositionalLayer(nn.Module):
    def __init__(self):
        super(CompositionalLayer, self).__init__()
        print("In compositional layer")
        self.a = nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.gated_feature_composer = nn.Sequential(
            ConCatModule(), nn.BatchNorm1d(2 * 768), nn.ReLU(),
            nn.Linear(2 * 768, 768))
        self.res_info_composer = nn.Sequential(
            ConCatModule(), nn.BatchNorm1d(2 * 768), nn.ReLU(),
            nn.Linear(2 * 768, 2 * 768), nn.ReLU(),
            nn.Linear(2 * 768, 768))

    def compose_img_text(self, img_embed, text_embed):
        return self.compose_img_text_features(img_embed, text_embed)

    def compose_img_text_features(self, img_features, text_features):
        f1 = self.gated_feature_composer((img_features, text_features))
        f2 = self.res_info_composer((img_features, text_features))
        f = F.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1]
        return f
