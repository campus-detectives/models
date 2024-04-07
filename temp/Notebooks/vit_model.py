from torch import nn
import torch
import timm


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model=timm.create_model("vit_base_patch16_224.orig_in21k_ft_in1k",pretrained=True,scriptable=True)

    def forward(self, x):
        return torch.argmax(torch.softmax((self.model(x)),dim=1),dim=1).item()