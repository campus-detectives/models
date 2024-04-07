import torch
import torchvision
from torch import nn
from torchvision.models.resnet import resnet18
from fastai.vision.learner import create_body

class emeddings(nn.Module):
    def __init__(self):
        super().__init__()
        model=create_body(resnet18(),pretrained=True, n_in=224, cut=-2)
        self.fc = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )


    def forward(self,x):
        return(model(x))

    def comparator(self,input1,target):
        
        output = torch.cat((input1, target), dim=1)
        output = self.fc(output)
        return output
