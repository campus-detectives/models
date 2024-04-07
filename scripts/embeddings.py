import torch
import torchvision
from torch import nn
from torchvision.models.resnet import resnet18
from fastai.vision.learner import create_body

class embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=create_body(torchvision.models.resnet18(),pretrained=True, n_in=3, cut=-2)
        self.flatten=nn.Flatten()
        
        self.encoder=nn.Sequential(
            self.model,
            self.flatten,
            nn.Linear(25088,1024),
            nn.ReLU(inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        


    def forward(self,x):
        return(self.encoder(x))

    def comparator(self,input1,target):
        
        output = torch.cat((input1, target), dim=1)
        output = self.fc(output)
        return output