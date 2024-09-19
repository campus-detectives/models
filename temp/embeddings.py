import torch
import torchvision
from torch import nn
from torchvision.models.resnet import resnet18
from fastai.vision.learner import create_body

class embeddings(nn.Module):
    def __init__(self, backbone="resnet18", input_size=(1,3,224,224)):
        super().__init__()

        
        # Create a backbone network from the pretrained models provided in torchvision.models 
        if backbone in torchvision.models.__dict__:
            self.backbone=create_body(torchvision.models.__dict__[backbone](pretrained=True, progress=True),pretrained=True, n_in=3, cut=-2)
        elif backbone in timm.list_models(pretrained=True):
            self.backbone=create_body(timm.create_model(backbone,pretrained=True),pretrained=True, n_in=3,cut=-2)
        else:
            raise Exception("No model named {} exists in torchvision.models or timm models.".format(backbone))
                  
        
        self.flatten=nn.Flatten()

        self.encoder=nn.Sequential(
            self.backbone,
            self.flatten
        )

        self.flattened_features=self.encoder(torch.rand(input_size)).shape[1]
 
        self.last_block=nn.Sequential(
            nn.Linear(self.flattened_features,1024),
            nn.ReLU(inplace=True)
            #nn.BatchNorm1d(1024)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        return(self.last_block(self.encoder(x)))

    def comparator(self,input1,target):
        
        output = torch.cat((input1, target), dim=1)
        output = self.fc(output)
        return output