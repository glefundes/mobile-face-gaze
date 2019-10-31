import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np 

from PIL import Image
from torchvision import transforms

class GazeNet(nn.Module):

    def __init__(self, device):    
        super(GazeNet, self).__init__()
        self.device = device
        self.preprocess = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model = models.mobilenet_v2(pretrained=True)
        model.features[-1] = models.mobilenet.ConvBNReLU(320, 256, kernel_size=1)
        self.backbone = model.features

        self.Conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.Conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.Conv3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )   
        self.fc_final = nn.Linear(512, 2)

        self._initialize_weight()
        self._initialize_bias()
        self.to(device)


    def _initialize_weight(self):
        nn.init.normal_(self.Conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.Conv2.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.Conv3.weight, mean=0.0, std=0.001)

    def _initialize_bias(self):
        nn.init.constant_(self.Conv1.bias, val=0.1)
        nn.init.constant_(self.Conv2.bias, val=0.1)
        nn.init.constant_(self.Conv3.bias, val=1)

    def forward(self, x):
        
        x = self.backbone(x)
        y = F.relu(self.Conv1(x))
        y = F.relu(self.Conv2(y))
        y = F.relu(self.Conv3(y))
        
        x = F.dropout(F.relu(torch.mul(x, y)), 0.5)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_final(x)

        return x

    def get_gaze(self, img):
        img = Image.fromarray(img)
        img = self.preprocess(img)[np.newaxis,:,:,:]
        x = self.forward(img.to(self.device))
        return x