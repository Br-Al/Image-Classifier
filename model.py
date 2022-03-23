
from collections import OrderedDict
from torch import nn
import torchvision

class Model():
    def __init__(self, arch = 'vgg19'):
        self.arch = arch
        self.model = getattr(torchvision.models, arch)(pretrained=True)
#         self.model = torch.hub.load(repo, arch, pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        
    def set_classifier(self, hidden_unit):
        h1 = int(hidden_unit/2)
        h2 = hidden_unit - h1
        
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, h1)),
                          ('drop', nn.Dropout(p=0.2)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(h1, h2)),
                          ('drop', nn.Dropout(p=0.2)),
                          ('relu', nn.ReLU()),
                          ('fc5', nn.Linear(h2, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        if self.arch == 'resnet':
            self.model.fc = classifier
        else:
            self.model.classifier = classifier
        return self.model
        
        
       