import torchvision.models as models
import torch
from models.mlp_head import MLPHead
from models_ova import Distance_1D


class ResNet18(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
    
    
class Multi_ResNet18(torch.nn.Module):
    def __init__(self, flag_ova, *args, **kwargs):
        super(Multi_ResNet18, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        elif kwargs['name'] == 'wideresenet':
            resnet = models.wide_resnet50_2(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])
        
        if flag_ova:
            print("ova is training!")
            self.linear = Distance_1D(out_features=resnet.fc.in_features,
                                   num_classes = 10)
        else:
            self.linear = torch.nn.Linear(resnet.fc.in_features, 10, bias=True)
            
    def forward(self, x):
        h = self.encoder(x)
        h_proj = h.view(h.shape[0], h.shape[1])
        proj = self.projetion(h_proj)
        out = self.linear(h_proj)
        return proj, out
    
class CE(torch.nn.Module):
    def __init__(self):
        super(CE, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.linear = torch.nn.Linear(resnet.fc.in_features, 10, bias=True)
        
    def forward(self, x):
        return self.linear(x)

class OVA(torch.nn.Module):
    def __init__(self):
        super(OVA, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.linear = Distance_1D(out_features=resnet.fc.in_features,
                                   num_classes = 10)
        
    def forward(self, x):
        return self.linear(x)
