import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models
from utils import save_net,load_net

class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features,features,kernel_size=1)

    def __make_weight(self,feature,scale_feature):
        weight_feature = feature - scale_feature
        return F.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.scales]
        weights = [self.__make_weight(feats,scale_feature) for scale_feature in multi_scales]
        overall_features = [(multi_scales[0]*weights[0]+multi_scales[1]*weights[1]+multi_scales[2]*weights[2]+multi_scales[3]*weights[3])/(weights[0]+weights[1]+weights[2]+weights[3])]+ [feats]
        bottle = self.bottleneck(torch.cat(overall_features, 1))
        return self.relu(bottle)

class CANNet2s(nn.Module):
    def __init__(self, load_weights=True):
        super(CANNet2s, self).__init__()
<<<<<<< HEAD
        self.context = ContextualModule(32, 32)
        # self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend_feat = [32, 32, 'M', 32, 32, 'M', 40, 40, 30, 'M', 32, 32, 32]
        self.backend_feat  = [40, 60, 40,40,20,10]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 64, batch_norm=True, dilation = True)
        self.output_layer = nn.Conv2d(64, 10, kernel_size=1)
=======
        # self.context = ContextualModule(32, 32)
        # self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend_feat = [32, 32, 'M', 32, 32, 'M', 40, 40, 30, 'M', 32, 32, 32]
        # self.backend_feat  = [40, 60, 40,40,20,10]
        self.backend_feat  = [40, 60, 40,40]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 64, batch_norm=True, dilation = True)
        self.output_layer = nn.Conv2d(40, 10, kernel_size=1)
>>>>>>> mobilecountx2
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2) 
        self.dropout = nn.Dropout(p=0.2)
        self.dropout8 = nn.Dropout(p=0.8)
        # customize 
        self.conv2dx10 = nn.Conv2d(3, 10, kernel_size=9, padding =4, dilation =1)
        self.conv2dx14 = nn.Conv2d(3, 14, kernel_size=7, padding =3, dilation =1)
        self.conv2dx16 = nn.Conv2d(3, 16, kernel_size=5, padding =2, dilation =1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)

    def forward(self,x_prev,x):
        x_prev1 = self.conv2dx10(x_prev)
        x_prev1 = self.relu(x_prev1)
        x_prev2 = self.conv2dx14(x_prev)
        x_prev2 = self.relu(x_prev2)
        x_prev3 = self.conv2dx16(x_prev)
        x_prev3 = self.relu(x_prev3)
        x_prev = torch.cat((x_prev1, x_prev2, x_prev3),1)

        x1 = self.conv2dx10(x)
        x1 = self.relu(x1)
        x2 = self.conv2dx14(x)
        x2 = self.relu(x2)
        x3 = self.conv2dx16(x)
        x3 = self.relu(x3)
        x = torch.cat((x1, x2, x3), 1)
        x_prev = self.frontend(x_prev)
        x = self.frontend(x)
        x = self.dropout(x)
        x_prev = self.dropout(x_prev)

        # x_prev = self.context(x_prev)
        # x = self.context(x)

        x = torch.cat((x_prev,x),1)

        x = self.backend(x)
        x = self.output_layer(x)
        x = self.relu(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels = 40,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        
    return nn.Sequential(*layers)

class CCNN(nn.Module):
    def __init__(self):
        self.CNN_feat = [40, 60 ,'M', 40,'M',20,10]
        self.Mid_cnn = make_layers(self.CNN_feat, in_channels=40)
        self.red_cnn = nn.Conv2d(3, 10, kernel_size=9)
        self.green_cnn = nn.Conv2d(3, 14, kernel_size=7)
        self.blue_cnn = nn.Conv2d(3, 16, kernel_size=5)

    def forward(self, x_prev, x):
        x_prev_red = self.red_cnn(x_prev)
        x_prev_green = self.green_cnn(x_prev)
        x_prev_blue = self.blue_cnn(x_prev)
        
        x_red = self.red_cnn(x)
        x_green = self.green_cnn(x)
        x_blue = self.blue_cnn(x)

        x_prev = nn.cat((x_prev_red,x_prev_green, x_prev_blue), dim =1)
        x = nn.cat((x_red, x_green, x_blue), dim = 1)

        x= self.Mid_cnn()

        