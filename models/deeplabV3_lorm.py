import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

affine_par = True

def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif init_mode == 'xavier':
                nn.init.xavier_uniform_(m.weight.data)
            elif init_mode == 'contant':
                #print(m)
                nn.init.constant_(m.weight, 0)
            else:
                raise ValueError('Invalid init_mode {}'.format(init_mode))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module): # modified to adapt deeplabv3+
    def __init__(self, inplanes, dilations, BatchNorm, num_classes):
        super(ASPP, self).__init__()

        self.aspp1 = _ASPPModule(inplanes, num_classes, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm) # 6,12,18,24  v3: 1,6,12,18
        self.aspp2 = _ASPPModule(inplanes, num_classes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, num_classes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, num_classes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, num_classes, 1, stride=1, bias=False),
                                             BatchNorm(num_classes),
                                             nn.ReLU(inplace=True))
        self.prject = nn.Sequential(
            nn.Conv2d(5*num_classes, num_classes, 1, bias=False),
            BatchNorm(num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        # self.conv1 = nn.Conv2d(5*num_classes, num_classes, 1, bias=False)
        # self.bn1 = BatchNorm(num_classes)
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.prject(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = BatchNorm(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)

        padding = dilation
        self.conv2 = conv3x3(planes, planes, stride=1, padding=padding, dilation = dilation)
        self.bn2 = BatchNorm(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,BatchNorm=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = BatchNorm(planes, affine = affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation = dilation)
        self.bn2 = BatchNorm(planes, affine = affine_par)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4, affine = affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet_lorm(nn.Module):
    def __init__(self, block, layers, num_classes,BatchNorm=nn.BatchNorm2d):
        self.inplanes = 64
        self.low_channels = 256
        self.high_channels = 2048
        super(ResNet_lorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64, affine = affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0],BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2,BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4,BatchNorm=BatchNorm,dilations=[4,8,16])
        
        
        if block.__name__ == 'Bottleneck':
            self.layer_pred0 = self._make_pred_layer(ASPP, 2048, [1,6,12,18], BatchNorm, self.high_channels // 8) # train with scribble
            self.layer_lorm = LoRM(in_dims=2048,out_dims=2048)
        else:
            self.layer_pred0 = self._make_pred_layer(ASPP, 512, [1,6,12,18], BatchNorm, num_classes)
            self.layer_lorm = LoRM(in_dims=512,out_dims=512)
            
        self.reduce =nn.Sequential(nn.Conv2d(self.low_channels, 48, 1, bias=False),
                                    BatchNorm(48),
                                    nn.ReLU(True))
        self.fuse = nn.Sequential(nn.Conv2d(self.high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  BatchNorm(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  BatchNorm(256),
                                  nn.ReLU(True))
        self.classifier = nn.Conv2d(256, num_classes, 1, bias=True)
        
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,BatchNorm=nn.BatchNorm2d,dilations=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion,affine = affine_par))
        layers = []
        if dilations == None:
            layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample,BatchNorm=BatchNorm))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
        else:
            layers.append(block(self.inplanes, planes, stride,dilation=dilations[0], downsample=downsample,BatchNorm=BatchNorm))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilations[i], BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplane, dilation_series, BatchNorm, num_classes):
        return block(inplane, dilation_series, BatchNorm, num_classes)

    def forward(self, x,scribble,cam): # scribble: b,h,w 0,1,2..  cam: b,h,w 0,1,2...
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x_ = self.layer2(x1)
        x_ = self.layer3(x_)
        x4 = self.layer4(x_)
        out4 = self.layer_pred0(x4)
        out4 = F.interpolate(out4,size=x1.shape[-2:],mode='bilinear',align_corners=True) # b,21,117,117
        out1= self.reduce(x1)# b,48,117,117
        feat = torch.cat([out1,out4],dim=1)
        feat = self.fuse(feat)
        
        pred = self.classifier(feat)
        #
        att_loss = self.layer_lorm(x4,cam)
        return pred,att_loss
    def forward_eval(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x_ = self.layer2(x1)
        x_ = self.layer3(x_)
        x4 = self.layer4(x_)
        out4 = self.layer_pred0(x4)
        out4 = F.interpolate(out4,size=x1.shape[-2:],mode='bilinear',align_corners=True) # b,21,117,117
        out1= self.reduce(x1)# b,48,117,117
        feat = torch.cat([out1,out4],dim=1)
        feat = self.fuse(feat)
        
        pred = self.classifier(feat)
        return pred

class LoRM(nn.Module):
    def __init__(self,in_dims=2048,out_dims=2048):
        super(LoRM, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims if out_dims!= None else in_dims
        self.query_conv = nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=1)
        self.distance_func = self.mse_loss
        self.gamma = nn.Parameter(torch.ones(1))
        initialize_weights(self.modules(), init_mode='xavier')
    
    def forward(self,x,cam,mode='cam'): #  cam: b,h,w 0,1,2... #log/Jun07_13-34-40_ubuntu 有用！
        cam[cam == 255] = 0
        new_x = torch.zeros_like(x) # b,c,h,w
        x_shape = x.size()[2:4] # x: b,c,h,w
        B,C,H,W = x.size()
        cam = cam.unsqueeze(1).float() # b,h,w -> b,1,h,w
        cam = F.interpolate(cam,size=x_shape,mode='bilinear',align_corners=True) # scale to features map's shape
        # all_cls = torch.unique(cam)
        # for clsid in all_cls:
        #     if clsid == 255:
        #         continue
        curr_mask = torch.zeros_like(cam)
        curr_mask[cam>0] = 1 # b,1,h,w
        curr_mask = curr_mask.view(B,-1,H*W).permute(0,2,1) # b,1,h,w -> b,1,hw -> b,hw,1
        curr_query = self.query_conv(x)
        # curr_query = query*curr_mask # b,2048,h,w
        curr_query = curr_query.view(B,-1,H*W).permute(0,2,1) # b,c,h,w -> b,c,hw -> b,hw,c
        curr_key = self.key_conv(x) # b,2048,h,w
        curr_key = curr_key.view(B,-1,H*W) # b,c,h,w -> b,c,hw
        energy = torch.bmm(curr_query,curr_key) # b,hw(query),hw(key)
        curr_query_norm = curr_query.pow(2).sum(-1,keepdim=True).sqrt() # b,hw,c -> b,hw,1
        curr_key_norm = curr_key.pow(2).sum(1,keepdim=True).sqrt() # b,c,hw -> b,1,hw
        curr_norm = torch.bmm(curr_query_norm,curr_key_norm) # b,hw(qurey),hw(key)
        atten = energy / curr_norm
        atten = torch.softmax(atten,dim=-1) # b,hw(query),hw(key,softmaxed)
        atten = curr_mask * atten # b,hw(query,masked),hw(key,softmaxed)
        x_tmp = x.view(B,-1,H*W).permute(0,2,1) # b,c,h,w -> b,c,hw -> b,hw,c 
        new_x = new_x + torch.bmm(atten,x_tmp).permute(0,2,1).view(B,C,H,W)# b,hw(query,masked),c(weighted channel) -> b,c,hw -> b,c,h,w
        new_x = self.gamma * new_x # ones
        x_tmp = x_tmp.permute(0,2,1).view(B,-1,H,W) # b,hw,c -> b,c,hw -> b,c,h,w 
        loss = self.distance_func(x,new_x)
        return loss
    

    def mse_loss(self,x1,x2):
        result = (x1-x2)**2
        loss = torch.sum(result) / torch.count_nonzero(result)
        return loss
def Res_Deeplab(num_classes=21, layers=18):
    layers = int(layers)
    if layers == 18:
        model = ResNet_lorm(BasicBlock, [2, 2, 2, 2], num_classes,SynchronizedBatchNorm2d)
    elif layers == 34:
        model = ResNet_lorm(BasicBlock, [3, 4, 6, 3], num_classes,SynchronizedBatchNorm2d)
    elif layers == 50:
        model = ResNet_lorm(Bottleneck, [3, 4, 6, 3], num_classes,SynchronizedBatchNorm2d)
    elif layers == 101:
        model = ResNet_lorm(Bottleneck, [3, 4, 23, 3], num_classes,SynchronizedBatchNorm2d)
    elif layers == 152:
        model = ResNet_lorm(Bottleneck, [3, 8, 36, 3], num_classes,SynchronizedBatchNorm2d)
    else:
        print('unsupport layer number: []'.format(layers))
        exit()
    return model
