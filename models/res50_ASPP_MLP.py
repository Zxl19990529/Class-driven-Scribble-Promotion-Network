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

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

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


class ASPP(nn.Module):
    def __init__(self, inplanes, dilations, BatchNorm, num_classes):
        super(ASPP, self).__init__()

        self.aspp1 = _ASPPModule(inplanes, num_classes, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, num_classes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, num_classes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, num_classes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, num_classes, 1, stride=1, bias=False),
                                             BatchNorm(num_classes),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(5*num_classes, num_classes, 1, bias=False)
        self.bn1 = BatchNorm(num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.relu(x)

        #return self.dropout(x)
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


class Classifier_Module_V2(nn.Module):
    def __init__(self, dilation_series, padding_series, num_classes, inplane):
        super(Classifier_Module_V2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(inplane, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class MLP(nn.Module):
    def __init__(self,in_feature,out_feature):
        super(MLP,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1=torch.nn.Linear(in_feature,2048)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(2048,2048) #2个隐层
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(2048,out_feature)

    def forward(self, x): # b,2048,59,59
        x = self.avgpool(x) # b,2048,1,1
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
    

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
def normalize_tensor(x):
    channel_vector = x.view(x.size()[0], x.size()[1], -1)
    minimum, _ = torch.min(channel_vector, dim=-1, keepdim=True)
    maximum, _ = torch.max(channel_vector, dim=-1, keepdim=True)
    normalized_vector = torch.div(channel_vector - minimum, maximum - minimum)
    normalized_tensor = normalized_vector.view(x.size())
    return normalized_tensor

class CAMClassifier(nn.Module):

    def __init__(self, num_feature, num_classes):
        super(CAMClassifier, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Conv2d(num_feature, num_classes, 1, 1, padding=0)
        self.dropout = nn.Dropout(0.7)
        initialize_weights(self.modules(), init_mode='xavier')
    
    def forward(self, x, labels=None, return_cam=False): # b,2048,59,59
        
        batch_size = x.shape[0]

        x_ = self.avg(x)
        output = self.fc(self.dropout(x_))
        if return_cam:
            output = self.fc(x) # b,20,59,59
            normalized = normalize_tensor(output.detach().clone()) # b,2048,59,59
            cams = normalized[range(batch_size), labels] # 2,59,59
            return output,cams
        else:
            return output


class ResNet_ASPP_MLP(nn.Module):
    def __init__(self, block, layers, num_classes,BatchNorm=nn.BatchNorm2d):
        self.inplanes = 64
        super(ResNet_ASPP_MLP, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64, affine = affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0],BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2,BatchNorm=BatchNorm)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4,BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4,BatchNorm=BatchNorm,dilations=[4,8,16])
        if block.__name__ == 'Bottleneck':
            self.seg_layer = self._make_seg_layer(ASPP, 2048, [6,12,18,24], BatchNorm, num_classes)
        else:
            self.seg_layer = self._make_seg_layer(ASPP, 512, [6,12,18,24], BatchNorm, num_classes)
        self.classification_layer = CAMClassifier(512*block.expansion,num_classes-1) # numclass-1, where we remove the background class.
        
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

    def _make_seg_layer(self, block, inplane, dilation_series, BatchNorm, num_classes):
        return block(inplane, dilation_series, BatchNorm, num_classes)

    def forward(self, x, return_cam=False,labels=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        seg_out = self.seg_layer(x4)
        if return_cam:
            class_out,cam = self.classification_layer(x4,return_cam=return_cam,labels=labels)
            return seg_out,class_out,cam
        else:
            class_out = self.classification_layer(x4)
            return seg_out,class_out


def Res_DeeplabMLP(num_classes=21, layers=18):
    layers = int(layers)
    if layers == 18:
        model = ResNet_ASPP_MLP(BasicBlock, [2, 2, 2, 2], num_classes,SynchronizedBatchNorm2d)
    elif layers == 34:
        model = ResNet_ASPP_MLP(BasicBlock, [3, 4, 6, 3], num_classes,SynchronizedBatchNorm2d)
    elif layers == 50:
        model = ResNet_ASPP_MLP(Bottleneck, [3, 4, 6, 3], num_classes,SynchronizedBatchNorm2d)
    elif layers == 101:
        model = ResNet_ASPP_MLP(Bottleneck, [3, 4, 23, 3], num_classes,SynchronizedBatchNorm2d)
    elif layers == 152:
        model = ResNet_ASPP_MLP(Bottleneck, [3, 8, 36, 3], num_classes,SynchronizedBatchNorm2d)
    else:
        print('unsupport layer number: []'.format(layers))
        exit()
    return model
