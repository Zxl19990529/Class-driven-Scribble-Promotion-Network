import os
import sys
import time
import shutil
import random
import argparse
import numpy as np
import torchnet as tnt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, Function
from torch.utils import data

# label = 255 is ambiguious label, and only some gts have this label.
class SegLoss ( nn.Module ) :
    def __init__(self, ignore_label=255, mode=1) :
        super ( SegLoss, self ).__init__ ()
        if mode == 1 :
            self.obj = torch.nn.CrossEntropyLoss ( ignore_index=ignore_label )
        else :
            self.obj = torch.nn.NLLLoss2d ( ignore_index=ignore_label )

    def __call__(self, pred, label) :
        loss = self.obj ( pred, label )
        return loss


class EntropyLoss ( nn.Module ) :
    def __init__(self) :
        super ( EntropyLoss, self ).__init__ ()

    def forward(self, x, mask, mode=1) : # mask : superpixel
        # mask_size = mask.size()[1:3]
        # x_softmax = F.softmax(x, dim = 1)
        # x_logsoftmax = F.log_softmax(x, dim = 1)
        # x_softmax_up = F.interpolate(x_softmax, size=mask_size, mode='bilinear', align_corners=True)
        # x_logsoftmax_up = F.interpolate(x_logsoftmax, size=mask_size, mode='bilinear', align_corners=True)
        # b = x_softmax_up * x_logsoftmax_up

        if mode == 1 :
            mask = 1.0 - mask / 255 # 除去superpixel位置，其余全是1  【4，465，465】
            b = F.softmax ( x, dim=1 ) * F.log_softmax ( x, dim=1 ) # (4,21,465,465)
            b = torch.sum ( b, dim=1 )  # (4,465,465)
            entropy = b.mul ( mask ) # 4,465,465
            loss = -1.0 * torch.sum ( entropy ) / torch.sum ( mask )
        else :
            b = F.softmax ( x, dim=1 ) * F.log_softmax ( x, dim=1 )
            b = torch.sum ( b, dim=1 )
            loss = -1.0 * torch.mean ( b )
        return loss



    
class DistanceMapLoss(nn.Module):
    def __init__(self,numclass=21) -> None:
        super(DistanceMapLoss,self).__init__()
        self.numclass = numclass
    def forward(self,x,label,distancemap,epsilon=0.1): # all tensors
        # x: b,21,h,w : the output of the network without softmax
        # label: b,h,w  [0,numclass-1]
        # disancemap: b,h,w   [0,255] 
        # pass
        batchsize = x.shape[0]
        x_logsoftmax = F.log_softmax(x,dim=1) # log softmax along 21  b,21,h,w
        label_onehot = F.one_hot(label,self.numclass) # b,h,w -> b,h,w,21
        label_onehot = label_onehot.permute(0,3,1,2) # b,h,w,21 -> b,21,h,w
        label_onehot = label_onehot +0.1
        label_onehot[label_onehot>1] = 1
        distancemap /= 255 # convert to problity
        distancemap = distancemap.unsqueeze(1) # b,h,w -> b,1,h,w
        
        softlabel = distancemap * label_onehot # broadcast  b,21,h,w
        
        loss = torch.sum(softlabel * x_logsoftmax,dim=1) # b,h,w
        
        loss = -torch.mean(loss)/batchsize
        return loss
    
    
class DistanceMapMinEntropy(nn.Module):
    def __init__(self,numclass=21) -> None:
        super(DistanceMapMinEntropy,self).__init__()
        self.numclass = numclass
    def forward(self,pred,distancemap,epsilon=0.1): # all tensors
        # pred: b,21,h,w : the output of the network without softmax
        # label: b,h,w  [0,numclass-1]
        # disancemap: b,h,w   [0,255] 
        # pass
        batchsize = pred.shape[0]
        x_logsoftmax = F.log_softmax(pred,dim=1) # log softmax along 21  b,21,h,w
        x_softmax = F.softmax(pred,dim=1) # softmax along 21  b,21,h,w
        distancemap /= 255 # convert to problity
        distancemap = distancemap.unsqueeze(1) # b,h,w -> b,1,h,w
        
        distance_logits = distancemap * x_softmax*x_logsoftmax # broadcast  b,21,h,w
        
        distance_entropy =  torch.sum(distance_logits,dim=1)# b,h,w
        loss = -torch.sum(distance_entropy)/ torch.count_nonzero(distance_entropy)
        # loss = -torch.mean(distance_entropy)
        return loss
        
class DistanceMinEntropy(nn.Module):
    def __init__(self,numclass=21) -> None:
        super(DistanceMinEntropy,self).__init__()
        self.numclass = numclass
    def forward(self,pred,pred_prob):
        batchsize = pred.shape[0]
        x_logsoftmax = F.log_softmax(pred,dim=1) # log softmax along 21  b,21,h,w
        x_softmax = F.softmax(pred,dim=1) # softmax along 21  b,21,h,w
        pred_prob = pred_prob.sigmoid() # convert to 0-1  b,1,h,w
        # pred_prob_mean = pred_prob.mean()
        # mask=torch.ones_like(pred_prob)
        # mask[pred_prob<pred_prob_mean] = 0
        # pred_prob = pred_prob * mask
        distance_logits = pred_prob * x_softmax * x_logsoftmax
        distance_entropy = torch.sum(distance_logits,dim=1)
        loss = -torch.sum(distance_entropy)/ torch.count_nonzero(distance_entropy)
        return loss
        
    