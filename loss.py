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

class EigLoss ( nn.Module ) :
    def __init__(self, eig=True) :
        super ( EigLoss, self ).__init__ ()
        self.eig = eig
        if self.eig == True :
            self.L1Loss = nn.L1Loss ( reduction='mean' )
        self.kld = nn.KLDivLoss ( reduction='mean' )

    def forward(self, f1, f2) :
        f1_softmax = F.softmax ( f1, dim=1 )
        f2_softmax = F.softmax ( f2, dim=1 )
        f1_logsoftmax = F.log_softmax ( f1, dim=1 )
        loss2 = self.kld ( f1_logsoftmax, f2_softmax )

        if self.eig == True :
            loss1 = self.L1Loss ( torch.diagonal ( f1_softmax, dim1=-2, dim2=-1 ).sum ( -1 ),
                                  torch.diagonal ( f2_softmax, dim1=-2, dim2=-1 ).sum ( -1 ) )
            loss = 1e-2 * loss1 + loss2
        else :
            loss = loss2
        return loss


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


class MSELoss_mask ( nn.Module ) :
    def __init__(self) :
        super ( MSELoss_mask, self ).__init__ ()
        self.criterion_mse = nn.MSELoss ( reduction='none' )
        self.criterion_mse_mean = nn.MSELoss ( reduction='mean' )

    def forward(self, x1, x2, mask=None, mask_type=0) :
        if mask_type == 0 :
            loss = self.criterion_mse_mean ( x1, x2 )
        elif mask_type == 1 :
            mse_loss = self.criterion_mse ( x1, x2 )
            input_size = x1.size ()[2 :4]
            batch_size = x1.size ()[1]
            mask = F.interpolate ( torch.unsqueeze ( mask, 1 ).float (), size=input_size, mode='nearest' )
            mask_ignore = (mask != 255) & (mask != 0)
            mse_mask_loss = mse_loss.mul ( mask_ignore )
            loss = torch.sum ( mse_mask_loss ) / (torch.sum ( mask_ignore ) * batch_size)
        else :
            mse_loss = self.criterion_mse ( x1, x2 )
            input_size = x1.size ()[2 :4]
            batch_size = x1.size ()[1]
            mask = F.interpolate ( torch.unsqueeze ( mask, 1 ), size=input_size, mode='bilinear' )
            mse_mask_loss = mse_loss.mul ( mask )
            loss = torch.sum ( mse_mask_loss ) / (torch.sum ( mask ) * batch_size)
        return loss


class EdgeLoss_entropy ( nn.Module ) :
    def __init__(self, class_num) :
        super ( EdgeLoss_entropy, self ).__init__ ()
        sobel_kernel = np.array ( [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32' )
        sobel_kernel = sobel_kernel.reshape ( (1, 1, 3, 3) )
        sobel_kernel = sobel_kernel.repeat ( class_num, 0 )
        self.weight = torch.from_numpy ( sobel_kernel )

    def forward(self, pred_sg_up, edge_v) :
        pred_sg_softmax = F.softmax ( pred_sg_up )
        edge_pred = F.conv2d ( pred_sg_softmax, self.weight.cuda (), padding=1, groups=21 )
        edge_pred = torch.tanh ( torch.sum ( torch.abs ( edge_pred ), dim=1, keepdim=True ) )
        loss_edge = torch.mean (
            torch.sum ( torch.mul ( edge_pred, torch.abs ( edge_pred - edge_v / 255 ) ), dim=(1, 2) ) / torch.sum (
                edge_pred, dim=(1, 2) ) )
        return loss_edge


class EdgeLoss ( nn.Module ) :
    def __init__(self, delta=0.1, edge_balance=False) :
        super ( EdgeLoss, self ).__init__ ()
        self.edge_balance = edge_balance
        self.delta = delta

    def forward(self, pred_sg_up, edge_v) :
        edge = torch.flatten ( edge_v, start_dim=1 )
        pred_seg_softmax = torch.softmax ( pred_sg_up, 1 )
        pred_seg = torch.flatten ( pred_seg_softmax, start_dim=2 )
        batch_size = pred_seg.size ()[0]
        channel = pred_seg.size ()[1]
        var_term = 0.0
        for i in range ( batch_size ) :
            unique_labels, unique_id, counts = torch.unique ( edge[i], return_inverse=True, return_counts=True )
            num_instances = unique_labels.size ()[0]
            unique_id_repeat = unique_id.unsqueeze ( 0 ).repeat ( channel, 1 )
            segmented_sum = torch.zeros ( channel, num_instances ).cuda ().scatter_add ( dim=1, index=unique_id_repeat,
                                                                                         src=pred_seg[i] )
            mu = torch.div ( segmented_sum, counts )
            mu_expand = torch.gather ( mu, 1, unique_id_repeat )
            tmp_distance = pred_seg[i] - mu_expand
            distance = torch.sum ( torch.abs ( tmp_distance ), dim=0 )
            distance = torch.clamp ( distance - self.delta, min=0.0 )
            if self.edge_balance == False :
                mask = (edge[i] != 0) & (edge[i] != 255)
                l_var = torch.sum ( distance * mask ) / (torch.sum ( mask ) + 1e-5)
            else :
                l_var = torch.zeros ( num_instances ).cuda ().scatter_add ( dim=0, index=unique_id, src=distance )
                l_var = torch.div ( l_var, counts )
                mask = (unique_labels != 0) & (unique_labels != 255)
                l_var = torch.sum ( l_var * mask ) / (torch.sum ( mask ) + 1e-5)
            var_term = var_term + l_var
        loss_edge = var_term / batch_size
        return loss_edge
    
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
        
        

class ConsistencyLoss(nn.Module):
    def __init__(self,loss_type='l2',cal_mode='logits') -> None:
        super(ConsistencyLoss,self).__init__()
        self.loss_type = loss_type # l2 or other ways
        self.mode = cal_mode # logits or softmax
        if self.loss_type == 'l2':
            self.loss_func = nn.MSELoss()
    def forward(self,pred1,pred2): # pred: b,21,51,51
        if self.mode == 'logits':
            return self.loss_func(pred1,pred2)
        elif self.mode == 'softmax':
            pred1 = F.softmax(pred1,dim=1)
            pred2 = F.softmax(pred2,dim=1)            
            return self.loss_func(pred1,pred2)