import os
import time
import random
import numpy as np
import torchnet as tnt
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
from tqdm import tqdm
import time
import cv2,argparse

from models import res50_ASPP_MLP,res50_ASPP_lorm,res50_deeplabV3_lorm

import basic_function as func
import dataset
import transform


parser = argparse.ArgumentParser(description='PyTorch Hierachy_dif Training')
parser.add_argument('--layers', type=int, metavar='LAYERS',default= 50, help='the layer number of resnet: 18, 34, 50, 101, 152')
parser.add_argument('--dataset_path', metavar='DATASET_PATH',default='dataset/ScribbleSup/VOC2012', help='path to the dataset(multiple paths are concated by "+")')
parser.add_argument('--dataset', metavar='DATASET',default='VOC2012', help='dataset: VOC2012|PascalContext')
parser.add_argument('--numclasses', type=int, metavar='NUMCLASSES', default=21, help='number of classes')
parser.add_argument('--workers', default=4, type=int, metavar='WORKERS', help='number of dataload worker')
parser.add_argument ('--shrink_factor', default=1, type=int, metavar='SHRINK',
                                   help='shrink factor of attention map, preserved as URSS.' )
parser.add_argument('--val_path', metavar='TRAIN_PATH',default='SegmentationClassAug', help='path to the dataset training file')
parser.add_argument('--batchsize', default=1, type=int, metavar='BATCH_SIZE', help='batchsize')
parser.add_argument('--epochs', default=50, type=int, metavar='EPOCH',help='number of total epochs to run')
parser.add_argument('--model_path', default= 'None' ,help='pretrain model path')
# val param
parser.add_argument('--model_type',default='res50_lorm',type=str,help='Model type selection. nonRW|RW|res50_cam|res50_labelFusion')
parser.add_argument('--checkpoint_path', metavar='CHECKPOINT_PATH', help='path to the checkpoint file',default='log/train_deeplabv2_r50/last_checkpoint.pth')
parser.add_argument('--save_path', default='eval_results', metavar='SAVE_PATH', help='path to save the visualizations')

args = parser.parse_args()

opt_manualSeed = 1000
print("Random Seed: ", opt_manualSeed)
np.random.seed ( opt_manualSeed )
random.seed ( opt_manualSeed )
torch.manual_seed ( opt_manualSeed )
torch.cuda.manual_seed_all ( opt_manualSeed )

# cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = False

if not os.path.exists ( args.save_path ) :
    os.makedirs ( args.save_path )

def net_process(args, model, image, mean, std=None, flip=False):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        if args.model_type == 'res50_lorm':
            print(input.shape)
            output = model.forward_eval(input)
        elif args.model_type == 'res50_deeplabv3_lorm':
            output = model.forward_eval(input)
        else:
            output, output_clas = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    
    return output


def scale_process(args, model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3): # crop_h crop_w 465
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean) # pad the image to 
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    #debug()
    i = 0
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            i +=1 
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            t1 = time.time()
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(args, model, image_crop, mean, std)
            # print(i,time.time()-t1)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def test(args, val_loader, model, classes, mean, std, base_size, crop_h, crop_w, scales): # base_size=512, crop_ are 465
    tbar = tqdm ( val_loader )
    confusion_meter = tnt.meter.ConfusionMeter ( args.numclasses, normalized=False )
    model.eval()
    for i, (input, gt, img_path) in enumerate(tbar):
        input1 = np.squeeze(input.numpy(), axis=0) # 1,c,h,w -> c,h,w
        # print(input1.shape)# (3, 366, 500)
        image = np.transpose(input1, (1, 2, 0)) # c,h,w -> h,w,c
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size) # 256
            new_h = long_size
            new_w = long_size # 长边对齐缩放
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR) # scale the image maintaining the w-h ratio 
            print(image_scale.shape)
            prediction += scale_process(args, model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)
        
        #entropy = entr(prediction).sum(axis=2)/np.log(2)
        #img_name = img_path[0][img_path[0].rfind ( '/' ) + 1 :-4]
        #io.savemat(os.path.join ( args.save_path, img_name + '.mat' ), {'name': entropy})
        
        prediction = np.argmax(prediction, axis=2)
        pred=torch.from_numpy(prediction)
        pred=torch.unsqueeze(pred,0)
        valid_pixel = gt.ne(255)
        confusion_meter.add(pred[valid_pixel], gt[valid_pixel])
        
        mask = func.get_mask_pallete ( pred[0].cpu ().numpy (), 'pascal_voc' )

        img_name = img_path[0][img_path[0].rfind ( '/' ) + 1 :-4]
        mask.save ( os.path.join ( args.save_path, img_name + '.png' ) )
        #vutils.save_image ( input[0], os.path.join ( args.save_path, img_name + '.jpg' ), nrow=1, padding=0, normalize=True )
                                
    confusion_matrix = confusion_meter.value()
    inter = np.diag(confusion_matrix)
    union = confusion_matrix.sum(1).clip(min=1e-12) + confusion_matrix.sum(0).clip(min=1e-12) - inter

    mean_iou_ind = inter/union
    mean_iou_all = mean_iou_ind.mean()
    mean_acc_pix = float(inter.sum())/float(confusion_matrix.sum())
    print(' * IOU_All {iou}'.format(iou=mean_iou_all))
    print(' * IOU_Ind {iou}'.format(iou=mean_iou_ind))
    print(' * ACC_Pix {acc}'.format(acc=mean_acc_pix))
    with open(args.save_path+'.txt','a') as f:
        f.writelines(' * IOU_All {iou}\n'.format(iou=mean_iou_all))
        f.writelines(' * IOU_Ind {iou}\n'.format(iou=mean_iou_ind))
        f.writelines(' * ACC_Pix {acc}\n'.format(acc=mean_acc_pix))
        f.close()


if __name__ == '__main__':

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    val_transform = transform.Compose ( [transform.ToTensor ()])
    val_dataset = dataset.SemData ( split='val', data_root=args.dataset_path, data_list='val.txt',
                                    transform=val_transform, path='SegmentationClassAug' )
    val_loader = data.DataLoader ( val_dataset, num_workers=args.workers,
                                batch_size=1, shuffle=False, pin_memory=True )
    model_type = args.model_type
    
    if model_type == 'res50_cam':
        model = res50_ASPP_MLP.Res_DeeplabMLP(args.numclasses,args.layers)
    elif model_type == 'res50_lorm':
        model = res50_ASPP_lorm.Res_Deeplab(args.numclasses,args.layers)
    elif args.model_type == 'res50_deeplabv3_lorm':
        model = res50_deeplabV3_lorm.Res_Deeplab(args.numclasses,args.layers)
    model_pretrain = torch.load ( args.checkpoint_path )
    # model = func.param_restore_all ( model, model_pretrain['state_dict'] )
    model.load_state_dict(model_pretrain['state_dict'])
    model = model.cuda ()

    test(args, val_loader, model, args.numclasses, mean, std, 512, 465, 465, [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

    for item in args.__dict__.items():
        print(item)