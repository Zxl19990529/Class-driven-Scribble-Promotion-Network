import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler,DataLoader
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from models import res50_ASPP_MLP,res50_ASPP_CAMATT,res50_deeplabV3_CAMATT,deeplabV3p,deeplabV2
import dataset
import transform as transform
import transform_exten

def get_model(model_type,distributed,args,local_rank,device):
    model = None
    if model_type == 'res50_cam':
        model = res50_ASPP_MLP.Res_DeeplabMLP(args.numclasses,args.layers)
    elif model_type == 'res50_CAMATT':
        model = res50_ASPP_CAMATT.Res_Deeplab(args.numclasses,args.layers)
    elif model_type == 'res50_deeplabv3_camatt':
        model = res50_deeplabV3_CAMATT.Res_Deeplab(args.numclasses,args.layers)
    elif model_type == 'deeplabv3p':
        model = deeplabV3p.Res_Deeplab(args.numclasses,args.layers)
    elif model_type == 'deeplabv2':
        model = deeplabV2.Res_Deeplab(args.numclasses,args.layers)
    model = model.to(device)
    if distributed:
        model = DistributedDataParallel(model,device_ids=[local_rank])
    
    return model


def get_dataloader(args,distributed):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    train_transform = transform_exten.Compose([
            transform_exten.RandScale([0.5, 2.0]),
            transform_exten.RandRotate([-10, 10], padding=mean, ignore_label=255),
            transform_exten.RandomGaussianBlur(),
            transform_exten.RandomHorizontalFlip(),
            transform_exten.Crop([465, 465], crop_type='rand', padding=mean, ignore_label=255),
            transform_exten.ToTensor(),
            transform_exten.Normalize(mean=mean, std=std)])
    val_transform = transform.Compose([
        transform.Crop([465, 465], crop_type='center', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    if args.dataset == 'VOC2012':
        train_dataset = dataset.Sem_ContourData(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path = args.train_path, path_contour= 'superpixels')    
        val_dataset = dataset.SemData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    elif args.dataset == 'ScribbleOnly':
        train_dataset = dataset.SemData(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path = args.train_path)    
        val_dataset = dataset.SemData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    elif args.dataset == 'SBD_pesudo':
        train_dataset = dataset.Sem_pesudoData(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path = args.train_path, path_pesudo=args.pesudo_path)
        val_dataset = dataset.SemData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    elif args.dataset == 'scribble_pesudo':
        train_dataset = dataset.Sem_pesudoData(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path = args.train_path, path_pesudo=args.pesudo_path)
        val_dataset = dataset.SemData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    elif args.dataset == 'SBD_pesudo_distancemap':
        train_dataset = dataset.Sem_pesudo_distancemap_Data(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path = args.train_path, path_pesudo=args.pesudo_path,path_distancemap=args.distance_map_Scrib,path_distanceCAM=args.distance_map_CAM)
        val_dataset = dataset.SemData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    elif args.dataset == 'scribble_pseudo_dsmp':
        train_dataset = dataset.Sem_distancemap_Data(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path = args.train_path, path_pesudo=args.pesudo_path,path_distancemap=args.distance_map_Scrib)
        val_dataset = dataset.SemData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(val_dataset)
        train_loader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batchsize,sampler=sampler_train, pin_memory=True,prefetch_factor=2,drop_last=True)
        val_loader = DataLoader(val_dataset, num_workers=args.workers, batch_size=args.batchsize,sampler=sampler_val, pin_memory=True,prefetch_factor=2)
    else:
        train_loader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batchsize, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, num_workers=args.workers, batch_size=int(args.batchsize), shuffle=False, pin_memory=True)
    
    return train_loader,val_loader

