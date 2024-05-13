import os
import cv2
import shutil
from tqdm import tqdm
import numpy as np
import distance_map as dm
import argparse
from multiprocessing import Pool
# 0 1.0
# 1 2.718281828459045
# 2 7.3890560989306495
# 3 20.085536923187664
# 4 54.59815003314423
# 5 148.41315910257657
# 6 403.428793492735
# 7 1096.6331584284583
# 8 2980.957987041727
# 9 8103.08392757538


def job(task_data,args,task_id):
    flist = task_data
    for filename in tqdm(flist,position=task_id,total=len(flist),ncols=70):
        # print(filename)
        basename = filename.split('.')[0]
        img_name = filename.split('.')[0]+'.jpg'
        img_path = os.path.join(args.img_dir,img_name)
        jpeg_img = cv2.imread(img_path)
        cam_path = os.path.join(src_dir,filename)
        cam = cv2.imread(cam_path,cv2.IMREAD_GRAYSCALE)
        classes = np.unique(cam)
        cam[cam == 255] = 0
        cam[cam >0 ] = 255
        cam_counters,hierarchy = cv2.findContours(cam,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        counter_matrix = np.zeros_like(jpeg_img)
        counter_matrix = cv2.drawContours(counter_matrix,cam_counters,-1, (255, 255, 255), 1)
        # points = np.argwhere(cam == 255)
        # shape = (cam.shape[0],cam.shape[1])
        mp = dm.distance_map_from_binary_matrix(counter_matrix,lambd=float(args.lambd))
        mp = np.clip(mp,0,255).astype(np.uint8)
        
        # mp = dm.distance_map(shape,points).astype(np.uint8)
        # mp = mp[:,:,None]
        # mp = np.repeat(mp,3,axis=-1)
        # mp =  255 -mp
        
        
        vis_mp = np.concatenate((mp,counter_matrix,jpeg_img),0)
        
        # cv2.imshow('window',vis_mp)
        # cv2.waitKey(0)
        # classes = np.unique(scribble)
        save_path = os.path.join(save_dir,basename+'.png')
        cv2.imwrite(save_path,mp)
        vis_save_path = os.path.join(vis_dir,filename)
        cv2.imwrite(vis_save_path,vis_mp)

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',default='/home/zxl/dataset/ScribbleSup/JPEGImages')
parser.add_argument('--workers',default=10,type=int)
parser.add_argument('--src_dir',default='CAMpseudolabels/bmp_ws_train_aug_dataset')
parser.add_argument('--save_dir',default='./pseudolabel_dsmp/bmp/pseudo_dsmp_lambd_e',type=str)
parser.add_argument('--lambd',default='2.718281828459045',type=str,help='The distance decay weight')
args = parser.parse_args()

if __name__ == "__main__":

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    src_dir = args.src_dir
    vis_dir = save_dir+'_vis'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    flist = os.listdir(src_dir)

    new_flist = []
    print('checking processed files...')
    for filename in tqdm(os.listdir(src_dir),total=len(flist),ncols=90):
        if not os.path.isfile(os.path.join(save_dir,filename)):
            new_flist.append(filename)
    print('%d files exist, processing %d files'%(len(flist)-len(new_flist),len(new_flist)))
    flist = new_flist
    workers=args.workers
    task_split = []
    split_batchnum = int(len(flist)/workers)
    for i in range(workers):
        if i < workers-1:
            current_slice = flist[i*split_batchnum:split_batchnum*(i+1)]
            task_split.append(current_slice)
        else:
            current_slice = flist[i*split_batchnum:]
            task_split.append(current_slice)
    
    p = Pool(workers)
    for taskid,task_data in enumerate(task_split):
        # imgs2scribble(task_imginfo,taskid)
        p.apply_async(job,(task_data,args,taskid))
    p.close()
    p.join()