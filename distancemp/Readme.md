# Make the distance map accrodding to the scribble or the pseudo label's boundary

## Dependency

```py
pip install opencv-python
pip install tqdm
```

## Make the distance map from the scribble

Run the file:
```py
python mamakedistancemap_multiprocessing.py \
    --workers 10 \ # This depends on your cpu cores.
    --src_dir /home/zxl/dataset/ScribbleSup/pascal_2012_scribble \ # path to the scribble mask folder.
    --save_dir ./dsmp_scribble_lambd_e2 \ # path to the save folder.
    --lambd 7.3890560989306495 # The lambd_s to control the distance map.
```

## Make the distance map from the pseudo label's boundary

Run the file:
```py
python makedistancemap_fromCAM_multiprocessing.py.py \
    --workers 10 \ # This depends on your cpu cores.
    --src_dir /home/zxl/dataset/ScribbleSup/CAMpseudolabels/bmp_ws_train_aug_dataset \ # path to the pseudo label folder.
    --save_dir ./pseudolabel_dsmp/bmp/pseudo_dsmp_lambd_e \ # path to the save folder.
    --lambd 7.3890560989306495 # The lambd_s to control the distance map.
```
