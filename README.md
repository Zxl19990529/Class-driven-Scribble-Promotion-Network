# Scribble Hides Class: Promoting Scribble-based Weakly Supervised Semantic Segmentation with Its Class Label

This repository contains training and evaluation codes, and some exmaple images of the dataset.  The pretrained checkpoint using ```resnet50+deeplabV2``` is avaliable, which is the used in our ablation study. The complete dataset with distance maps and pseudo-label will be avaliable upon paper acceptance.

## Environment setup

- anaconda3
    - python>=3.8
    - ipykernel
    - pytorch = 1.13
    - numpy
    - matplotlib
    - torchnet

It is recommanded to in install a anaconda3 for convient environment setup. Run the conda command:  
```sh
conda env create -f cdsp.yaml
```

## Scripts
The training and evaluating scripts are in ```scripts/*.sh```

### Train

Run the example script:  
```sh
sh scripts/train_r50_deeplabv2.sh
```  
Not avaliable for now, the complete dataset with distance maps and pseudo-label will be avaliable upon paper acceptance.
### Eval
Download the pretrained checkpoint from [dropbox](https://www.dropbox.com/scl/fi/4eki9ioib3pj4g60hu6hq/train_deeplabv2_r50.zip?rlkey=qtep6d4r9ctoawxnqk0w0porw&dl=0) or [googledrive](https://drive.google.com/file/d/1EBHTmvRaYkCJKcmN26HgiBzJ6LJFZLay/view?usp=sharing), and move the checkpoint to the ```log```  folder,
Run the example script:  
```sh
sh scripts/emlc_r50_deeplabv2.sh
```  
The visualization reults will be saved in emlc_r50_deeplabv2, and the quantity results will be saved in the ```.txt``` file.
### Demo
Open the ```demo.ipynb``` with jupyter notebook, and follow the commands.

### Thanks
This repo is inspired by [URSS](https://github.com/panzhiyi/URSS)