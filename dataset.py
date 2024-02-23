import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def load_image_label_list_from_npy(img_name_list, data_root):
    cls_labels_dict = np.load('{}/cls_labels.npy'.format(data_root), allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]

class Sem_pesudo_distancemap_Data(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'pascal_2012_scribble',path_pesudo = 'bmp_ws_train_aug_dataset',path_distancemap='distance_map_lambd_1',path_distanceCAM='None'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.lab_pesudo_names = ['{}/{}/{}.png'.format(data_root, path_pesudo, i) for i in self.indices]
        self.distancemap_nams = ['{}/{}/{}.png'.format(data_root, path_distancemap, i) for i in self.indices]
        self.distanceCAM_nams = ['{}/{}/{}.png'.format(data_root, path_distanceCAM, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        label_pesudo_path = self.lab_pesudo_names[index]
        distance_map_path = self.distancemap_nams[index]
        distance_CAM_path = self.distanceCAM_nams[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        label_pesudo = cv2.imread(label_pesudo_path,cv2.IMREAD_GRAYSCALE)
        distance_map = cv2.imread(distance_map_path,cv2.IMREAD_GRAYSCALE)
        distance_CAM = cv2.imread(distance_CAM_path,cv2.IMREAD_GRAYSCALE)
        
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1] or image.shape[1] != label_pesudo.shape[1] or image.shape[1] != distance_map.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label,label_pesudo,distance_map,distance_CAM = self.transform(image, label, label_pesudo,distance_map,distance_CAM )
            distance_map = distance_map.float()
            distance_CAM = distance_CAM.float()
        return image, label,label_pesudo,distance_map,distance_CAM, image_path

class Sem_pesudoData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'pascal_2012_scribble',path_pesudo = 'bmp_ws_train_aug_dataset'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.lab_pesudo_names = ['{}/{}/{}.png'.format(data_root, path_pesudo, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        label_pesudo_path = self.lab_pesudo_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        label_pesudo = cv2.imread(label_pesudo_path,cv2.IMREAD_GRAYSCALE)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1] or image.shape[1] != label_pesudo.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label,label_pesudo = self.transform(image, label, label_pesudo )
        return image, label,label_pesudo, image_path


class Sem_ContourData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'pascal_2012_scribble',path_contour='superpixels'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.contour_names = ['{}/{}/{}.png'.format(data_root, path_contour, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        contour_path=self.contour_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        contour = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1] or image.shape[1] != contour.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label, contour = self.transform(image, label, contour)
        return image, label, contour, image_path

class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'SegmentationClassAug'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label, image_path
    
class SemData_test(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'SegmentationClassAug'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        label = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + image_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label, image_path
        
        
class ClsData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'SegmentationClassAug'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = [i for i in self.indices]
        self.lab_names2 = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.label_list = load_image_label_list_from_npy(self.lab_names, data_root)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path = self.img_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = self.label_list[index]
        label_path=self.lab_names2[index]
        label2 = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            image,label2 = self.transform(image,label2)
        return image, label, image_path
