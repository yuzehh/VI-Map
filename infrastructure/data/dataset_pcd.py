import math
import os

import cv2
import numpy as np
import torch
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from .const import CAMS, IMG_ORIGIN_H, IMG_ORIGIN_W, NUM_CLASSES
from .image import img_transform, normalize_img
from .lidar import get_lidar_data
from .rasterize import preprocess_map
from .utils import label_onehot_encoding
from .vector_map import VectorizedLocalMap
from .voxel import pad_or_trim_to_np



# dataset class for collected CARLA dataset
class HDMapNetSemanticDataset(Dataset):
    def __init__(self, dataroot, data_conf, is_train):
        super(HDMapNetSemanticDataset, self).__init__()
        self.is_train = is_train
        patch_h = data_conf['xbound'][1] - data_conf['xbound'][0]
        patch_w = data_conf['ybound'][1] - data_conf['ybound'][0]
        canvas_h = int(patch_h / data_conf['xbound'][2])
        canvas_w = int(patch_w / data_conf['ybound'][2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.data_conf = data_conf
        self.samples = self.get_samples(dataroot, is_train) 
        self.thickness = data_conf['thickness']
        self.angle_class = data_conf['angle_class']

    def __len__(self):
        return len(self.samples)

    def get_samples(self, dataroot, is_train):
        if is_train:
            folder = dataroot + "train/"
        else:
            folder = dataroot + "val/"
        sample_list = os.listdir(folder)
        samples = [os.path.join(folder, s) for s in sample_list]
        return samples

    def sample_augmentation(self):
        fH, fW = self.data_conf['image_size']
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)
        resize_dims = (fW, fH)
        return resize, resize_dims

  
    def get_pcd(self, rec):
        rec = rec.replace("lamp_dataset_lessregion", "lamp_pcd_dense")
        input = os.path.join(rec, "10.npy") 
        input = np.load(input)
    
        return torch.tensor(input).type(torch.FloatTensor)
    

    def get_ego_pose(self, rec):
        # for lamp_dataset and lamp_dataset_bigline 
        # location = np.loadtxt(os.path.join(rec, 'lamp_location.txt'))
        # car_trans = location[:3] # x,y,z
        # pitch, yaw, roll = location[-3:]
        # yaw_pitch_roll = np.array([yaw, pitch, roll]) # pitch, yaw, roll
        # return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)
        
        # for lamp_dataset_lessregion
        car_trans = np.array([0, 0, 0])
        yaw_pitch_roll = np.array([0, 0, 0]) # pitch, yaw, roll
        return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)


    def get_semantic_map(self, rec):
        ''' (c, h, w)
            semantic_mask: --0 dim: crosswalks, lane lines, contours all white, others all black 
                           --1 dim: lane lines black, others white
                           --2 dim: crosswalks black, others white
                           --3 dim: contours black, others white
                           --4 dim: lane regions black, others white
            instance_mask: --0 dim: lane lines different instance with different gray degree, others all white, labels are 1,2,3... , backgroud is 0
                           --1 dim: crosswalks different instance with different gray degree, others all white, labels are predefined and cropped, backgroud is 0
                           --2 dim: contours different instance with different gray degree, others all white, labels are 1,2,3..., backgroud is 0
                           --3 dim: laneinstances different instance with different gray degree, others all white, labels are predefined and cropped, backgroud is 0
            direction_mask:--0 dim: crosswalks, lane lines, contours all white, others all black 
                           --1-36 dim: pixel with that degree black, others all white
                                       from 0 degree to 360 degree, with o degree == x axis positive, degrees adding along clockwise
        '''
        semantic_mask = np.load(file=os.path.join(rec, "semantic_mask.npy")) # (c,h,w)
        instance_mask = np.load(file=os.path.join(rec, "instance_mask.npy"))
        # direction_mask = np.loadtxt(os.path.join(rec, "direction_mask.txt"))
        # semantic_mask = semantic_mask.transpose(2,0,1) # (c,h,w)
        # instance_mask = instance_mask.transpose(2,0,1)
        # direction_mask = direction_mask.reshape((37, self.canvas_size[0], self.canvas_size[1]))
        # binarization semantic mask 
    
        # only 3 semantic categories 
        # semantic_mask = semantic_mask[0:4]
        # instance_mask = instance_mask[0:3]
        
        # 4 semantic categories
        ll = semantic_mask[1]
        cw = semantic_mask[2]
        ct = semantic_mask[3]
        li = semantic_mask[4]
        bgd = ( np.logical_not(ll) | np.logical_not(cw) | np.logical_not(ct) | np.logical_not(li) ) * 1
        semantic_mask[0] = bgd
        label_idxs = np.unique(semantic_mask)
        assert len(label_idxs)==2 # black is 1, and white is 0
        semantic_mask = np.logical_not(semantic_mask) * 1
        # instance masks 
        im = np.zeros((instance_mask.shape[1], instance_mask.shape[2]))
        # transfer laneinstance => lane lines => contours => crosswalks, instance label: 1,2,3 ...
        label = 1
        # label_idxs = np.unique(laneinstance_mask)  
        # for l in label_idxs:
        #     if l != 0:
        #         c_pixels = np.argwhere(laneinstance_mask==l)
        #         im[c_pixels[:,0], c_pixels[:,1]] = label
        #         label += 1
        # label += 1
        for i in range(4):
            mask_i = instance_mask[i]
            label_idxs = np.unique(mask_i)  
            for l in label_idxs:
                if l != 0:
                    c_pixels = np.argwhere(mask_i==l)
                    im[c_pixels[:,0], c_pixels[:,1]] = label
                    label += 1

        return torch.tensor(semantic_mask), torch.tensor(im)


    def __getitem__(self, idx):   
        rec = self.samples[idx]
        pcd = self.get_pcd(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        semantic_mask, instance_mask = self.get_semantic_map(rec) 
        return pcd, car_trans, yaw_pitch_roll, semantic_mask, instance_mask, rec.split('/')[-1]


def semantic_dataset(dataroot, data_conf, bsz, nworkers):
    train_dataset = HDMapNetSemanticDataset(dataroot, data_conf, is_train=True)
    val_dataset = HDMapNetSemanticDataset(dataroot, data_conf, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return train_loader, val_loader


if __name__ == '__main__':
    data_conf = {
        'image_size': (900, 1600),
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }

    dataset = HDMapNetSemanticDataset(version='v1.0-mini', dataroot='dataset/nuScenes', data_conf=data_conf, is_train=False)
    for idx in range(dataset.__len__()):
        imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_mask = dataset.__getitem__(idx)

