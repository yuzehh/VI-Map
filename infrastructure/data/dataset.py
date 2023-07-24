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

# class HDMapNetDataset(Dataset):
#     def __init__(self, version, dataroot, data_conf, is_train):
#         super(HDMapNetDataset, self).__init__()
#         patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]
#         patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]
#         canvas_h = int(patch_h / data_conf['ybound'][2])
#         canvas_w = int(patch_w / data_conf['xbound'][2])
#         self.is_train = is_train
#         self.data_conf = data_conf
#         self.patch_size = (patch_h, patch_w)
#         self.canvas_size = (canvas_h, canvas_w)
#         self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
#         self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size)
#         self.scenes = self.get_scenes(version, is_train)
#         self.samples = self.get_samples()

#     def __len__(self):
#         return len(self.samples)

#     def get_scenes(self, version, is_train):
#         # filter by scene split
#         split = {
#             'v1.0-trainval': {True: 'train', False: 'val'},
#             'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
#         }[version][is_train]

#         return create_splits_scenes()[split]

#     def get_samples(self):
#         samples = [samp for samp in self.nusc.sample]

#         # remove samples that aren't in this split
#         samples = [samp for samp in samples if
#                    self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

#         # sort by scene, timestamp (only to make chronological viz easier)
#         samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

#         return samples

#     def get_lidar(self, rec):
#         lidar_data = get_lidar_data(self.nusc, rec, nsweeps=3, min_distance=2.2)
#         lidar_data = lidar_data.transpose(1, 0)
#         num_points = lidar_data.shape[0]
#         lidar_data = pad_or_trim_to_np(lidar_data, [81920, 5]).astype('float32')
#         lidar_mask = np.ones(81920).astype('float32')
#         lidar_mask[num_points:] *= 0.0
#         return lidar_data, lidar_mask

#     def get_ego_pose(self, rec):
#         sample_data_record = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
#         ego_pose = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
#         car_trans = ego_pose['translation']
#         pos_rotation = Quaternion(ego_pose['rotation'])
#         yaw_pitch_roll = pos_rotation.yaw_pitch_roll
#         return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)

#     def sample_augmentation(self):
#         fH, fW = self.data_conf['image_size']
#         resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)
#         resize_dims = (fW, fH)
#         return resize, resize_dims

#     # def sample_augmentation(self):
#     #     self.data_conf['resize_lim'] = (0.193, 0.225)
#     #     self.data_conf['bot_pct_lim'] = (0.0, 0.22)
#     #     self.data_conf['rand_flip'] = True
#     #     self.data_conf['rot_lim'] = (-5.4, -5.4)

#     #     fH, fW = self.data_conf['image_size']
#     #     if self.is_train:
#     #         resize = np.random.uniform(*self.data_conf['resize_lim'])
#     #         resize_dims = (int(IMG_ORIGIN_W*resize), int(IMG_ORIGIN_H*resize))
#     #         newW, newH = resize_dims
#     #         crop_h = int((1 - np.random.uniform(*self.data_conf['bot_pct_lim']))*newH) - fH
#     #         crop_w = int(np.random.uniform(0, max(0, newW - fW)))
#     #         crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
#     #         flip = False
#     #         if self.data_conf['rand_flip'] and np.random.choice([0, 1]):
#     #             flip = True
#     #         rotate = np.random.uniform(*self.data_conf['rot_lim'])
#     #     else:
#     #         resize = max(fH/IMG_ORIGIN_H, fW/IMG_ORIGIN_W)
#     #         resize_dims = (int(IMG_ORIGIN_W*resize), int(IMG_ORIGIN_H*resize))
#     #         newW, newH = resize_dims
#     #         crop_h = int((1 - np.mean(self.data_conf['bot_pct_lim']))*newH) - fH
#     #         crop_w = int(max(0, newW - fW) / 2)
#     #         crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
#     #         flip = False
#     #         rotate = 0
#     #     return resize, resize_dims, crop, flip, rotate


#     def get_imgs(self, rec):
#         imgs = []
#         trans = []
#         rots = []
#         intrins = []
#         post_trans = []
#         post_rots = []

#         for cam in CAMS:
#             samp = self.nusc.get('sample_data', rec['data'][cam])
#             imgname = os.path.join(self.nusc.dataroot, samp['filename'])
#             img = Image.open(imgname)

#             resize, resize_dims = self.sample_augmentation()
#             img, post_rot, post_tran = img_transform(img, resize, resize_dims)
#             # resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
#             # img, post_rot, post_tran = img_transform(img, resize, resize_dims, crop, flip, rotate)

#             img = normalize_img(img)
#             post_trans.append(post_tran)
#             post_rots.append(post_rot)
#             imgs.append(img)

#             sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
#             trans.append(torch.Tensor(sens['translation']))
#             rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))
#             intrins.append(torch.Tensor(sens['camera_intrinsic']))
#         return torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(post_trans), torch.stack(post_rots)

#     def get_vectors(self, rec):
#         location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
#         ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
#         vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
#         return vectors

#     def __getitem__(self, idx):
#         rec = self.samples[idx]
#         imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
#         lidar_data, lidar_mask = self.get_lidar(rec)
#         car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
#         vectors = self.get_vectors(rec)

#         return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, vectors


# class HDMapNetSemanticDataset(HDMapNetDataset):
#     def __init__(self, version, dataroot, data_conf, is_train):
#         super(HDMapNetSemanticDataset, self).__init__(version, dataroot, data_conf, is_train)
#         self.thickness = data_conf['thickness']
#         self.angle_class = data_conf['angle_class']

#     def get_semantic_map(self, rec):
#         vectors = self.get_vectors(rec)
#         instance_masks, forward_masks, backward_masks = preprocess_map(vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness, self.angle_class)
#         semantic_masks = instance_masks != 0
#         semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])
#         instance_masks = instance_masks.sum(0)
#         forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class+1)
#         backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class+1)
#         direction_masks = forward_oh_masks + backward_oh_masks
#         direction_masks = direction_masks / direction_masks.sum(0)
#         return semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks

#     def __getitem__(self, idx):
#         rec = self.samples[idx]
#         imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
#         lidar_data, lidar_mask = self.get_lidar(rec)
#         car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
#         semantic_masks, instance_masks, _, _, direction_masks = self.get_semantic_map(rec)
#         return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_masks


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


    # def get_img(self, rec):
    #     imgname = os.path.join(rec, "lidar.jpg")
    #     img = cv2.imread(imgname)
    #     img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     img = normalize_img(img)
    #     return img


    def get_img(self, rec):
        input1 = os.path.join(rec, "traj_dens.npy")
        input2 = os.path.join(rec, "traj_dirmean.npy")
        input3 = os.path.join(rec, "traj_dirvar.npy")
        input1 = np.load(input1)
        # normalize traj dens to 0-1
        _range = np.max(input1) - np.min(input1)
        input1 = (input1 - np.min(input1)) / _range
        input1 = np.expand_dims(input1, axis=2)
        input2 = np.load(input2)
        input3 = np.load(input3)
        input3 = np.expand_dims(input3, axis=2)
        inputs = np.concatenate((input1,input2,input3), axis=2)
        inputs = inputs.transpose(2,0,1)
        
        return torch.tensor(inputs).type(torch.FloatTensor)
    
  
    def get_pcd(self, rec):
        '''
        traj_dens: already normalized to 0-1
        traj_dirmean: 2 dimentions with each dimention -1~1
        traj_dirvar: did not normalize
        pcd_reflt_mean: already normalized to 1-2
        h_max: already normalized to 1-2
        '''
        input1 = os.path.join(rec, "traj_dens.npy")
        input2 = os.path.join(rec, "traj_dirmean.npy")
        input3 = os.path.join(rec, "traj_dirvar.npy")
        input1 = np.load(input1)
        input1 = np.expand_dims(input1, axis=2)
        input2 = np.load(input2)
        input3 = np.load(input3)
        input3 = np.expand_dims(input3, axis=2)
        inputs = np.concatenate((input1,input2,input3), axis=2)
        inputs_traj = inputs.transpose(2,0,1)

        input1 = os.path.join(rec, "pcd_reflt_mean.npy") # input1 range: (1,2)
        input2 = os.path.join(rec, "h_max.npy") # input2 range: (1,2)
        input1 = np.load(input1)
        
        # # random mute some reflectivity
        # valid_locs = np.where(input1==1)
        # if len(valid_locs[0]) > 0:
        #     c = np.random.choice(range( len(valid_locs[0]) ), int(len(valid_locs[0])*0.5))
        #     input1[valid_locs[0][c],valid_locs[1][c]] = 0
        
        
        input1 = np.expand_dims(input1, axis=2)
        input2 = np.load(input2)
        input2 = np.expand_dims(input2, axis=2)
        inputs = np.concatenate((input1,input2), axis=2)
        inputs_pcd = inputs.transpose(2,0,1)
        # inputs = inputs_pcd
        
        inputs = np.concatenate((inputs_traj, inputs_pcd), axis=0)
        assert not math.isnan(inputs.mean())
        return torch.tensor(inputs).type(torch.FloatTensor)
    

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


    '''the following get_semantic_map: it returns three results: the semantic GT, 
    the instance segmentation GT for 4 kinds of map elements(lane regions, lane lines, crosswalks, and contours). '''
    # def get_semantic_map(self, rec):
    #     ''' (c, h, w)
    #         semantic_mask: --0 dim: crosswalks, lane lines, contours all white, others all black 
    #                        --1 dim: lane lines black, others white
    #                        --2 dim: crosswalks black, others white
    #                        --3 dim: contours black, others white
    #                        --4 dim: lane regions black, others white
    #         instance_mask: --0 dim: lane lines different instance with different gray degree, others all white, labels are 1,2,3... , backgroud is 0
    #                        --1 dim: crosswalks different instance with different gray degree, others all white, labels are predefined and cropped, backgroud is 0
    #                        --2 dim: contours different instance with different gray degree, others all white, labels are 1,2,3..., backgroud is 0
    #                        --3 dim: laneinstances different instance with different gray degree, others all white, labels are predefined and cropped, backgroud is 0
    #         direction_mask:--0 dim: crosswalks, lane lines, contours all white, others all black 
    #                        --1-36 dim: pixel with that degree black, others all white
    #                                    from 0 degree to 360 degree, with o degree == x axis positive, degrees adding along clockwise
    #     '''
    #     semantic_mask = np.load(file=os.path.join(rec, "semantic_mask.npy")) # (c,h,w)
    #     instance_mask = np.load(file=os.path.join(rec, "instance_mask.npy"))
    #     # direction_mask = np.loadtxt(os.path.join(rec, "direction_mask.txt"))
    #     # semantic_mask = semantic_mask.transpose(2,0,1) # (c,h,w)
    #     # instance_mask = instance_mask.transpose(2,0,1)
    #     # direction_mask = direction_mask.reshape((37, self.canvas_size[0], self.canvas_size[1]))
    #     # binarization semantic mask 
    #     label_idxs = np.unique(semantic_mask)
    #     assert len(label_idxs)==2 # black is 1, and white is 0
    #     semantic_mask = np.logical_not(semantic_mask) * 1
    #     # instance masks 
    #     im = np.zeros((instance_mask.shape[1], instance_mask.shape[2]))
    #     # transfer laneinstance => lane lines => contours => crosswalks, instance label: 1,2,3 ...
    #     label = 1
    #     # label_idxs = np.unique(laneinstance_mask)  
    #     # for l in label_idxs:
    #     #     if l != 0:
    #     #         c_pixels = np.argwhere(laneinstance_mask==l)
    #     #         im[c_pixels[:,0], c_pixels[:,1]] = label
    #     #         label += 1
    #     # label += 1
    #     for i in range(4):
    #         mask_i = instance_mask[i]
    #         label_idxs = np.unique(mask_i)  
    #         for l in label_idxs:
    #             if l != 0:
    #                 c_pixels = np.argwhere(mask_i==l)
    #                 im[c_pixels[:,0], c_pixels[:,1]] = label
    #                 label += 1

    #     return torch.tensor(semantic_mask), torch.tensor(im)


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
        
        # # 4 semantic categories
        # ll = semantic_mask[1]
        # cw = semantic_mask[2]
        # ct = semantic_mask[3]
        # li = semantic_mask[4]
        # bgd = ( np.logical_not(ll) | np.logical_not(cw) | np.logical_not(ct) | np.logical_not(li) ) * 1
        # semantic_mask[0] = bgd

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

    '''the following get_semantic_map: it returns three results: the semantic GT, 
    the instance segmentation GT for 3 kinds of map elements(lane lines, crosswalks, and contours),
    and the instance segmentation GT for different lane regions. '''
    # def get_semantic_map(self, rec):
    #     ''' (c, h, w)
    #         semantic_mask: --0 dim: crosswalks, lane lines, contours all white, others all black 
    #                        --1 dim: lane lines black, others white
    #                        --2 dim: crosswalks black, others white
    #                        --3 dim: contours black, others white
    #         instance_mask: --0 dim: lane lines different instance with different gray degree, pthers all white
    #                        --1 dim: crosswalks different instance with different gray degree, pthers all white
    #                        --2 dim: contours different instance with different gray degree, pthers all white
    #         direction_mask:--0 dim: crosswalks, lane lines, contours all white, others all black 
    #                        --1-36 dim: pixel with that degree black, others all white
    #                                    from 0 degree to 360 degree, with o degree == x axis positive, degrees adding along clockwise
    #     '''
    #     semantic_mask = np.load(file=os.path.join(rec, "semantic_mask.npy")) # (h,w,c)
    #     instance_mask = np.load(file=os.path.join(rec, "instance_mask.npy"))
    #     laneinstance_mask = np.load(file=os.path.join(rec, "laneinstance_GT.npy"))
    #     # direction_mask = np.loadtxt(os.path.join(rec, "direction_mask.txt"))
    #     semantic_mask = semantic_mask.transpose(2,0,1) # (c,h,w)
    #     instance_mask = instance_mask.transpose(2,0,1)
    #     # direction_mask = direction_mask.reshape((37, self.canvas_size[0], self.canvas_size[1]))
    #     # binarization semantic mask 
    #     label_idxs = np.unique(semantic_mask)
    #     assert len(label_idxs)==2
    #     semantic_mask[semantic_mask==0] = 1
    #     semantic_mask[semantic_mask==255] = 0 # black is 1, and white is 0
    #     # transfer instance mask to 1 dim 
    #     im = np.zeros((instance_mask.shape[1], instance_mask.shape[2]))
    #     label = 1
    #     for i in range(3):
    #         mask_i = instance_mask[i]
    #         label_idxs = np.unique(mask_i)  
    #         for l in label_idxs:
    #             if l != 255:
    #                 c_pixels = np.argwhere(mask_i==l)
    #                 im[c_pixels[:,0], c_pixels[:,1]] = label
    #                 label += 1
    #     # transfer laneinstance GT to 1,2,3 ... 
    #     lim = np.zeros((laneinstance_mask.shape[0], laneinstance_mask.shape[1]))
    #     label = 1
    #     label_idxs = np.unique(laneinstance_mask)  
    #     for l in label_idxs:
    #         if l != 0:
    #             c_pixels = np.argwhere(laneinstance_mask==l)
    #             lim[c_pixels[:,0], c_pixels[:,1]] = label
    #             label += 1

    #     return torch.tensor(semantic_mask), torch.tensor(im), torch.tensor(lim)


    def __getitem__(self, idx):   
        rec = self.samples[idx]
        # img = self.get_img(rec)
        pcd = self.get_pcd(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        # semantic_mask, instance_mask, laneinstance_mask = self.get_semantic_map(rec) 
        # semantic_mask, instance_mask = self.get_semantic_map(rec) 
        # return img, car_trans, yaw_pitch_roll, semantic_mask, instance_mask, laneinstance_mask
        return pcd, car_trans, yaw_pitch_roll, rec.split('/')[-1]





# # yuzeh: for hdmapnet_lidar only
# class HDMapNetSemanticDataset(HDMapNetDataset):
#     def __init__(self, version, dataroot, data_conf, is_train):
#         super(HDMapNetSemanticDataset, self).__init__(version, dataroot, data_conf, is_train)
#         self.thickness = data_conf['thickness']
#         self.angle_class = data_conf['angle_class']

#     def get_semantic_map(self, rec):
#         vectors = self.get_vectors(rec)
#         instance_masks, forward_masks, backward_masks = preprocess_map(vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness, self.angle_class)
#         semantic_masks = instance_masks != 0
#         semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])
#         instance_masks = instance_masks.sum(0)
#         forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class+1)
#         backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class+1)
#         direction_masks = forward_oh_masks + backward_oh_masks
#         direction_masks = direction_masks / direction_masks.sum(0)
#         return semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks

#     def __getitem__(self, idx):
#         rec = self.samples[idx]
#         # imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
#         lidar_data, lidar_mask = self.get_lidar(rec)
#         car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
#         semantic_masks, instance_masks, _, _, direction_masks = self.get_semantic_map(rec)
#         return lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_masks



def semantic_dataset(dataroot, data_conf, bsz, nworkers):
    # train_dataset = HDMapNetSemanticDataset(dataroot, data_conf, is_train=True)
    val_dataset = HDMapNetSemanticDataset(dataroot, data_conf, is_train=False)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return val_loader

# def semantic_dataset(version, dataroot, data_conf, bsz, nworkers):
#     train_dataset = HDMapNetSemanticDataset(version, dataroot, data_conf, is_train=True)
#     val_dataset = HDMapNetSemanticDataset(version, dataroot, data_conf, is_train=False)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    # return train_loader, val_loader




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

