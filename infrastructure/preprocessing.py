import math
import os
import random
import shutil
from sqlite3 import Row

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd

# import pcl
# import pcl.pcl_visualization





def mkdir_folder(path, subfolder):
    if not os.path.isdir(os.path.join(path, subfolder)):
        os.makedirs(os.path.join(path, subfolder))
    return os.path.join(path, subfolder)


def CalAngle(v1, v2):
    # v1旋转到v2，逆时针为正，顺时针为负
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
    # 叉乘
    rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / TheNorm))
    # 点乘
    theta = np.rad2deg(np.arccos(np.dot(v1, v2) / TheNorm))
    if rho < 0:
        return - theta
    else:
        return theta


def normalization(data):
    eps = 0.00001
    _range = np.max(data) - np.min(data)
    return (data- np.min(data)) / (_range+eps)

def rasterization_traj(npy, xbound, ybound, res, canvas_size):
    # crop traj pts  
    x = npy[:,0]
    y = npy[:,1]
    x_values = pd.Series(x)
    y_values = pd.Series(y)
    x_in_canvas = x_values.between(xbound[0], xbound[1])
    y_in_canvas = y_values.between(ybound[0], ybound[1])
    pt_in_canvas = np.array(x_in_canvas) & np.array(y_in_canvas)
    ptidx = np.argwhere(pt_in_canvas==True)
    # pts = npy[ptidx, :].reshape((-1,4))
    pts = npy[ptidx, :]
    if pts.shape[0] == 0:
        return 1,1,1
    else:
        # traj pts => grids   
        pts = pts.reshape((-1,4))     
        xmin = xbound[0]
        ymin = ybound[0]
        pts[:, 0] -= xmin 
        pts[:, 1] -= ymin 
        row = canvas_size[0]
        column = canvas_size[1]
        gridnum = row * column
        grid_list = [[] for i in range(gridnum)]
        for i in range(pts.shape[0]):
            x = pts[i][0]
            y = pts[i][1]
            grid_index = np.floor(x / res) * column + np.floor(y / res)
            if grid_index <= (gridnum-1):
                grid_list[int(grid_index)].append(i)
            else:
                print("Warning: There are points that are not in the canvas!")
        
        # rules of generating images grids => images
        # check ipad notes for the permutation of girds 
        # INPUT 1: traj pts density
        image = np.zeros((row, column))
        for i, ls in enumerate(reversed(grid_list)): 
            pts_in_grid = pts[ls]
            image[i//column , i%column] = pts_in_grid.shape[0]
        traj_density = np.int8(normalization(image) * 250) + 3
        # INPUT 2: traj pts direction mean
        image = np.zeros((row, column))
        for i, ls in enumerate(reversed(grid_list)): 
            pts_in_grid = pts[ls]
            dir_mean = np.mean(pts_in_grid[:,-1])
            image[i//column , i%column] = dir_mean
        traj_dir_mean = np.int8(normalization(image) * 250) + 3
        # INPUT 3: traj pts direction variance 
        image = np.zeros((row, column))
        for i, ls in enumerate(reversed(grid_list)): 
            pts_in_grid = pts[ls]
            dir_var = np.var(pts_in_grid[:,-1])
            image[i//column , i%column] = dir_var
        traj_dir_var = np.int8(normalization(image) * 250 + 3)

        return traj_density, traj_dir_mean, traj_dir_var


# def rasterization_pcd(npy, xbound, ybound, res, canvas_size):
#     # crop pts  
#     x = npy[:,0]
#     y = npy[:,1]
#     x_values = pd.Series(x)
#     y_values = pd.Series(y)
#     x_in_canvas = x_values.between(xbound[0], xbound[1])
#     y_in_canvas = y_values.between(ybound[0], ybound[1])
#     pt_in_canvas = np.array(x_in_canvas) & np.array(y_in_canvas)
#     ptidx = np.argwhere(pt_in_canvas==True)
#     pts = npy[ptidx, :].reshape((-1,4))

#     # pts => grids        
#     xmin = xbound[0]
#     ymin = ybound[0]
#     pts[:, 0] -= xmin 
#     pts[:, 1] -= ymin 
#     row = canvas_size[0]
#     column = canvas_size[1]
#     gridnum = row * column
#     grid_list = [[] for i in range(gridnum)]
#     for i in range(pts.shape[0]):
#         x = pts[i][0]
#         y = pts[i][1]
#         grid_index = np.floor(x / res) * column + np.floor(y / res)
#         if grid_index <= (gridnum-1):
#             grid_list[int(grid_index)].append(i)
#         else:
#             print("Warning: There are points that are not in the canvas!")
    
#     # rules of generating images: grids => images
#     # check ipad notes for the permutation of girds 
#     # INPUT 1: pcd reflection mean
#     image = np.zeros((row, column))
#     value_dict = {}
#     for i, ls in enumerate(reversed(grid_list)): 
#         if len(ls) != 0:
#             pts_in_grid = pts[ls]
#             value = np.sum(pts_in_grid[:,-1]==6) / pts_in_grid.shape[0]
#             value_dict[i] = value
#     # pcd_reflt_mean = np.int8(normalization(np.array(list(value_dict.values()))) * 250) + 3 
#     pcd_reflt_mean = normalization(np.array(list(value_dict.values())))+1
#     pcd_reflt_mean = list(pcd_reflt_mean)
#     value_dict = dict(zip(value_dict.keys(), pcd_reflt_mean))
#     for i, ls in enumerate(reversed(grid_list)):
#         if i in value_dict.keys():
#             image[i//column , i%column] = value_dict[i]
#     input1 = image
    
#     # INPUT 2: pcd height difference 
#     image = np.zeros((row, column))
#     value_dict = {}
#     for i, ls in enumerate(reversed(grid_list)): 
#         if len(ls) != 0:
#             pts_in_grid = pts[ls]
#             # TODO: max or mean?
#             value = np.max(pts_in_grid[:,2])
#             value_dict[i] = value
#     # h_max = np.int8(normalization(np.array(list(value_dict.values()))) * 250) + 3 
#     h_max = normalization(np.array(list(value_dict.values()))) + 1
#     h_max = list(h_max)
#     value_dict = dict(zip(value_dict.keys(), h_max))
#     for i, ls in enumerate(reversed(grid_list)):
#         if i in value_dict.keys():
#             image[i//column , i%column] = value_dict[i]
#     input2 = image

#     return input1, input2



def rasterization_pcd(npy, xbound, ybound, res, canvas_size):
    # crop pts for estimate ground plane 
    agg_pcd = npy 
    x = agg_pcd[:,0]
    y = agg_pcd[:,1]
    z = agg_pcd[:,2]
    x_values = pd.Series(x)
    y_values = pd.Series(y)
    z_values = pd.Series(z)
    x_in_canvas = x_values.between(0, 10)
    y_in_canvas = y_values.between(-10, 10)
    z_in_canvas = z_values.between(-6, -4) # lamp lidar is 5 m above ground 
    pt_in_canvas = np.array(x_in_canvas) & np.array(y_in_canvas) & np.array(z_in_canvas)
    ptidx = np.argwhere(pt_in_canvas==True)
    filter_pcd = agg_pcd[ptidx, :].reshape((-1,4))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filter_pcd[:,:3])
    ground_plane, inliers = pcd.segment_plane(distance_threshold=0.3, ransac_n=50, num_iterations=1000)
    [a,b,c,d] = ground_plane
    ground_h = -(d+a*agg_pcd[:,0]+b*agg_pcd[:,1]) / c
    # change z values to height difference
    agg_pcd[:,-2] = abs(agg_pcd[:,-2] - ground_h)

    # crop pts  
    npy = agg_pcd
    x = npy[:,0]
    y = npy[:,1]
    z = npy[:,2]
    x_values = pd.Series(x)
    y_values = pd.Series(y)
    z_values = pd.Series(z)
    x_in_canvas = x_values.between(xbound[0], xbound[1])
    y_in_canvas = y_values.between(ybound[0], ybound[1])
    z_in_canvas = z_values.between(0, 0.3)
    pt_in_canvas = np.array(x_in_canvas) & np.array(y_in_canvas) & np.array(z_in_canvas)  
    ptidx = np.argwhere(pt_in_canvas==True)
    pts = npy[ptidx, :].reshape((-1,4))

    # pts => grids        
    xmin = xbound[0]
    ymin = ybound[0]
    pts[:, 0] -= xmin 
    pts[:, 1] -= ymin 
    row = canvas_size[0]
    column = canvas_size[1]
    gridnum = row * column
    grid_list = [[] for i in range(gridnum)]
    for i in range(pts.shape[0]):
        x = pts[i][0]
        y = pts[i][1]
        grid_index = np.floor(x / res) * column + np.floor(y / res)
        if grid_index <= (gridnum-1):
            grid_list[int(grid_index)].append(i)
        else:
            print("Warning: There are points that are not in the canvas!")
    
    # rules of generating images: grids => images
    # check ipad notes for the permutation of girds 
    # INPUT 1: pcd reflection mean
    image = np.zeros((row, column))
    value_dict = {}
    for i, ls in enumerate(reversed(grid_list)): 
        if len(ls) != 0:
            pts_in_grid = pts[ls]
            # TODO try different value rules 
            # value = np.sum(pts_in_grid[:,-1]==6) / pts_in_grid.shape[0]
            value = (pts_in_grid[:,-1]==6).any()
            value_dict[i] = value * 1   
    # pcd_reflt_mean = np.int8(normalization(np.array(list(value_dict.values()))) * 250) + 3 
    pcd_reflt_mean = normalization(np.array(list(value_dict.values())))
    pcd_reflt_mean = list(pcd_reflt_mean)
    value_dict = dict(zip(value_dict.keys(), pcd_reflt_mean))
    for i, ls in enumerate(reversed(grid_list)):
        if i in value_dict.keys():
            image[i//column , i%column] = value_dict[i]
    input1 = image
    
    # INPUT 2: pcd height difference 
    image = np.zeros((row, column))
    value_dict = {}
    for i, ls in enumerate(reversed(grid_list)): 
        if len(ls) != 0:
            pts_in_grid = pts[ls]
            # TODO: max or mean?
            value = np.max(pts_in_grid[:,2])
            value_dict[i] = value
    # h_max = np.int8(normalization(np.array(list(value_dict.values()))) * 250) + 3 
    h_max = normalization(np.array(list(value_dict.values()))) 
    h_max = list(h_max)
    value_dict = dict(zip(value_dict.keys(), h_max))
    for i, ls in enumerate(reversed(grid_list)):
        if i in value_dict.keys():
            image[i//column , i%column] = value_dict[i]
    input2 = image

    return input1, input2


if __name__ == "__main__":
    ############################ First step: initialize dataset #############################
    # from_folder = "/home/yuzeh/transfer/"
    # to_folder = "/media/yuzeh/硬盘/CARLA Dataset/22-12-5/lamp_dataset/"
    # lampsample_list = os.listdir(from_folder)
    # for ls in lampsample_list:
    #     ls_path = os.path.join(from_folder, ls)
    #     if os.path.isdir(ls_path): 
    #         ls_list = os.listdir(ls_path)
    #         if 'rgb' in ls_list and 'lidar' in ls_list:
    #             dst_path = os.path.join(to_folder, ls)
    #             shutil.copytree(ls_path, dst_path)

    # ######################### Second step: aggregate point clouds ########################
    # # from_folder = "/media/yuzeh/硬盘/CARLA_Dataset/22-12-5/lamp_dataset/train/"
    # from_folder = "/media/yuzeh/硬盘/CARLA_Dataset/23-2-5/lamp_collect/"
    # lampsample_list = os.listdir(from_folder)
    # for ls in lampsample_list:
    #     if 'lampslidar' in ls:
    #         print(ls)
    #         pcd_folder = from_folder + ls + "/"
    #         pcd = np.zeros((1,4))
    #         npys = os.listdir(pcd_folder)
    #         for _npy in os.listdir(pcd_folder):
    #             # if not os.path.getsize(pcd_folder+_npy)<200:
    #             try:
    #                 npy = np.array(np.load(pcd_folder+_npy),dtype=np.float32)
    #             except:
    #                 print(_npy)
    #             else:
    #                 pcd = np.vstack((pcd, npy))
    #         pcd = pcd[1:,:]
    #         np.save(from_folder+ls+'/agg_pcd.npy', pcd)


    # ######################### Third step: generate pcd images ###########################
    ''' pcd_reflt_mean & h_max are all normalized to 0~1 '''
    
    # from_folder = "/media/yuzeh/硬盘/CARLA_Dataset/22-12-5/lamp_dataset/train/"
    # to_folder = "/media/yuzeh/硬盘/CARLA_Dataset/22-12-5/lamp_dataset_server/train/"
    from_folder = "/media/yuzeh/硬盘/CARLA_Dataset/23-2-5/lamp_collect_last/"
    to_folder = "/media/yuzeh/硬盘/CARLA_Dataset/23-2-5/lamp_dataset_dense/"
    # some hyperparameters
    xbound=[-15.0, 15.0]
    ybound=[-30.0, 30.0]
    res = 0.15
    canvas_size = [int((xbound[1]-xbound[0])/res), int((ybound[1]-ybound[0])/res)] 
    lampsample_list = os.listdir(from_folder)
    for ls in lampsample_list:
        if 'lampslidar' in ls:
            print(ls)
            try:
                agg_pcd = from_folder + ls + '/10.npy'
            except:
                agg_pcd = from_folder + ls + '/9.npy'
            # agg_pcd = from_folder + ls + '/agg_pcd.npy' 
            agg_pcd = np.load(agg_pcd)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(agg_pcd[:,:3])
            # ground_plane, inliers = pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=1000)
            # agg_pcd = agg_pcd[inliers]
            # [a,b,c,d] = ground_plane
            # ground_h = -(d+a*agg_pcd[:,0]+b*agg_pcd[:,1]) / c
            # agg_pcd[:,-2] = agg_pcd[:,-2] - ground_h
            input1, input2 = rasterization_pcd(agg_pcd, xbound, ybound, res, canvas_size)
            ls_path = to_folder + ls
            mkdir_folder(to_folder, ls)
            cv2.imwrite(os.path.join(ls_path, "pcd_reflt_mean.jpg"), input1*255)
            cv2.imwrite(os.path.join(ls_path, "h_max.jpg"), input2*255)
            np.save(os.path.join(ls_path, "pcd_reflt_mean.npy"), input1) # input1, input2 range: (1,2)
            np.save(os.path.join(ls_path, "h_max.npy"), input2)


#####################################  for check point cloud  ######################################
    # inputpcd = '/media/yuzeh/硬盘/CARLA Dataset/22-12-5/lamp_dataset/15421067722903398922/lidar/8.npy'
    # inputpcd = np.load(inputpcd)[:,:3]

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(inputpcd)
    # o3d.visualization.draw_geometries([pcd],
    #                               zoom=0.3412,
    #                               front=[0.4257, -0.2125, -0.8795],
    #                               lookat=[2.6172, 2.0475, 1.532],
    #                               up=[-0.0694, -0.9768, 0.2024])


'''The following steps are discarded'''

    # ########################## Third step: calculate directions of traj points ###########################
    # from_folder = "/media/yuzeh/硬盘/CARLA Dataset/22-12-5/lamp_dataset/"
    # lampsample_list = os.listdir(from_folder)
    # for ls in lampsample_list:
    #     traj_folder = from_folder + ls + "/traj/"
    #     traj = np.zeros((1,4))
    #     for traji in os.listdir(traj_folder):
    #         traji = np.loadtxt(traj_folder+traji)
    #         if traji.shape[0] > 0:
    #             # calculate directions
    #             directions = []
    #             # TODO: handle drct -- 361 
    #             for x in range(traji.shape[0]-1):
    #                 vct = traji[x+1,:2]-traji[x,:2]
    #                 if (vct == 0).all() and len(directions) != 0:
    #                     drct = directions[-1]
    #                 elif(vct == 0).all() and len(directions) == 0:
    #                     drct = 361
    #                 else:
    #                     vct0 = np.array([1,0])
    #                     drct = CalAngle(vct0, vct)
    #                 directions.append(drct)
    #             assert len(directions) == traji.shape[0]-1
    #             directions.append(directions[-1])
    #             directions = np.array(directions).reshape((-1,1))
    #             traji = np.hstack((traji[:,:2], traji[:,6].reshape(-1,1), directions)) # x,y,frameidx(=time),direction
    #             traj = np.vstack((traj, traji))
    #     traj = traj[1:,:]
    #     np.savetxt(from_folder+ls+'/agg_traj.txt', traj)


    ########################## Fourth step: check the traj in lamp ######################
    # from_folder = "/media/yuzeh/硬盘/CARLA Dataset/22-12-5/lamp_dataset/"
    # lampsample_list = os.listdir(from_folder)
    # for ls in lampsample_list:
    #     ls_path = from_folder + ls
    #     agg_traj = from_folder + ls + '/agg_traj.txt' 
    #     pcd_1frame = from_folder + ls + '/lidar/10.npy'
    #     agg_traj = np.loadtxt(agg_traj)[:,:3]
    #     agg_traj = np.hstack((agg_traj, 5*np.ones((agg_traj.shape[0], 1))))
    #     cloud = pcl.PointCloud_PointXYZI()
    #     cloud.from_array(agg_traj.astype(np.float32))
    #     pcl.save(cloud, from_folder + ls + "/agg_traj.pcd")
    #     pcd = np.array(np.load(pcd_1frame),dtype=np.float32)[:,:3]
    #     pcd = np.hstack((pcd, 20*np.ones((pcd.shape[0], 1))))
    #     cloud = pcl.PointCloud_PointXYZI()
    #     cloud.from_array(pcd.astype(np.float32))
    #     pcl.save(cloud, from_folder + ls + "/pcd_10.pcd")
    #     com = np.vstack((agg_traj, pcd))
    #     cloud = pcl.PointCloud_PointXYZI()
    #     cloud.from_array(com.astype(np.float32))
    #     pcl.save(cloud, from_folder + ls + "/combination.pcd")
    


    # ######################### Fourth step: generate traj images #######################
    # from_folder = "/media/yuzeh/硬盘/CARLA Dataset/22-12-5/lamp_dataset/"
    # # some hyperparameters
    # xbound=[-60.0, 60.0]
    # ybound=[-60.0, 60.0]
    # res = 0.15
    # canvas_size = [int((xbound[1]-xbound[0])/res), int((ybound[1]-ybound[0])/res)] 
    # lampsample_list = os.listdir(from_folder)
    # for ls in lampsample_list:
    #     ls_path = from_folder + ls
    #     agg_traj = from_folder + ls + '/agg_traj.txt' 
    #     agg_traj = np.loadtxt(agg_traj)
    #     traj_density, traj_dir_mean, traj_dir_var = rasterization_traj(agg_traj, xbound, ybound, res, canvas_size)
        # cv2.imwrite(os.path.join(ls_path, "traj_density.jpg"), traj_density)
        # cv2.imwrite(os.path.join(ls_path, "traj_dir_mean.jpg"), traj_dir_mean)
        # cv2.imwrite(os.path.join(ls_path, "traj_dir_var.jpg"), traj_dir_var)




    



    
        


