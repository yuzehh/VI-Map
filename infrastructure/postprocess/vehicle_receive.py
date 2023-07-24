import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
from scipy.special import comb
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN


def list_txt(path, list=None):
    '''
    :param path: 储存list的位置
    :param list: list数据
    :return: None/relist 当仅有path参数输入时为读取模式将txt读取为list
             当path参数和list都有输入时为保存模式将list保存为txt
    '''
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist


def bernstein_poly(i, n, t):
# The Bernstein polynomial of n, i as a function of t
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=50):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

  
  
def DT_weight(lcpts, vcpts, newmap_size=[1200, 600]):
    newmap = np.ones((newmap_size[0], newmap_size[1]), dtype=np.uint8) * 255
    lxvals, lyvals = bezier_curve(lcpts, nTimes=100)
    vxvals, vyvals = bezier_curve(vcpts, nTimes=100)
    marginx, marginy = newmap_size/3
    newmap[lxvals+marginx, lyvals+marginy] = 0
    newmap[vxvals+marginx, vyvals+marginy] = 0
    newmap = cv2.distanceTransform(newmap, cv2.DIST_L2, 3)
    
    return newmap



def f_try(xdata,ydata,m):
    data = [ [m[0],m[1]], [m[2],m[3]], [m[4],m[5]], [m[6],m[7]] ]
    xvals, yvals = bezier_curve(data, nTimes=1000)
    xyvals = np.hstack((xvals.reshape(-1,1), yvals.reshape(-1,1)))
    xydata = np.hstack((xdata.reshape(-1,1), ydata.reshape(-1,1)))
    error = []
    for i in range(xydata.shape[0]):
        xy = xydata[i]
        distances = np.sqrt( np.sum( np.asarray(xyvals-xy)**2, axis=1 ) )
        error.append(np.min(distances)) 
    return np.array(error)

def term1(xdata,ydata,wdata,m):
    # TODO: Two chocies: f() and f_try()
    # error = np.sum( wdata * ( ydata - f(xdata,m) ) **2 )
    error = np.sum( wdata * f_try(xdata,ydata,m) )
    return error

def fun_bdy(m, *args):
    xdata, ydata, weight, trj_para = args
    # TODO add term 2 or not
    # v = term1(xdata, ydata, weight, m) + term2_bdy(trj_para,m)
    v = term1(xdata, ydata, weight, m) 
    return v

# TODO lcpts or vcpts 
def fit_new_curve(img, vcpts):
    weight = (255 - img) / 255 # weight between 0-1
    trj_para = np.array(vcpts).reshape(-1)
    wmean = np.mean(weight)
    wpts = np.where(weight > wmean)
    xdata = wpts[1]
    ydata = wpts[0]
    # TODO check the index method right or not 
    weight = weight[wpts[0],wpts[1]]
    init_guess = trj_para
    res = minimize(fun_bdy, init_guess, args=(xdata, ydata, weight, trj_para), method='SLSQP')
    return res
     
    

def main(vpos, lpos, vmap_cpts, lmap_cpts, vmap_size, lmap_size, margin=80):
    # 1. find rotation/translation 
    # 2. plot lmap and vmap (with width) 
    # 3. rotate/translate l to v 
    # 4. Distance Transform
    # 5. Fit bezier curve 
    vx, vy, vz, vpitch, vyaw, vroll = vpos[:6]
    vy = -vy
    lx, ly, lz = lpos[:3]
    ly = -ly
    lpitch, lyaw, lroll = lpos[12:]
    
    # xmm = [vx-80, vx+80, lx-80, lx+80]
    # ymm = [vy-80, vy+80, ly-80, ly+80]
    # xlim = [min(xmm), max(xmm)]
    # ylim = [min(ymm), max(ymm)]
    
    # the following code are revised from generate_lamp_GT.py
    lx = -111
    ly = 65
    lyaw = 0
    map_xlim = (-287, 222)
    map_ylim = (-220, 219)
    imgsize = (3393,2926)
    (xmin,xmax) = map_xlim
    xrange = xmax-xmin
    (ymin,ymax) = map_ylim
    yrange = ymax-ymin
    (w,h) = imgsize
    # for lamppost map 
    angle = 90 + lyaw 
    center = ( (lx-xmin)*w/xrange+1000, (ymax-ly)*h/yrange+1000 ) 
    Mr = cv2.getRotationMatrix2D(center, angle, scale=1)
    Mr = np.vstack(Mr, np.array([0,0,1]).reshape((1,3)))
    center = ( round(center[0]), round(center[1]) )
    w = center[0]-400
    h = center[1]-200
    Mt = np.array([[1,0,-w], [0,1,-h], [0,0,1]])
    Ml = Mt @ Mr
    Ml = np.linalg.inv(Ml)
    # for vehicle map 
    angle = 90 + vyaw
    center = ( (vx-xmin)*w/xrange+1000, (ymax-vy)*h/yrange+1000 ) 
    Mr = cv2.getRotationMatrix2D(center, angle, scale=1)
    Mr = np.vstack(Mr, np.array([0,0,1]).reshape((1,3)))
    # a = Mr @ np.array([center[0],center[1], 1]).reshape((3,1))
    center = ( round(center[0]), round(center[1]) )
    w = center[0]-200
    h = center[1]-100
    Mt = np.array([[1,0,-w], [0,1,-h], [0,0,1]])
    Mv = Mt @ Mr
    Ml2v = Mv @ Ml
    # transform control points in lamp map to vehicle map 
    lcptsinv = []
    crspd_vcpts = []
    for i,lcpts in enumerate(lmap_cpts):
        lcpts = np.hstack( lcpts, np.ones((lcpts.shape[0],1)) ).T
        lcpts = Ml2v @ lcpts # 3*n
        lcpts = lcpts.T[:,:2]
        lcptsinv.append(lcpts)
        xvals, yvals = bezier_curve(lcpts, nTimes=100)
        curpts = np.hstack((xvals.reshape(-1,1), yvals.reshape(-1,1)))
        # find instance correspondence -- nearest method
        min_diss = []
        for j,vcpts in enumerate(vmap_cpts):
            vstart = vcpts[0]
            vend = vcpts[-1]
            dis2start = np.sqrt( np.sum(np.asarray(vstart-curpts)**2, axis=1) )
            dis2end = np.sqrt( np.sum(np.asarray(vend-curpts)**2, axis=1) )
            min_dis = min((min(dis2start), min(dis2end)))
            min_diss.append(min_dis)
        crspd_vid = min_diss.index(min(min_diss))  
        crspd_vcpts.append(vmap_cpts[crspd_vid])  
    # instance pair fusion => new bezier curves
    newcurvs = []
    for i,lcpts in enumerate(lcptsinv):
        vcpts = crspd_vcpts[i]
        # Distance transform to assign weights to grids
        img = DT_weight(lcpts, vcpts, newmap_size=[1200, 600])
        # solve optimization function 
        newcpts = fit_new_curve(img, vcpts)
        newcurvs.append(newcurvs)
   
   

if __name__ == '__main__':
    # # TODO: assume all vehicle data to be evaluated is in folder 'vfolder' 
    # vfolder = '/'
    # lamp_dataset_path  = '/'
    # vmapfolder = '/'
    # lmapfolder = '/'
    # vxbound=[-30.0, 30.0]
    # vybound=[-15.0, 15.0]
    # res = 0.15
    # vmap_size = [int((vxbound[1]-vxbound[0])/res), int((vybound[1]-vybound[0])/res)]
    # lxbound=[-30.0, 30.0]
    # lybound=[-60.0, 60.0]
    # lmap_size = [int((lxbound[1]-lxbound[0])/res), int((lybound[1]-lybound[0])/res)]
    # ltrain_list = os.listdir(lamp_dataset_path+'train/')
    # lval_list = os.listdir(lamp_dataset_path+'val/')
    # items = os.listdir(vfolder)
    # for item in items:
    #     vpos = np.loadtxt(vfolder + item + '/' + 'vehicle_location.txt')
    #     with open(vfolder + item + '/' + 'correspond_lampid.txt', 'r') as file:
    #         crspd_l = file.read().replace('\n', '')
    #     if crspd_l in ltrain_list:
    #         lpos = np.loadtxt(lamp_dataset_path + 'train/' + crspd_l + '/lamp_location.txt') 
    #     elif crspd_l in lval_list:
    #         lpos = np.loadtxt(lamp_dataset_path + 'val/' + crspd_l + '/lamp_location.txt')
    #     else:
    #         print(item)
    #         print("Can not find the corresponding lamp data!")
    #     vmappath = vmapfolder + item + '.txt'
    #     lmappath = lmapfolder + crspd_l + '.txt'
    #     vmap_cpts = list_txt(vmappath)
    #     lmap_cpts = list_txt(lmappath)
    #     main(vpos, lpos, vmap_cpts, lmap_cpts, vmap_size, lmap_size)




