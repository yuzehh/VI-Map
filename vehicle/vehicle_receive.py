import math
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import BSpline, splev, splrep
from scipy.optimize import minimize
from scipy.special import comb
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def list_txt(path, list=None):
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
    # error = np.sum( wdata * ( ydata - f(xdata,m) ) **2 )
    error = np.sum( wdata * f_try(xdata,ydata,m) )
    return error

def fun_bdy(m, *args):
    xdata, ydata, weight, trj_para = args
    # v = term1(xdata, ydata, weight, m) + term2_bdy(trj_para,m)
    v = term1(xdata, ydata, weight, m) 
    return v

def fit_new_curve(img, vcpts):
    weight = (255 - img) / 255 # weight between 0-1
    trj_para = np.array(vcpts).reshape(-1)
    wmean = np.mean(weight)
    wpts = np.where(weight > wmean)
    xdata = wpts[1]
    ydata = wpts[0]
    weight = weight[wpts[0],wpts[1]]
    init_guess = trj_para
    res = minimize(fun_bdy, init_guess, args=(xdata, ydata, weight, trj_para), method='SLSQP')
    return res
     

def get_bspline_parameters(ts, ys, knots_n):
    # Fit
    knots_n = 5
    qs = np.linspace(0, 1, knots_n+2)[1:-1]
    knots = np.quantile(ts, qs)
    try:
        tck = splrep(ts, ys, t=knots, k=3)
        ys_smooth = splev(ts, tck)
    except:
        tck=None
        ys_smooth = None

    # Alternative if one really wants to use BSpline: 
    # ys_smooth = BSpline(*tck)(ts)

    # # Display
    # plt.figure(figsize=(12, 6))
    # plt.plot(ts, ys, '.c')
    # plt.plot(ts, ys_smooth, '-m')
    # plt.show()

    return tck, ys_smooth

def get_bezier_parameters(X, Y, degree=4):
    """ Least square bezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    """
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(X) < degree + 1:
        raise ValueError('Not enough points.')

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)
        #return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))
    
    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]

    return final


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def main_bazier(vpos, lpos, vmap_cpts, lmap_cpts, vmap_size, lmap_size, margin=80):
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
    map_xlim = (-287, 222)
    map_ylim = (-220, 219)
    imgsize = (3393,2926)
    (xmin,xmax) = map_xlim
    xrange = xmax-xmin
    (ymin,ymax) = map_ylim
    yrange = ymax-ymin
    (mapw,maph) = imgsize
    # for lamppost map 
    angle = 90 + lyaw 
    center = ( (lx-xmin)*mapw/xrange+1000, (ymax-ly)*maph/yrange+1000 ) 
    Mr = cv2.getRotationMatrix2D(center, angle, scale=1)
    Mr = np.vstack((Mr, np.array([0,0,1]).reshape((1,3))))
    center = ( round(center[0]), round(center[1]) )
    w = center[0]-400
    h = center[1]-200
    # w = center[1]-400
    # h = center[0]-200
    Mt = np.array([[1,0,-w], [0,1,-h], [0,0,1]])
    Ml = Mt @ Mr
    Ml = np.linalg.inv(Ml)

    # for vehicle map
    angle = 90 + vyaw
    center = ( (vx-xmin)*mapw/xrange+1000, (ymax-vy)*maph/yrange+1000 ) 
    Mr = cv2.getRotationMatrix2D(center, angle, scale=1)
    Mr = np.vstack((Mr, np.array([0,0,1]).reshape((1,3))))
    # a = Mr @ np.array([center[0],center[1], 1]).reshape((3,1))
    center = ( round(center[0]), round(center[1]) )
    w = center[0]-100
    h = center[1]-200
    # w = center[1]-100
    # h = center[0]-200
    Mt = np.array([[1,0,-w], [0,1,-h], [0,0,1]])
    Mv = Mt @ Mr
    Ml2v = Mv @ Ml

    # debug
    # corner_w = np.array([[0, 0, 1], [3393, 0, 1], [0, 2926, 1], [3393, 2926, 1]])
    # scale = np.array([1, 10, 20, 30])
    # corner_l = np.array([[0, 0, 1], [200, 0, 1], [0, 400, 1], [200, 400, 1]])
    
    # corner_l2w =  (np.linalg.inv(Mv) @ corner_l.T).T
    # plt.scatter(corner_w[:, 0], corner_w[:, 1], scale)
    # plt.scatter(corner_l2w[:, 0], corner_l2w[:, 1], scale)
    # plt.scatter([center[0]], [center[1]])
    # plt.show()


    # # check transformation right or not 
    # # Plot the lamp control points
    lcptsinv = []
    for l in lmap_cpts: 
        for i,lcpts in enumerate(l):
                lcpts = np.array(lcpts)
                lcpts = np.hstack( (lcpts[:, ::-1], np.ones((lcpts.shape[0],1))) ).T
                lcpts = Ml2v @ lcpts # 3*n
                lcpts = lcpts.T[:,:2]
                lcptsinv.append(lcpts)
    for i,lcpts in enumerate(lcptsinv):
        # Plot the lamp control points
        # x_val = [x[0] for x in lcpts]
        # y_val = [x[1] for x in lcpts]
        # plt.plot(x_val,y_val,'k--o', label='Lamp Control Points')
        # Plot the transformed lamp Bezier curve
        xvals, yvals = bezier_curve(lcpts, nTimes=1000)
        # plt.plot(xvals, yvals, 'b-', label='lamp B Curve')
        plt.plot(xvals, yvals, 'b-')

    for v in vmap_cpts: 
        for i,vcpts in enumerate(v):
            # Plot the lamp control points
            # x_val = [x[0] for x in vcpts]
            # y_val = [x[1] for x in vcpts]
            # plt.plot(x_val,y_val,'k--o', label='Vehicle Control Points')
            # Plot the transformed lamp Bezier curve
            xvals, yvals = bezier_curve(np.array(vcpts)[:, ::-1], nTimes=1000)
            # plt.plot(xvals, yvals, 'r-', label='vehicle B Curve')
            plt.plot(xvals, yvals, 'r-')
    plt.legend()
    plt.show()  
    
    # x_val = [x[0] for x in lcpts]
    # y_val = [x[1] for x in lcpts]
    # plt.plot(x_val,y_val,'k--o', label='Lamp Control Points')
    # # Plot the transformed lamp Bezier curve
    # xvals, yvals = bezier_curve(lcpts, nTimes=1000)
    # plt.plot(xvals, yvals, 'b-', label='lamp B Curve')
    # # Plot the vehicle control points

    # # Plot the vehicle Bezier curve
    # plt.legend()
    # plt.show()

    # # transform control points in lamp map to vehicle map 
    # lcptsinv = []
    # crspd_vcpts = []
    # for i,lcpts in enumerate(lmap_cpts):
    #     lcpts = np.hstack( lcpts, np.ones((lcpts.shape[0],1)) ).T
    #     lcpts = Ml2v @ lcpts # 3*n
    #     lcpts = lcpts.T[:,:2]
    #     lcptsinv.append(lcpts)
    #     xvals, yvals = bezier_curve(lcpts, nTimes=100)
        

    #     curpts = np.hstack((xvals.reshape(-1,1), yvals.reshape(-1,1)))
    #     # find instance correspondence -- nearest method
    #     min_diss = []
    #     for j,vcpts in enumerate(vmap_cpts):
    #         vstart = vcpts[0]
    #         vend = vcpts[-1]
    #         dis2start = np.sqrt( np.sum(np.asarray(vstart-curpts)**2, axis=1) )
    #         dis2end = np.sqrt( np.sum(np.asarray(vend-curpts)**2, axis=1) )
    #         min_dis = min((min(dis2start), min(dis2end)))
    #         min_diss.append(min_dis)
    #     crspd_vid = min_diss.index(min(min_diss))  
    #     crspd_vcpts.append(vmap_cpts[crspd_vid])  
    # # instance pair fusion => new bezier curves
    # newcurvs = []
    # for i,lcpts in enumerate(lcptsinv):
    #     vcpts = crspd_vcpts[i]
    #     # Distance transform to assign weights to grids
    #     img = DT_weight(lcpts, vcpts, newmap_size=[1200, 600])
    #     # solve optimization function 
    #     newcpts = fit_new_curve(img, vcpts)
    #     newcurvs.append(newcurvs)

def main_bspline(vpos, lpos, vmap_cpts, lmap_cpts, vmap_size, lmap_size, margin=80):
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
    map_xlim = (-287, 222)
    map_ylim = (-220, 219)
    imgsize = (3393,2926)
    (xmin,xmax) = map_xlim
    xrange = xmax-xmin
    (ymin,ymax) = map_ylim
    yrange = ymax-ymin
    (mapw,maph) = imgsize
    # for lamppost map 
    angle = 90 + lyaw 
    center = ( (lx-xmin)*mapw/xrange+1000, (ymax-ly)*maph/yrange+1000 ) 
    Mr = cv2.getRotationMatrix2D(center, angle, scale=1)
    Mr = np.vstack((Mr, np.array([0,0,1]).reshape((1,3))))
    center = ( round(center[0]), round(center[1]) )
    w = center[0]-400
    h = center[1]-200
    # w = center[1]-400
    # h = center[0]-200
    Mt = np.array([[1,0,-w], [0,1,-h], [0,0,1]])
    Ml = Mt @ Mr
    Ml = np.linalg.inv(Ml)

    # for vehicle map
    angle = 90 + vyaw
    center = ( (vx-xmin)*mapw/xrange+1000, (ymax-vy)*maph/yrange+1000 ) 
    Mr = cv2.getRotationMatrix2D(center, angle, scale=1)
    Mr = np.vstack((Mr, np.array([0,0,1]).reshape((1,3))))
    # a = Mr @ np.array([center[0],center[1], 1]).reshape((3,1))
    center = ( round(center[0]), round(center[1]) )
    w = center[0]-100
    h = center[1]-200
    # w = center[1]-100
    # h = center[0]-200
    Mt = np.array([[1,0,-w], [0,1,-h], [0,0,1]])
    Mv = Mt @ Mr
    Ml2v = Mv @ Ml

    # # check transformation right or not 
    # # Plot the lamp control points
    lcptsinv = []
    for l in lmap_cpts: 
        for i,lcpts in enumerate(l):
                ts = np.linspace(lcpts[0][0], lcpts[0][-1], 50)
                ys_smooth = splev(ts, lcpts)
                lcpts = np.hstack((ys_smooth.reshape((-1,1)), ts.reshape((-1,1))))
                lcpts = np.hstack( (lcpts, np.ones((lcpts.shape[0],1))) ).T
                lcpts = Ml2v @ lcpts # 3*n
                lcpts = lcpts.T[:,:2]
                lcptsinv.append(lcpts)
    for i,lcpts in enumerate(lcptsinv):
        # Plot the lamp control points
        # x_val = [x[0] for x in lcpts]
        # y_val = [x[1] for x in lcpts]
        # plt.plot(x_val,y_val,'k--o', label='Lamp Control Points')
        # Plot the transformed lamp Bezier curve
        # plt.plot(xvals, yvals, 'b-', label='lamp B Curve')
        plt.plot(lcpts[:,0], lcpts[:,1], 'b-')

    for v in vmap_cpts: 
        for i,vcpts in enumerate(v):
            # Plot the lamp control points
            # x_val = [x[0] for x in vcpts]
            # y_val = [x[1] for x in vcpts]
            # plt.plot(x_val,y_val,'k--o', label='Vehicle Control Points')
            # Plot the transformed lamp Bezier curve
            ts = np.linspace(vcpts[0][0], vcpts[0][-1], 50)
            ys_smooth = splev(ts, vcpts)
            # plt.plot(xvals, yvals, 'r-', label='vehicle B Curve')
            plt.plot(ys_smooth, ts, 'r-')
    plt.legend()
    plt.show()  
    

def list_map_instances(map, Ml2v=None):
    if Ml2v is not None:
        lmap = map
        ll_instances = []
        cw_instances = []
        ct_instances = []
        ll_label = np.unique(lmap[0])
        cw_label = np.unique(lmap[1])
        ct_label = np.unique(lmap[2])
        for l in ll_label:
            if l != 0:
                pixels = np.where(lmap[0]==l)
                pixels = np.array(pixels).T
                pixels = np.hstack( (pixels[:, ::-1], np.ones((pixels.shape[0],1))) ).T
                start = time.time()
                rotpixels = Ml2v @ pixels # 3*n
                stop = time.time()
                print('------ transformation time -------')
                print(stop-start)
                rotpixels = rotpixels.T[:,:2]
                ll_instances.append(rotpixels)
        for l in cw_label:
            if l != 0:
                pixels = np.where(lmap[1]==l)
                pixels = np.array(pixels).T
                pixels = np.hstack( (pixels[:, ::-1], np.ones((pixels.shape[0],1))) ).T
                rotpixels = Ml2v @ pixels # 3*n
                rotpixels = rotpixels.T[:,:2]
                cw_instances.append(rotpixels)
        for l in ct_label:
            if l != 0:
                pixels = np.where(lmap[2]==l)
                pixels = np.array(pixels).T
                pixels = np.hstack( (pixels[:, ::-1], np.ones((pixels.shape[0],1))) ).T
                rotpixels = Ml2v @ pixels # 3*n
                rotpixels = rotpixels.T[:,:2]
                ct_instances.append(rotpixels)

    elif Ml2v is None:
        vmap = map
        ll_instances = []
        cw_instances = []
        ct_instances = []
        ll_label = np.unique(vmap[0])
        cw_label = np.unique(vmap[1])
        ct_label = np.unique(vmap[2])
        for l in ll_label:
            if l != 0:
                pixels = np.where(vmap[0]==l)
                pixels = np.array(pixels).T
                pixels = pixels[:, ::-1]
                ll_instances.append(pixels)
        for l in cw_label:
            if l != 0:
                pixels = np.where(vmap[1]==l)
                pixels = np.array(pixels).T
                pixels = pixels[:, ::-1]
                cw_instances.append(pixels)
        for l in ct_label:
            if l != 0:
                pixels = np.where(vmap[2]==l)
                pixels = np.array(pixels).T
                pixels = pixels[:, ::-1]
                ct_instances.append(pixels)

    return ll_instances, cw_instances, ct_instances


def get_Ml2v(vpos, lpos):
    vx, vy, vz, vpitch, vyaw, vroll = vpos[:6]
    vy = -vy
    lx, ly, lyaw = lpos[:3]
    ly = -ly
    map_xlim = (-287, 222)
    map_ylim = (-220, 219)
    imgsize = (3393,2926)
    (xmin,xmax) = map_xlim
    xrange = xmax-xmin
    (ymin,ymax) = map_ylim
    yrange = ymax-ymin
    (mapw,maph) = imgsize
    # for lamppost map 
    angle = 90 + lyaw 
    center = ( (lx-xmin)*mapw/xrange+1000, (ymax-ly)*maph/yrange+1000 ) 
    Mr = cv2.getRotationMatrix2D(center, angle, scale=1)
    Mr = np.vstack((Mr, np.array([0,0,1]).reshape((1,3))))
    center = ( round(center[0]), round(center[1]) )
    # w = center[0]-400
    # h = center[1]-200
    w = center[0]-200
    h = center[1]-100
    Mt = np.array([[1,0,-w], [0,1,-h], [0,0,1]])
    Ml = Mt @ Mr
    Ml = np.linalg.inv(Ml)

    # for vehicle map
    angle = 90 + vyaw
    center = ( (vx-xmin)*mapw/xrange+1000, (ymax-vy)*maph/yrange+1000 ) 
    Mr = cv2.getRotationMatrix2D(center, angle, scale=1)
    Mr = np.vstack((Mr, np.array([0,0,1]).reshape((1,3))))
    center = ( round(center[0]), round(center[1]) )
    # w = center[0]-100
    # h = center[1]-200
    w = center[0]-80
    h = center[1]-120
    Mt = np.array([[1,0,-w], [0,1,-h], [0,0,1]])
    Mv = Mt @ Mr
    Ml2v = Mv @ Ml

    return Ml2v


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist



def main(vpos, lpos, vmap, lmap):
    Ml2v = get_Ml2v(vpos, lpos)
    # lamp map
    lll_instances, lcw_instances, lct_instances = list_map_instances(lmap, Ml2v=Ml2v)
    for l in lll_instances:
        plt.scatter(l[:,0], l[:,1], c='r')
    for l in lcw_instances:
        plt.scatter(l[:,0], l[:,1], c='g')
    for l in lct_instances:
        plt.scatter(l[:,0], l[:,1], c='b')
    # vehicle map 
    vll_instances, vcw_instances, vct_instances = list_map_instances(vmap)
    for l in vll_instances:
        plt.scatter(l[:,0], l[:,1], c='r')
    for l in vcw_instances:
        plt.scatter(l[:,0], l[:,1], c='g')
    for l in vct_instances:
        plt.scatter(l[:,0], l[:,1], c='b')
    plt.legend()
    plt.show()  
    fusion_ll = []
    fusion_cw = []
    fusion_ct = []
    # for use bspine
    # for v in vll_instances: 
    #     dis = [ chamfer_distance(v,l) for l in lll_instances]
    #     l = lll_instances[np.argmin(dis)]
    #     new_instance = np.vstack((v,l))
    #     data = rotate_fit_bspline(new_instance)
    #     fusion_ll.append(data)
    # for v in vcw_instances: 
    #     dis = [ chamfer_distance(v,l) for l in lcw_instances]
    #     l = lcw_instances[np.argmin(dis)]
    #     new_instance = np.vstack((v,l))
    #     fusion_cw.append(new_instance)
    # for v in vct_instances: 
    #     dis = [ chamfer_distance(v,l) for l in lct_instances]
    #     l = lct_instances[np.argmin(dis)]
    #     new_instance = np.vstack((v,l))
    #     data = rotate_fit_bspline(new_instance)
    #     fusion_ct.append(data)
    # for use bazier 
    for v in vll_instances: 
        dis = [ chamfer_distance(v,l) for l in lll_instances]
        l = lll_instances[np.argmin(dis)]
        lll_instances.pop(np.argmin(dis))
        new_instance = np.vstack((v,l))
        data = rotate_fit_bspline(new_instance)
        # data = rotate_fit_bazier(new_instance)
        # xvals, yvals = bezier_curve(data, nTimes=1000)
        # data = np.hstack(( xvals.reshape((-1,1)), yvals.reshape((-1,1)) ))
        fusion_ll.append(data)
    for v in vcw_instances: 
        dis = [ chamfer_distance(v,l) for l in lcw_instances]
        l = lcw_instances[np.argmin(dis)]
        lcw_instances.pop(np.argmin(dis))
        new_instance = np.vstack((v,l))
        fusion_cw.append(new_instance)
    for v in vct_instances: 
        dis = [ chamfer_distance(v,l) for l in lct_instances]
        l = lct_instances[np.argmin(dis)]
        lct_instances.pop(np.argmin(dis))
        new_instance = np.vstack((v,l))
        data = rotate_fit_bspline(new_instance)
        # data = rotate_fit_bazier(new_instance)
        # xvals, yvals = bezier_curve(data, nTimes=1000)
        # data = np.hstack(( xvals.reshape((-1,1)), yvals.reshape((-1,1)) ))
        fusion_ct.append(data)

    for l in fusion_ll:
        plt.plot(l[:,0], l[:,1], 'r-')
    for l in fusion_cw:
        plt.plot(l[:,0], l[:,1], 'g-')
    for l in fusion_ct:
        plt.plot(l[:,0], l[:,1], 'b-')
    for l in lll_instances:
        # data = rotate_fit_bazier(l)
        # xvals, yvals = bezier_curve(data, nTimes=1000)
        # plt.plot(xvals, yvals, 'r-')
        data = rotate_fit_bspline(l)
        plt.plot(data[:,0], data[:,1], 'r-')
    for l in lcw_instances:
        plt.plot(l[:,0], l[:,1], 'g-')
    for l in lct_instances:
        # data = rotate_fit_bazier(l)
        # xvals, yvals = bezier_curve(data, nTimes=1000)
        # plt.plot(xvals, yvals, 'b-')
        data = rotate_fit_bspline(l)
        plt.plot(data[:,0], data[:,1], 'b-')

    plt.legend()
    plt.show() 

    return fusion_ll, fusion_cw, fusion_ct  



def rotate_fit_bspline(pts):
    theta = [math.radians(10), math.radians(45), math.radians(100)]
    e_errors = []
    ctrl_pts = []
    for e in theta:
        start = time.time()
        M = np.array([[math.cos(e), -math.sin(e)], [math.sin(e), math.cos(e)]])
        rot_pts = M @ (pts.T)
        rot_pts = rot_pts.T
        rot_pts = rot_pts[np.argsort(rot_pts[:, 0])]
        tck, y_smooth = get_bspline_parameters(rot_pts[:, 0], rot_pts[:, 1], knots_n=3)
        stop = time.time()
        print('------ refitting time -------')
        print(stop-start)
        # get fitting errors 
        # if y_smooth is not None:
        error = np.sum(np.abs(y_smooth-rot_pts[:, 1]))
        e_errors.append(error)
        # transform control points back
        curve = np.hstack(( rot_pts[:,0].reshape(-1,1), y_smooth.reshape(-1,1) ))
        data =  np.linalg.inv(M) @ (np.array(curve).T)
        ctrl_pts.append(data.T)
    idx = np.argmin(np.array(e_errors))
    data = ctrl_pts[idx]
    # # plot & check 
    # # Plot the control points
    # x_val = [x[0] for x in data]
    # y_val = [x[1] for x in data]
    # plt.plot(x_val, y_val, 'b-', label='B SplineCurve')
    # # plot original points
    # plt.scatter(pts[:,0], pts[:,1], c='r', label='original points')
    # plt.legend()
    # plt.show()
    
    return data


def rotate_fit_bazier(pts):
    theta = [math.radians(0), math.radians(45), math.radians(90)]
    e_errors = []
    ctrl_pts = []
    for e in theta:
        M = np.array([[math.cos(e), -math.sin(e)], [math.sin(e), math.cos(e)]])
        rot_pts = M @ (pts.T)
        rot_pts = rot_pts.T
        rot_pts = rot_pts[np.argsort(rot_pts[:, 0])]
        data = get_bezier_parameters(rot_pts[:, 0], rot_pts[:, 1], degree=3)
        # get fitting errors 
        xvals, yvals = bezier_curve(data, nTimes=1000)
        curve = np.hstack((xvals.reshape((-1,1)), yvals.reshape((-1,1))))
        error = 0
        for rotpt in rot_pts:
            distances = np.sqrt(np.sum(np.asarray(rotpt-curve)**2, axis=1))
            dis = np.min(distances)
            error += dis
        e_errors.append(error)
        # transform control points back
        data =  np.linalg.inv(M) @ (np.array(data).T)
        ctrl_pts.append(data.T)
    idx = np.argmin(np.array(e_errors))
    data = ctrl_pts[idx]
    # # plot & check 
    # # Plot the control points
    # x_val = [x[0] for x in data]
    # y_val = [x[1] for x in data]
    # plt.plot(x_val,y_val,'k--o', label='Control Points')
    # # Plot the resulting Bezier curve
    # xvals, yvals = bezier_curve(data, nTimes=1000)
    # plt.plot(xvals, yvals, 'b-', label='B Curve')
    # # plot original points
    # plt.scatter(pts[:,0], pts[:,1], c='r', label='original points')
    # plt.legend()
    # plt.show()
    
    return data

def samecat_cluster(coords, eps=2, minsample=1):
    simplified_coords = []
    db = DBSCAN(eps=eps, min_samples=minsample)
    db.fit(coords)
    db_labels = db.labels_
    unique_labels = np.unique(db_labels)
    for index, label in enumerate(unique_labels.tolist()):
        if label == -1:
            continue
        idx = np.where(db_labels == label)
        simplified_coords.append(coords[idx])
    return simplified_coords
            

def netouput_fit(coords, lists):
    coords = np.load(coords, allow_pickle=True)
    lists = np.load(lists, allow_pickle=True)
    color = {1: 'r', 2: 'g', 3: 'b'}
    vlli = []
    vcti = []
    vcw_instances = []
    for i,c in enumerate(coords):
        cat = lists[i]
        if cat ==1:
            # plt.scatter(c[:,0], c[:,1], c=color[cat])
            vll_instances = samecat_cluster(c)
            for ins in vll_instances:
                ins = ins[:, ::-1]
                data = rotate_fit_bspline(ins)
                vlli.append(data)
                plt.plot(data[:,0], data[:,1], color[cat])
        if cat ==3:
            vct_instances = samecat_cluster(c)
            for ins in vct_instances:
                ins = ins[:, ::-1]
                data = rotate_fit_bspline(ins)
                vcti.append(data)
                plt.plot(data[:,0], data[:,1], color[cat])
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('vout_lines.png', bbox_inches='tight')
    plt.show()
    plt.close()

    img = cv2.imread('vout_lines.png')
    img = np.rot90(img, 2)
    img = cv2.flip(img, 1)
    img = np.ones_like(img) * 255
    for i,c in enumerate(coords):
        cat = lists[i]
        if cat==2:
            vcw_instances = samecat_cluster(c)
            for ins in vcw_instances:
                ins = ins[:, ::-1]
                rect = cv2.minAreaRect(ins)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                img = cv2.fillConvexPoly(img, box, color=(0,255,0))
    cv2.imwrite('vout_fit.png', img)

    return vlli, vcti, vcw_instances


# class data_linewidth_plot():
#     def __init__(self, x, y, **kwargs):
#         self.ax = kwargs.pop("ax", plt.gca())
#         # self.fig = self.ax.get_figure()
#         self.fig = kwargs.pop("fig")
#         self.lw_data = kwargs.pop("linewidth", 1)
#         self.lw = 1
#         self.fig.canvas = kwargs.pop("canvas")
#         self.fig.canvas.draw()

#         self.ppd = 72./self.fig.dpi
#         self.trans = self.ax.transData.transform
#         self.linehandle, = self.ax.plot([],[],**kwargs)
#         if "label" in kwargs: kwargs.pop("label")
#         self.line, = self.ax.plot(x, y, **kwargs)
#         self.line.set_color(self.linehandle.get_color())
#         self._resize()
#         self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

#     def _resize(self, event=None):
#         lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
#         if lw != self.lw:
#             self.line.set_linewidth(lw)
#             self.lw = lw
#             self._redraw_later()

#     def _redraw_later(self):
#         self.timer = self.fig.canvas.new_timer(interval=10)
#         self.timer.single_shot = True
#         self.timer.add_callback(lambda : self.fig.canvas.draw_idle())
#         self.timer.start()


def lamp_fit(instance_mask):
    lll_instances, lcw_instances, lct_instances = list_map_instances(instance_mask)
    for l in lll_instances:
        data = rotate_fit_bspline(l)
        # data = rotate_fit_bazier(l)
        # xvals, yvals = bezier_curve(data, nTimes=1000)
        # data = np.hstack(( xvals.reshape((-1,1)), yvals.reshape((-1,1)) ))
        plt.plot(data[:,0], data[:,1], 'r-', linewidth=2)
    for l in lct_instances:
        data = rotate_fit_bspline(l)
        # data = rotate_fit_bazier(l)
        # xvals, yvals = bezier_curve(data, nTimes=1000)
        # data = np.hstack(( xvals.reshape((-1,1)), yvals.reshape((-1,1)) ))
        plt.plot(data[:,0], data[:,1], 'b-', linewidth=2)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('lamp_lines.png', bbox_inches='tight')
    for l in lcw_instances:
        l = l.astype(np.float32)
        plt.scatter(l[:,0], l[:,1], c=(0,1,0))
    plt.savefig('lamp_fit.png', bbox_inches='tight')
    plt.show()
    img = cv2.imread('lamp_lines.png')
    img = np.rot90(img, 2)
    img = cv2.flip(img, 1)
    img = np.ones_like(img) * 255
    for l in lcw_instances:
        # l = l[:, ::-1]
        rect = cv2.minAreaRect(l)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = cv2.fillConvexPoly(img, box, color=(0,255,0))
    cv2.imwrite('lamp_cw.png', img)


def vout_lamp_fusion(vpos, lpos, coordspath, listspath, lamp_map):
    # vpos = np.loadtxt('/media/yuzeh/硬盘/CARLA_Dataset/23-2-5/vehicle_dataset/372/vehicle_location.txt')
    # lamp_poss = np.loadtxt('lamppose_incarla.txt')
    # lpos = lamp_poss[120]
    # coordspath = '/home/yuzeh/Downloads/venddata/coord372.npy'
    # listspath = '/home/yuzeh/Downloads/venddata/type372.npy'
    # lamp_map = np.load('/media/yuzeh/硬盘/CARLA_Dataset/23-2-5/lamp_dataset_dense/lampslidar_120/instance_mask.npy')
    Ml2v = get_Ml2v(vpos, lpos)
    lll_instances, lcw_instances, lct_instances = list_map_instances(lamp_map, Ml2v=Ml2v)
    vll_instances, vct_instances, vcw_instances = netouput_fit(coordspath, listspath)
    fusion_ll = []
    fusion_cw = []
    fusion_ct = []

    vll_crspds = []
    for v in vll_instances: 
        start = time.time()
        dis = [ chamfer_distance(v,l) for l in lll_instances]
        vll_crspds.append(np.argmin(dis))
        stop = time.time()
        print('------ correspond time ----------')
        print(stop-start)
    lllidxs = np.unique(vll_crspds)
    for idx in lllidxs:
        vidxs = [i for i,x in enumerate(vll_crspds) if x == idx]
        l = lll_instances[idx]
        # lll_instances.pop(idx)
        new_instance = np.zeros((1,2))
        for vid in vidxs:
            v = vll_instances[vid]
            new_instance = np.vstack((new_instance, v))
        new_instance = np.vstack((new_instance, l))
        data = rotate_fit_bspline(new_instance[1:])
        # data = rotate_fit_bazier(new_instance)
        # xvals, yvals = bezier_curve(data, nTimes=1000)
        # data = np.hstack(( xvals.reshape((-1,1)), yvals.reshape((-1,1)) ))
        fusion_ll.append(data)
    lll_instances = [lll_instances[i] for i in range(len(lll_instances)) if i not in lllidxs]
    # for v in vcw_instances: 
    #     dis = [ chamfer_distance(v,l) for l in lcw_instances]
    #     l = lcw_instances[np.argmin(dis)]
    #     lcw_instances.pop(np.argmin(dis))
    #     new_instance = np.vstack((v,l))
    #     fusion_cw.append(new_instance)

    vct_crspds = []
    for v in vct_instances: 
        start = time.time()
        dis = [ chamfer_distance(v,l) for l in lct_instances]
        vct_crspds.append(np.argmin(dis))
        stop = time.time()
        print('------ correspond time ----------')
        print(stop-start)
    lctidxs = np.unique(vct_crspds)
    for idx in lctidxs:
        vidxs = [i for i,x in enumerate(vct_crspds) if x == idx]
        l = lct_instances[idx]
        new_instance = np.zeros((1,2))
        for vid in vidxs:
            v = vct_instances[vid]
            new_instance = np.vstack((new_instance,v))
        new_instance = np.vstack((new_instance, l))
        data = rotate_fit_bspline(new_instance[1:])
        # data = rotate_fit_bazier(new_instance)
        # xvals, yvals = bezier_curve(data, nTimes=1000)
        # data = np.hstack(( xvals.reshape((-1,1)), yvals.reshape((-1,1)) ))
        fusion_ct.append(data)

    lct_instances = [lct_instances[i] for i in range(len(lct_instances)) if i not in lctidxs]
    # del(lct_instances[lctidxs])
    # lct_instances.pop([lctidxs])

    for l in fusion_ll:
        plt.plot(l[:,0], l[:,1], 'r-')
    # for l in fusion_cw:
    #     plt.plot(l[:,0], l[:,1], 'g-')
    for l in fusion_ct:
        plt.plot(l[:,0], l[:,1], 'b-')
    for l in lll_instances:
        # data = rotate_fit_bazier(l)
        # xvals, yvals = bezier_curve(data, nTimes=1000)
        # plt.plot(xvals, yvals, 'r-')
        data = rotate_fit_bspline(l)
        plt.plot(data[:,0], data[:,1], 'r-')
    for l in lcw_instances:
        plt.plot(l[:,0], l[:,1], 'g-')
    for l in lct_instances:
        # data = rotate_fit_bazier(l)
        # xvals, yvals = bezier_curve(data, nTimes=1000)
        # plt.plot(xvals, yvals, 'b-')
        data = rotate_fit_bspline(l)
        plt.plot(data[:,0], data[:,1], 'b-')
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('vehiclelamp_fusion.png', bbox_inches='tight')
    plt.show()

    return fusion_ll, fusion_cw, fusion_ct  



if __name__ == '__main__':
    lamp_poss = np.loadtxt('./data/lamppose_incarla.txt')
    folderlist = os.listdir('./data/')
    for itm in folderlist:
        itmpath = './data/' + itm
        if os.path.isdir(itmpath):
            vpos = np.loadtxt(itmpath + '/vehicle_location.txt')
            lampid = np.loadtxt(itmpath + '/correspond_lampid.txt')
            lpos = lamp_poss[int(lampid)]
            coordspath = itmpath + '/coord.npy'
            listspath = itmpath + '/type.npy'
            lamp_map = np.load(itmpath + '/lamp_map.npy')
            vout_lamp_fusion(vpos, lpos, coordspath, listspath, lamp_map)