import numpy as np
import torch
import torch.nn as nn
from scipy.special import comb
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .cluster import LaneNetPostProcessor
from .connect import connect_by_direction, sort_points_by_dist


def onehot_encoding(logits, dim=0):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def onehot_encoding_spread(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    one_hot.scatter_(dim, torch.clamp(max_idx-1, min=0), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx-2, min=0), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx+1, max=logits.shape[dim]-1), 1)
    one_hot.scatter_(dim, torch.clamp(max_idx+2, max=logits.shape[dim]-1), 1)

    return one_hot


def get_pred_top2_direction(direction, dim=1):
    direction = torch.softmax(direction, dim)
    idx1 = torch.argmax(direction, dim)
    idx1_onehot_spread = onehot_encoding_spread(direction, dim)
    idx1_onehot_spread = idx1_onehot_spread.bool()
    direction[idx1_onehot_spread] = 0
    idx2 = torch.argmax(direction, dim)
    direction = torch.stack([idx1, idx2], dim) - 1
    return direction


def get_bezier_parameters(X, Y, degree=3):
    """ Least square qbezier fit using penrose pseudoinverse.

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


# def vectorize(segmentation, embedding, direction, angle_class):
def vectorize(segmentation, embedding, angle_class):
    segmentation = segmentation.softmax(0)
    embedding = embedding.cpu()
    # direction = direction.permute(1, 2, 0).cpu()
    # direction = get_pred_top2_direction(direction, dim=-1)

    max_pool_1 = nn.MaxPool2d((1, 5), padding=(0, 2), stride=1)
    avg_pool_1 = nn.AvgPool2d((9, 5), padding=(4, 2), stride=1)
    max_pool_2 = nn.MaxPool2d((5, 1), padding=(2, 0), stride=1)
    avg_pool_2 = nn.AvgPool2d((5, 9), padding=(2, 4), stride=1)
    # post_processor = LaneNetPostProcessor(dbscan_eps=1.5, postprocess_min_samples=50)
    post_processor = LaneNetPostProcessor(dbscan_eps=5, postprocess_min_samples=4)

    oh_pred = onehot_encoding(segmentation).cpu().numpy()
    confidences = []
    line_types = []
    simplified_coords = []
    for i in range(1, oh_pred.shape[0]):
        single_mask = oh_pred[i].astype('uint8')
        single_embedding = embedding.permute(1, 2, 0)

        single_class_inst_mask, single_class_inst_coords = post_processor.postprocess(single_mask, single_embedding)
        # single_class_inst_mask, single_class_inst_coords = post_processor.postprocess(single_mask, single_embedding, min_area_threshold=10)
        if single_class_inst_mask is None:
            continue
        num_inst = len(single_class_inst_coords)
        for j in range(num_inst):
            coords = single_class_inst_coords[j]
            simplified_coords.append(coords)
            line_types.append(i)
                
    return simplified_coords, line_types       
    #     ######### yuzeh: add another DBSCAN after the instance embedding feature clustering ###########
    #     num_inst = len(single_class_inst_coords)
    #     for j in range(num_inst):
    #         coords = single_class_inst_coords[j]
    #         db = DBSCAN(eps=10, min_samples=10)
    #         # features = StandardScaler().fit_transform(coords)
    #         # db.fit(features)
    #         db.fit(coords)
    #         db_labels = db.labels_
    #         unique_labels = np.unique(db_labels)
    #         for index, label in enumerate(unique_labels.tolist()):
    #             if label == -1:
    #                 continue
    #             idx = np.where(db_labels == label)
    #             simplified_coords.append(coords[idx])
                
    # return simplified_coords

     
    # num_inst = len(simplified_coords)
    # mask = np.zeros(shape=[segmentation.shape[1], segmentation.shape[2]], dtype=np.int)
    # label = 1
    # Control_pts = []
    # for j in range(num_inst):
    #     coords = simplified_coords[j]
    #     pix_coord_idx = tuple((coords[:, 0], coords[:, 1])) # yuzeh
    #     mask[pix_coord_idx] = label
    #     label += 1
    #     ############# yuzeh: fit the Bezier Curve ###################
    #     data = get_bezier_parameters(coords[:, 0], coords[:, 1], degree=3)
    #     Control_pts.append(data)
    #     # check results 
    #     x_val = [x[0] for x in data]
    #     y_val = [x[1] for x in data]
    #     print(data)
    #     # Plot the control points
    #     plt.plot(x_val,y_val,'k--o', label='Control Points')
    #     # Plot the resulting Bezier curve
    #     xvals, yvals = bezier_curve(data, nTimes=1000)
    #     plt.plot(xvals, yvals, 'b-', label='B Curve')
    #     plt.legend()
    #     plt.show()
    
    #     #TODO: add line width to the bezier curve 
    

    

        
        ######### TODO: may delete the following NMS for the instance segmentation ############### 
        # num_inst = len(single_class_inst_coords)

        # prob = segmentation[i]
        # prob[single_class_inst_mask == 0] = 0
        # nms_mask_1 = ((max_pool_1(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
        # avg_mask_1 = avg_pool_1(prob.unsqueeze(0))[0].cpu().numpy()
        # nms_mask_2 = ((max_pool_2(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
        # avg_mask_2 = avg_pool_2(prob.unsqueeze(0))[0].cpu().numpy()
        # vertical_mask = avg_mask_1 > avg_mask_2
        # horizontal_mask = ~vertical_mask
        # nms_mask = (vertical_mask & nms_mask_1) | (horizontal_mask & nms_mask_2)

        # for j in range(1, num_inst + 1):
        #     full_idx = np.where((single_class_inst_mask == j))
        #     full_lane_coord = np.vstack((full_idx[1], full_idx[0])).transpose()
        #     confidence = prob[single_class_inst_mask == j].mean().item()

        #     idx = np.where(nms_mask & (single_class_inst_mask == j))
        #     if len(idx[0]) == 0:
        #         continue
        #     lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        #     range_0 = np.max(full_lane_coord[:, 0]) - np.min(full_lane_coord[:, 0])
        #     range_1 = np.max(full_lane_coord[:, 1]) - np.min(full_lane_coord[:, 1])
        #     if range_0 > range_1:
        #         lane_coordinate = sorted(lane_coordinate, key=lambda x: x[0])
        #     else:
        #         lane_coordinate = sorted(lane_coordinate, key=lambda x: x[1])

        #     lane_coordinate = np.stack(lane_coordinate)
        #     lane_coordinate = sort_points_by_dist(lane_coordinate)
        #     lane_coordinate = lane_coordinate.astype('int32')
        #     # lane_coordinate = connect_by_direction(lane_coordinate, direction, step=7, per_deg=360 / angle_class)

        #     simplified_coords.append(lane_coordinate)
        #     confidences.append(confidence)
        #     line_types.append(i-1)

    # return simplified_coords, confidences, line_types
