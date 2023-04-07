import numpy as np
import torch

from zod.data_classes.box import Box3D
from zod.constants import Lidar
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from smoother.data.loading.loader import convert_to_quaternion, convert_to_sine_cosine


def l2(gt_boxes, pred_box):
    gt_centers = [gt_box["translation"] for gt_box in gt_boxes]
    pred_center = pred_box.center
    diff = torch.tensor(gt_centers, dtype=torch.float32) - torch.tensor(pred_center, dtype=torch.float32)
    dists = torch.norm(diff, dim=1)
    return dists

def calculate_giou3d_matrix(gts, pred):
    dists = np.zeros(len(gts))
    for i, gt in enumerate(gts):
        gt_array = list(gt['translation']) + list(gt['size']) + list(gt['rotation'])
        pred_array = list(pred.center) + list(pred.size) + list(pred.rotation)
        giou = giou3d(gt_array, pred_array)
        #giou = giou3d(gt, pred)
        dists[i] = giou
    return dists

def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area

def giou3d(gt, pred):
    center = np.array(pred[0:3])
    size = np.array(pred[3:6])
    #rotation = np.array(pred[6:10])
    rotation = convert_to_quaternion(np.array(pred[6:8]))

    gt_center = np.array(gt[0:3])
    gt_size = np.array(gt[3:6])
    #gt_rotation = np.array(gt[6:10])
    gt_rotation = convert_to_quaternion(np.array(gt[6:8]))
    
    #gt_center = np.array(gts['translation'])
    #gt_size = np.array(gts['size'])
    #gt_rotation = gts['rotation']

    box = Box3D(
        center,
        size,
        #Quaternion(rotation),
        rotation,
        Lidar
    )

    gt_box = Box3D(
        gt_center,
        gt_size,
        #Quaternion(gt_rotation),
        gt_rotation,
        Lidar
    ) 

    boxa_corners = box.corners_bev
    boxb_corners = gt_box.corners_bev

    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)

    #ha, hb = box_a.h, box_b.h
    ha, hb = box.size[2], gt_box.size[2]
    za, zb = box.center[2], gt_box.center[2]

    overlap_height = max(0, min((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2)))
    union_height = max((za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2))

    # compute intersection and union
    I = reca.intersection(recb).area * overlap_height
    U = box.size[1] * box.size[0] * ha + gt_box.size[1] * gt_box.size[0] * hb - I

    # compute the convex area
    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)
    C = convex_area * union_height

    # compute giou
    giou = I / U - (C - U) / C
    return giou