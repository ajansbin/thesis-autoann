import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from pyquaternion import Quaternion
from zod.data_classes.box import Box3D
from zod.constants import Lidar, EGO
import torch


def convert_to_yaw(q_elem):
    assert type(q_elem) == list or type(q_elem) == np.ndarray
    assert len(q_elem) == 4
    q = Quaternion(q_elem)
    yaw = q.yaw_pitch_roll[0]

    return [yaw]


def convert_yaw_to_quat(yaw: list):
    return Quaternion(axis=[0, 0, 1], angle=yaw[0])


def convert_to_sine_cosine(q_elem):
    assert type(q_elem) == list or type(q_elem) == np.ndarray
    assert len(q_elem) == 4

    q = Quaternion(q_elem)
    yaw = q.yaw_pitch_roll[0]
    rot_sine = np.sin(yaw)
    rot_cosine = np.cos(yaw)
    return [rot_sine, rot_cosine]


def convert_to_quaternion(sine_cosine):
    assert type(sine_cosine) == list or type(sine_cosine) == np.ndarray
    assert len(sine_cosine) == 2
    q = Quaternion(axis=[0, 0, 1], radians=np.arctan2(sine_cosine[0], sine_cosine[1]))
    return q


def l2(gt_boxes, pred_box):
    gt_centers = [gt_box["translation"] for gt_box in gt_boxes]
    pred_center = pred_box.center
    diff = torch.tensor(gt_centers, dtype=torch.float32) - torch.tensor(
        pred_center, dtype=torch.float32
    )
    dists = torch.norm(diff, dim=1)
    return dists


def calculate_giou3d_matrix(gts, pred):
    gt_array = np.array([gt["translation"] + gt["size"] + gt["rotation"] for gt in gts])
    pred_array = np.array(list(pred.center) + list(pred.size) + list(pred.rotation))
    giou = giou3d(gt_array, pred_array)
    return giou


def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = (
        np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    )
    return area


def giou3d(gts, pred):
    centers, sizes, rotations = gts[:, 0:3], gts[:, 3:6], gts[:, 6:7]
    gt_boxes = [
        Box3D(c, s, convert_yaw_to_quat(r), Lidar)
        for c, s, r in zip(centers, sizes, rotations)
    ]

    pred_center, pred_size, pred_rotation = (
        pred[0:3],
        pred[3:6],
        convert_yaw_to_quat(pred[6:7]),
    )
    pred_box = Box3D(pred_center, pred_size, pred_rotation, Lidar)

    gt_corners_bev = np.array([box.corners_bev for box in gt_boxes])
    pred_corners_bev = pred_box.corners_bev

    reca, recb = Polygon(pred_corners_bev), [
        Polygon(box_corners) for box_corners in gt_corners_bev
    ]
    I_areas = np.array([reca.intersection(rec).area for rec in recb])

    ha, hb = pred_box.size[2], np.array([box.size[2] for box in gt_boxes])
    za, zb = pred_box.center[2], np.array([box.center[2] for box in gt_boxes])

    overlap_height = np.maximum(
        0, np.minimum(za + ha / 2 - (zb - hb / 2), zb + hb / 2 - (za - ha / 2))
    )
    union_height = np.maximum(za + ha / 2 - (zb - hb / 2), zb + hb / 2 - (za - ha / 2))

    I = I_areas * overlap_height
    U = (
        pred_size[1] * pred_size[0] * ha
        + np.array([box.size[1] * box.size[0] * box.size[2] for box in gt_boxes])
        - I
    )

    all_corners = np.array(
        [np.vstack((pred_corners_bev, box_corners)) for box_corners in gt_corners_bev]
    )
    convex_area = np.array(
        [PolyArea2D(corners[ConvexHull(corners).vertices]) for corners in all_corners]
    )
    C = convex_area * union_height

    giou = I / U - (C - U) / C
    return giou
