import numpy as np
from dataclasses import dataclass
import torch
import copy
from typing import Any, Dict, Optional
from smoother.data.common.utils import (
    calculate_giou3d_matrix,
    l2,
    convert_yaw_to_quat,
    calculate_iou3d_matrix,
)
from scipy.spatial.transform import Rotation as R


@dataclass
class TrackingBox:
    tracking_id: str
    center: list
    size: list
    rotation: list
    is_foi: bool
    frame_index: int
    frame_token: str
    tracking_name: str
    tracking_score: float

    center_offset: Optional[list]
    yaw_offset: Optional[list]

    @classmethod
    def from_dict(cls, box_dict):
        return cls(
            tracking_id=box_dict["tracking_id"],
            center=box_dict["translation"],
            size=box_dict["size"],
            rotation=box_dict["rotation"],
            is_foi=box_dict["is_foi"],
            frame_index=box_dict["frame_index"],
            frame_token=box_dict["sample_token"],
            tracking_name=box_dict["tracking_name"],
            tracking_score=box_dict["tracking_score"],
            center_offset=None,
            yaw_offset=None,
        )

    def to_dict(self):
        return {
            "tracking_id": self.tracking_id,
            "center": self.center,
            "size": self.size,
            "rotation": self.rotation,
            "is_foi": self.is_foi,
            "frame_token": self.frame_token,
            "frame_index": self.frame_index,
            "tracking_name": self.tracking_name,
            "tracking_score": self.tracking_score,
            "center_offset": self.center_offset,
            "yaw_offset": self.yaw_offset,
        }

    def _get_rotation_matrix(self, yaw):
        rotation_matrix = R.from_euler("z", yaw).as_matrix()[0]
        return rotation_matrix

    def contains(self, points: np.ndarray) -> np.ndarray:
        rotation_matrix = self._get_rotation_matrix(self.rotation)
        rot_mat_transpose = rotation_matrix.T
        inverse_translate = -rot_mat_transpose @ self.center
        transform = np.hstack((rot_mat_transpose, inverse_translate.reshape(-1, 1)))
        transform = np.vstack((transform, np.array([0, 0, 0, 1])))

        points_homogeneous = np.vstack((points.T, np.ones((1, points.shape[0]))))
        local_points = transform @ points_homogeneous
        local_points = local_points.T[:, :3]  # Remove the homogeneous coordinate

        # Check if points are within the dimensions of the bounding box
        size = [self.size[0], self.size[1], self.size[2]]
        half_size = np.array(size) / 2
        mask = np.all(
            (local_points >= -half_size) & (local_points <= half_size), axis=1
        )

        return mask

    def get_points_in_bbox(self, points: np.ndarray) -> np.ndarray:
        mask = self.contains(points)
        return points[mask]

    def get_corners(self) -> np.ndarray:
        size = np.array(self.size)
        center = np.array(self.center)
        rotation = convert_yaw_to_quat(self.rotation)

        # Get the 3d corners of the box
        corners = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ]
        )
        corners *= size
        corners = rotation.rotation_matrix @ corners.T
        corners += center.reshape((-1, 1))
        return corners.T


class Tracklet:
    def __init__(
        self, sequence_id, tracking_id, starting_frame_index, assoc_metric, assoc_thres
    ):
        self.sequence_id = sequence_id
        self.tracking_id = tracking_id
        self.starting_frame_index = starting_frame_index
        self.assoc_metric = assoc_metric
        self.assoc_thres = assoc_thres

        self.boxes = []

        self.foi_index = None
        self.has_gt = False
        self.gt_box = None
        self.gt_dist = None

        self.center_offset = None
        self.yaw_offset = None

    def __repr__(self) -> str:
        return str(
            {
                "sequence_id": self.sequence_id,
                "tracking_id": self.tracking_id,
                "starting_frame_index": self.starting_frame_index,
                "track_length": self.__len__(),
                "foi_index": self.foi_index,
                "assoc_metric": self.assoc_metric,
                "assoc_thresh": self.assoc_thres,
                "has_gt": self.has_gt,
                "gt_dist": self.gt_dist,
            }
        )

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, index):
        return copy.deepcopy(self.boxes[index])

    def add_box(self, box: TrackingBox):
        self.boxes.append(box)

        if box.is_foi:
            self.foi_index = box.frame_index

    def get_foi_box(self) -> TrackingBox:
        assert self.foi_index is not None, "Error:box Track has no foi_index!"
        foi_box = self.boxes[self.foi_index - self.starting_frame_index]
        return copy.deepcopy(foi_box)

    def set_gt(self, gt_box, gt_dist):
        assert not self.has_gt
        self.has_gt = True
        self.gt_box = gt_box
        self.gt_dist = gt_dist

    def associate(self, gt_boxes):
        foi_box = self.get_foi_box()

        if len(gt_boxes) == 0:
            return

        if self.assoc_metric == "l2":
            dists = l2(gt_boxes, foi_box)
            gt_idx = torch.argmin(dists)
            gt_dist = torch.min(dists)
            valid_match = dists[gt_idx] <= self.assoc_thres
        elif self.assoc_metric == "giou":
            dists = torch.from_numpy(calculate_giou3d_matrix(gt_boxes, foi_box))
            gt_idx = torch.argmax(dists)
            gt_dist = torch.max(dists)
            valid_match = dists[gt_idx] >= self.assoc_thres
        elif self.assoc_metric == "iou":
            dists = torch.from_numpy(calculate_iou3d_matrix(gt_boxes, foi_box))
            gt_idx = torch.argmax(dists)
            gt_dist = torch.max(dists)
            valid_match = dists[gt_idx] > self.assoc_thres

        closest_gt = copy.deepcopy(gt_boxes[gt_idx])
        if valid_match and closest_gt["detection_name"] == foi_box.tracking_name:
            self.has_gt = True
            self.gt_box = closest_gt
            self.gt_dist = gt_dist

    def set_center_offset(self, offset):
        self.center_offset = offset

    def set_yaw_offset(self, offset):
        self.yaw_offset = offset
