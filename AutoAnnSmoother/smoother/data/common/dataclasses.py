import numpy as np
from dataclasses import dataclass
import torch
import copy
from typing import Any, Dict, Optional
from tools.utils.evaluation import giou3d,calculate_giou3d_matrix, l2


@dataclass
class TrackingBox():
    
    tracking_id: str
    center: list
    size: list
    rotation: list
    is_foi: bool
    frame_index: int
    tracking_name: str

    center_offset: Optional[list]

    @classmethod
    def from_dict(cls, box_dict):
        return cls(
            tracking_id = box_dict["tracking_id"],
            center = box_dict["translation"],
            size = box_dict["size"],
            rotation = box_dict["rotation"],
            is_foi = box_dict["is_foi"],
            frame_index = box_dict["frame_index"],
            tracking_name = box_dict["tracking_name"],
            center_offset = None
        )

        

class Tracklet():

    def __init__(self, sequence_id, tracking_id, starting_frame_index, assoc_metric, assoc_thres):
        self.sequence_id = sequence_id
        self.tracking_id = tracking_id
        self.starting_frame_index = starting_frame_index
        self.assoc_metric = assoc_metric
        self.assoc_thres = assoc_thres

        self.boxes = []
        self.n_samples = 1

        self.foi_index = None
        self.has_gt = False
        self.gt_box = None

        self.offset = None

    def __len__(self):
        return len(self.boxes)
    
    def __getitem__(self, index):
        return copy.deepcopy(self.boxes[index])

    def add_box(self, box:TrackingBox):
        self.boxes.append(box)

        if box.is_foi:
            self.foi_index = box.frame_index

    def get_foi_box(self):
        assert self.foi_index is not None, "Error: Track has no foi_index!"
        foi_box = self.boxes[self.foi_index-self.starting_frame_index]
        assert foi_box.is_foi
        return copy.deepcopy(foi_box)

    def associate(self, gt_boxes):
        foi_box = self.get_foi_box()

        if len(gt_boxes) == 0:
            return
        
        if self.assoc_metric == 'l2':
            dists = l2(gt_boxes, foi_box)
            gt_idx = np.argmin(dists)
            valid_match = dists[gt_idx] <= self.assoc_thres
        elif self.assoc_metric == 'giou':
            dists = calculate_giou3d_matrix(gt_boxes, foi_box)
            gt_idx = np.argmax(dists)
            valid_match = dists[gt_idx] >= self.assoc_thres

        closest_gt = copy.deepcopy(gt_boxes[gt_idx])

        if valid_match:
            self.has_gt = True
            self.gt_box = closest_gt
    
    def set_offset(self, offset):
        self.offset = offset