import numpy as np
from dataclasses import dataclass
import torch
import copy
from typing import Any, Dict, Optional

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

    def __init__(self, sequence_id, tracking_id, starting_frame_index):
        self.sequence_id = sequence_id
        self.tracking_id = tracking_id
        self.starting_frame_index = starting_frame_index

        self.boxes = []
        self.n_samples = 1

        self.foi_index = None
        self.has_gt = False
        self.gt_box = None

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
        
        dists = l2(gt_boxes, foi_box)
        
        gt_idx = np.argmin(dists)

        closest_gt = copy.deepcopy(gt_boxes[gt_idx])
        closest_gt["distance"] = dists[gt_idx]
        valid_match = dists[gt_idx] <= 10 
        if valid_match:
            self.has_gt = True
            self.gt_box = closest_gt

def l2(gt_boxes, pred_box):
    gt_centers = [gt_box["translation"] for gt_box in gt_boxes]
    pred_center = pred_box.center
    diff = torch.tensor(gt_centers, dtype=torch.float32) - torch.tensor(pred_center, dtype=torch.float32)
    dists = torch.norm(diff, dim=1)
    return dists