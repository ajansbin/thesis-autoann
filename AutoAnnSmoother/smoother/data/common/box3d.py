import numpy as np

# class Box3d():

#     def __init__(self, center, size, rotation, score, class_names):
#         self.center = center
#         self.size = size
#         self.rotation = rotation
#         self.score = score
#         self.class_names = class_names

#         self.has_gt = False
#         self.gt_center = None
#         self.size = None
#         self.rotation = None
    
#     def update_gt(self, gt_box):
#         self.has_gt = True
#         self.gt_center = gt_box["translation"]
#         self.gt_size = gt_box["size"]
#         self.gt_rotation = gt_box["rotation"]

# class TemporalBox3D(Box3D):

#     def __init__(self, center, size, rotation, score, class_names, tracking_id, temporal_encoding):
#         super.__init__(self, center, size, rotation, score, class_names)
#         self.tracking_id = tracking_id
#         self.temporal_encoding = temporal_encoding

#     def get_point(self):
#         return self.center + self.size + self.rotation + self.temporal_encoding

# Associate gt-box to pred_boxes
def associate(gt_boxes, pred_boxes, threshold = 100):
    dists = l2(gt_boxes, pred_boxes)
    
    gt_assoc = {}
    for pred_idx, pred_box in enumerate(pred_boxes):
        # loop through all predictions
        this_dists = dists[:, pred_idx].copy()
        gt_idx = np.argmin(this_dists)
        valid_match = dists[gt_idx] <= threshold
        if valid_match:
            gt_assoc[pred_box.tracking_id] =  gt_boxes[gt_idx]
    return gt_assoc   

def l2(gts, preds):
    gt_centers = np.stack([gt["translation"] for gt in gts]).reshape((-1, 1, 3))  # M x 3
    pred_centers = np.stack([pred["translation"] for pred in preds]).reshape((1, -1, 3))
    dists = np.linalg.norm(gt_centers[:, :, :2] - pred_centers[:, :, :2], axis=2)
    return dists