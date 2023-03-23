from torch import nn
from torch.nn import functional as F

class BoxRefinementLoss():

    def __init__(self, loss_config):
        self.loss_type = loss_config["type"]
        self.loss_weights = loss_config["weight"]

        self.loss_fn = self._get_loss_fn(self.loss_type)

    def _get_loss_fn(self, loss_type):
        if loss_type.lower() == 'l1':
            loss_fn = self.l1_loss
        elif loss_type.lower() == 'iou':
            loss_fn = self.iou_loss
        elif loss_type.lower() == 'giou':
            loss_fn = self.giou_loss
        else:
            raise NotImplementedError(f"loss {self.loss_type} is not implemented. Please use 'l1', 'iou' or 'giou' instead.")
        return loss_fn

    def l1_loss(self, predictions,gts):
        centers = predictions[:,:3]
        gt_centers = gts[:,:3]
        center_loss = F.l1_loss(centers,gt_centers, reduction="mean")

        sizes = predictions[:,3:6]
        gt_sizes = gts[:,3:6]
        size_loss = F.l1_loss(sizes,gt_sizes, reduction="mean")

        rotation = predictions[:,6:10]
        gt_rotation = gts[:,6:10]
        rotation_loss = F.l1_loss(rotation,gt_rotation, reduction="mean")
        return center_loss, size_loss, rotation_loss

    def iou_loss(self, predictions, gts):
        raise NotImplementedError

    def giou_loss(self, predictions, gts):
        raise NotImplementedError

    def compute_score_loss(self, predictions, gts):
        scores = predictions[:,10]
        gt_score = gts[:,10]
        m = nn.Sigmoid()
        loss = nn.BCELoss()
        out = loss(m(scores),gt_score)
        return out

    def compute_loss(self, predictions, gts):
        centers = predictions[:,:3]
        gt_centers = gts[:,:3]
        center_loss = self.l1_loss(centers,gt_centers)

        sizes = predictions[:,3:6]
        gt_sizes = gts[:,3:6]
        size_loss = self.l1_loss(sizes,gt_sizes)

        rotation = predictions[:,6:10]
        gt_rotation = gts[:,6:10]
        rotation_loss = self.l1_loss(rotation,gt_rotation)

        return center_loss, size_loss, rotation_loss

    def loss(self, predictions, gt):
        center_loss, size_loss, rotation_loss = self.loss_fn(predictions, gt)
        score_loss = self.compute_score_loss(predictions,gt)
        center_loss *= self.loss_weights["center"]
        size_loss *= self.loss_weights["size"]
        rotation_loss *= self.loss_weights["rotation"]
        score_loss *= self.loss_weights["score"]
        l = center_loss + size_loss + rotation_loss + score_loss
        return center_loss + size_loss + rotation_loss + score_loss
    
    def evaluate_model(self, dets, refined_dets, gt_anns):
        
        ### CENTER MAE/MSE
        dets_centers = dets[:,:3]
        ref_centers = refined_dets[:,:3]
        gt_centers = gt_anns[:,:3]

        mae_dets_center = F.l1_loss(dets_centers, gt_centers, reduction='sum')
        mse_dets_center = F.mse_loss(dets_centers, gt_centers, reduction='sum')

        mae_refinement_center = F.l1_loss(ref_centers, gt_centers, reduction='sum')
        mse_refinement_center = F.mse_loss(ref_centers, gt_centers, reduction='sum')

        ### SIZE MAE/MSE
        dets_sizes = dets[:,3:6]
        ref_sizes = refined_dets[:,3:6]
        gt_sizes = gt_anns[:,3:6]

        mae_dets_size = F.l1_loss(dets_sizes, gt_sizes, reduction='sum')
        mse_dets_size = F.mse_loss(dets_sizes, gt_sizes, reduction='sum')

        mae_refinement_size = F.l1_loss(ref_sizes, gt_sizes, reduction='sum')
        mse_refinement_size = F.mse_loss(ref_sizes, gt_sizes, reduction='sum')

        ### ROTATION MAE/MSE
        dets_rotations = dets[:,6:10]
        ref_rotations = refined_dets[:,6:10]
        gt_rotations = gt_anns[:,6:10]

        mae_dets_rotation = F.l1_loss(dets_rotations, gt_rotations, reduction='sum')
        mse_dets_rotation = F.mse_loss(dets_rotations, gt_rotations, reduction='sum')

        mae_refinement_rotation = F.l1_loss(ref_rotations, gt_rotations, reduction='sum')
        mse_refinement_rotation = F.mse_loss(ref_rotations, gt_rotations, reduction='sum')

        return {
            "rmse_dets_center": mse_dets_center,
            "rmse_refinement_center": mse_refinement_center,
            "rmse_dets_size":mse_dets_size,
            "rmse_refinement_size":mse_refinement_size,
            "rmse_dets_rotation":mse_dets_rotation,
            "rmse_refinement_rotation":mse_refinement_rotation
        }
