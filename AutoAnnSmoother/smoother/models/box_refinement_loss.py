from torch import nn
from torch.nn import functional as F
import torch
import numpy as np

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

    def l1_loss(self, center_preds, size_preds, rotation_preds, gts):

        has_gt = gts[:, -2]

        gt_centers = gts[:,:3]
        center_loss = F.l1_loss(center_preds,gt_centers, reduction="none").sum(-1)

        gt_sizes = gts[:,3:6]
        size_loss = F.l1_loss(size_preds,gt_sizes, reduction="none").sum(-1)

        gt_rotations = gts[:,6].unsqueeze(-1)
        yaw_err = self.get_yaw_err(rotation_preds, gt_rotations).squeeze(-1)

        #only compute center, size and rotation loss on boxes with gt
        n_gt = has_gt.sum()
        center_loss = torch.mul(center_loss, has_gt).sum() / n_gt
        size_loss = torch.mul(size_loss, has_gt).sum() / n_gt
        rotation_loss = torch.mul(yaw_err, has_gt).sum() / n_gt

        return center_loss, size_loss, rotation_loss

    def iou_loss(self, center_pred, size_pred, rotation_pred, gts):
        raise NotImplementedError

    def giou_loss(self, center_pred, size_pred, rotation_pred, gts):
        raise NotImplementedError

    def compute_score_loss(self, score_pred, gts):
        gt_score = gts[:,-2]
        loss = nn.BCELoss()
        out = loss(score_pred.squeeze(-1),gt_score)
        return out
    
    def loss(self, center_preds, size_preds, rotation_preds, score_pred, gts):
        center_loss, size_loss, rotation_loss = self.loss_fn(center_preds, size_preds, rotation_preds, gts)
        score_loss = self.compute_score_loss(score_pred,gts)
        center_loss *= self.loss_weights["center"]
        size_loss *= self.loss_weights["size"]
        rotation_loss *= self.loss_weights["rotation"]
        score_loss *= self.loss_weights["score"]
        return center_loss + size_loss + rotation_loss + score_loss
        
    def evaluate_model(self, dets, refined_dets, gt_anns):

        #Only compute MSE for detection with ground-truth associations
        non_zero_gt_indices = torch.nonzero(gt_anns[:,-2], as_tuple=True)[0]
        n_non_zero = len(non_zero_gt_indices)

        ### CENTER MSE
        dets_centers = dets[non_zero_gt_indices,:3]
        ref_centers = refined_dets[non_zero_gt_indices,:3]
        gt_centers = gt_anns[non_zero_gt_indices,:3]

        sos_det_center = F.mse_loss(dets_centers, gt_centers, reduction='none')
        sos_ref_center = F.mse_loss(ref_centers, gt_centers, reduction='none')

        sos_det_x = sos_det_center[:,0]
        sos_det_y = sos_det_center[:,1]
        sos_det_z = sos_det_center[:,2]
        sos_ref_x = sos_ref_center[:,0]
        sos_ref_y = sos_ref_center[:,1]
        sos_ref_z = sos_ref_center[:,2]

        ### SIZE MSE
        dets_sizes = dets[non_zero_gt_indices,3:6]
        ref_sizes = refined_dets[non_zero_gt_indices,3:6]
        gt_sizes = gt_anns[non_zero_gt_indices,3:6]

        sos_det_size = F.mse_loss(dets_sizes, gt_sizes, reduction='none')
        sos_ref_size = F.mse_loss(ref_sizes, gt_sizes, reduction='none')

        ### ROTATION MSE
        dets_rotations = dets[non_zero_gt_indices,6]
        ref_rotations = refined_dets[non_zero_gt_indices,6]
        gt_rotations = gt_anns[non_zero_gt_indices,6]

        sos_det_rotation = self.get_yaw_err(dets_rotations, gt_rotations).pow(2)
        sos_ref_rotation = self.get_yaw_err(ref_rotations, gt_rotations).pow(2)

        ### SCORE 
        gt_score = gt_anns[:,-2]
        refined_dets = torch.where(refined_dets[:,-1]<0.5, 0.0, 1.0)
        acc_ref = torch.sum(refined_dets == gt_score).float()
        
        return {
            "rmse_dets_center": sos_det_center.sum(-1),
            "rmse_refinement_center": sos_ref_center.sum(-1),
            "rmse_dets_size":sos_det_size.sum(-1),
            "rmse_refinement_size":sos_ref_size.sum(-1),
            "rmse_dets_rotation":sos_det_rotation,
            "rmse_refinement_rotation":sos_ref_rotation,
            "rmse_det_x":sos_det_x,
            "rmse_det_y":sos_det_y,
            "rmse_det_z":sos_det_z,
            "rmse_ref_x":sos_ref_x,
            "rmse_ref_y":sos_ref_y,
            "rmse_ref_z":sos_ref_z
        }, {
            "acc_ref": acc_ref
        }, n_non_zero
    
    def get_yaw_err(self, pred_yaws, gt_yaws):
        pi = torch.tensor(np.pi)
        err = torch.abs(pred_yaws - gt_yaws)
        err = torch.where(err>pi,2*pi-err, err)
        return err 
    

    def get_yaw_err_spot(self, predictions, gts):
        yaws_sincos_unsized = predictions
        yaws_sincos = yaws_sincos_unsized / torch.norm(yaws_sincos_unsized, dim=1, keepdim=True)
        yaws = torch.atan2(yaws_sincos[:, 0], yaws_sincos[:, 1]).view(-1, 1)

        yaws_sincos_unsized = gts
        yaws_sincos = yaws_sincos_unsized / torch.norm(yaws_sincos_unsized, dim=1, keepdim=True)
        gt_yaws = torch.atan2(yaws_sincos[:, 0], yaws_sincos[:, 1]).view(-1, 1)
        yaw_err = torch.norm((yaws - gt_yaws + np.pi) % (2*np.pi) - np.pi, dim=1)
        
        return yaw_err