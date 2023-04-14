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

    def l1_loss(self, predictions,gts):
        centers = predictions[:,:3]
        gt_centers = gts[:,:3]
        center_loss = F.l1_loss(centers,gt_centers, reduction="mean")

        sizes = predictions[:,3:6]
        gt_sizes = gts[:,3:6]
        size_loss = F.l1_loss(sizes,gt_sizes, reduction="mean")

        rotation_sin = predictions[:,6]
        rotation_cos = predictions[:,7]
        gt_rotation_sin = gts[:,6]
        gt_rotation_cos = gts[:,7]

        rotation_loss = F.l1_loss(rotation_sin,gt_rotation_sin) + F.l1_loss(rotation_cos,gt_rotation_cos)

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
    
    def loss(self, predictions, gts):
        center_loss, size_loss, rotation_loss = self.loss_fn(predictions, gts)
        #score_loss = self.compute_score_loss(predictions,gt)
        center_loss *= self.loss_weights["center"]
        size_loss *= self.loss_weights["size"]
        rotation_loss *= self.loss_weights["rotation"]
        #score_loss *= self.loss_weights["score"]
        l = center_loss + size_loss + rotation_loss #+ score_loss
        return center_loss + size_loss + rotation_loss #+ score_loss
        
    def evaluate_model(self, dets, refined_dets, gt_anns):

        #Only compute MSE for detection with ground-truth associations
        non_zero_gt_indices = torch.nonzero(torch.sum(gt_anns, dim=1), as_tuple=True)[0]
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
        #dets_rotations = dets[non_zero_gt_indices,6:10]
        #ref_rotations = refined_dets[non_zero_gt_indices,6:10]
        #gt_rotations = gt_anns[non_zero_gt_indices,6:10]
        #sos_det_rotation = F.mse_loss(dets_rotations, gt_rotations, reduction='none')
        #sos_ref_rotation = F.mse_loss(ref_rotations, gt_rotations, reduction='none')

        dets_rotations = dets[non_zero_gt_indices,6:8]
        ref_rotations = refined_dets[non_zero_gt_indices,6:8]
        gt_rotations = gt_anns[non_zero_gt_indices,6:8]
        det_yaw_err = self.get_yaw_err(dets_rotations, gt_rotations)
        ref_yaw_err = self.get_yaw_err(ref_rotations, gt_rotations)
        sos_det_rotation = torch.pow(det_yaw_err, 2)
        sos_ref_rotation = torch.pow(ref_yaw_err, 2)

        return {
            "rmse_dets_center": sos_det_center.sum(-1),
            "rmse_refinement_center": sos_ref_center.sum(-1),
            "rmse_dets_size":sos_det_size.sum(-1),
            "rmse_refinement_size":sos_ref_size.sum(-1),
            "rmse_dets_rotation":sos_det_rotation.sum(-1),
            "rmse_refinement_rotation":sos_ref_rotation.sum(-1),
            "rmse_det_x":sos_det_x,
            "rmse_det_y":sos_det_y,
            "rmse_det_z":sos_det_z,
            "rmse_ref_x":sos_ref_x,
            "rmse_ref_y":sos_ref_y,
            "rmse_ref_z":sos_ref_z
        }, n_non_zero
    

    def get_yaw_err(self, predictions, gts):
        yaws_sincos_unsized = predictions
        yaws_sincos = yaws_sincos_unsized / torch.norm(yaws_sincos_unsized, dim=1, keepdim=True)
        yaws = torch.atan2(yaws_sincos[:, 0], yaws_sincos[:, 1]).view(-1, 1)

        yaws_sincos_unsized = gts
        yaws_sincos = yaws_sincos_unsized / torch.norm(yaws_sincos_unsized, dim=1, keepdim=True)
        gt_yaws = torch.atan2(yaws_sincos[:, 0], yaws_sincos[:, 1]).view(-1, 1)
        yaw_err = torch.norm((yaws - gt_yaws + np.pi) % (2*np.pi) - np.pi, dim=1)
        #yaw_err = torch.norm((yaws - gt_yaws) % (2*np.pi), dim=1)
        
        return yaw_err