from torch.nn import functional as F
import torch
import numpy as np
import math


class BoxRefinementLoss:
    def __init__(self, loss_config, normalize_conf, device):
        self.loss_type = loss_config["type"]
        self.loss_weights = loss_config["weight"]
        self.device = device

        self.loss_fn = self._get_loss_fn(self.loss_type)
        self.std_center = normalize_conf["center"]["stdev"]
        self.std_size = normalize_conf["size"]["stdev"]
        self.std_rotation = normalize_conf["rotation"]["stdev"]

        self.temperature = 1

    def _get_loss_fn(self, loss_type):
        if loss_type.lower() == "l1":
            loss_fn = self.l1_loss
        elif loss_type.lower() == "iou":
            loss_fn = self.iou_loss
        elif loss_type.lower() == "giou":
            loss_fn = self.giou_loss
        else:
            raise NotImplementedError(
                f"loss {self.loss_type} is not implemented. Please use 'l1', 'iou' or 'giou' instead."
            )
        return loss_fn

    def l1_loss(self, center_preds, size_preds, rotation_preds, gts):
        has_gt = gts[:, -2]  # [x,y,z,l,w,h,r,hasgt,wheregt]

        gt_centers = gts[:, :3]
        center_loss = torch.sum(
            F.smooth_l1_loss(center_preds, gt_centers, reduction="none", beta=0), dim=-1
        )

        gt_sizes = gts[:, 3:6]
        size_loss = torch.sum(
            F.smooth_l1_loss(size_preds, gt_sizes, reduction="none", beta=0), dim=-1
        )

        gt_rotations = gts[:, 6:7]
        yaw_err = self.get_yaw_err(rotation_preds, gt_rotations).squeeze(-1)

        # only compute center, size and rotation loss on boxes with gt
        n_gt = max(1, has_gt.sum())
        center_loss = torch.mul(center_loss, has_gt).sum() / n_gt
        size_loss = torch.mul(size_loss, has_gt).sum() / n_gt
        rotation_loss = torch.mul(yaw_err, has_gt).sum() / n_gt
        return center_loss, size_loss, rotation_loss

    def iou_loss(self, center_pred, size_pred, rotation_pred, gts):
        raise NotImplementedError

    def giou_loss(self, center_pred, size_pred, rotation_pred, gts):
        raise NotImplementedError

    def compute_score_loss(
        self, center_preds, size_preds, rotation_preds, score_pred, gts
    ):
        n_preds = center_preds.shape[0]

        has_gt = gts[:, -2]  # [x,y,z,l,w,h,r,hasgt,wheregt]

        l2_err = F.mse_loss(center_preds, gts[:, :3], reduction="none").sum(-1)
        gt_scores = torch.exp(-self.temperature * l2_err)

        gt_scores = torch.mul(gt_scores, has_gt)

        out = F.l1_loss(score_pred.squeeze(-1), gt_scores, reduction="mean")

        return out

    def loss(self, center_preds, size_preds, rotation_preds, score_pred, gts):
        center_loss, size_loss, rotation_loss = self.loss_fn(
            center_preds, size_preds, rotation_preds, gts
        )
        score_loss = self.compute_score_loss(
            center_preds, size_preds, rotation_preds, score_pred, gts
        )
        center_loss *= self.loss_weights["center"]
        size_loss *= self.loss_weights["size"]
        rotation_loss *= self.loss_weights["rotation"]
        score_loss *= self.loss_weights["score"]

        return center_loss + size_loss + rotation_loss + score_loss

    def evaluate_model(self, dets, refined_dets, gt_anns, device):
        # Only compute MSE for detection with ground-truth associations
        non_zero_gt_indices = torch.nonzero(gt_anns[:, -2], as_tuple=True)[0]
        n_non_zero = max(1, len(non_zero_gt_indices))

        ### CENTER MSE
        dets_centers = dets[non_zero_gt_indices, :3]
        ref_centers = refined_dets[non_zero_gt_indices, :3]
        gt_centers = gt_anns[non_zero_gt_indices, :3]

        det_center_err = F.l1_loss(dets_centers, gt_centers, reduction="none")
        ref_center_err = F.l1_loss(ref_centers, gt_centers, reduction="none")

        det_x_err = det_center_err[:, 0]
        det_y_err = det_center_err[:, 1]
        det_z_err = det_center_err[:, 2]
        ref_x_err = ref_center_err[:, 0]
        ref_y_err = ref_center_err[:, 1]
        ref_z_err = ref_center_err[:, 2]

        ### SIZE MSE
        dets_sizes = dets[non_zero_gt_indices, 3:6]
        ref_sizes = refined_dets[non_zero_gt_indices, 3:6]
        gt_sizes = gt_anns[non_zero_gt_indices, 3:6]
        det_size_err = F.l1_loss(dets_sizes, gt_sizes, reduction="none")
        ref_size_err = F.l1_loss(ref_sizes, gt_sizes, reduction="none")

        ### ROTATION MSE
        dets_rotations = dets[non_zero_gt_indices, 6:7]
        ref_rotations = refined_dets[non_zero_gt_indices, 6:7]
        gt_rotations = gt_anns[non_zero_gt_indices, 6:7]

        det_rot_err = self.get_yaw_err(dets_rotations, gt_rotations).squeeze(-1)
        ref_rot_err = self.get_yaw_err(ref_rotations, gt_rotations).squeeze(-1)

        ### SCORE
        l2_err = F.mse_loss(refined_dets[:, :3], gt_anns[:, :3], reduction="none").sum(
            -1
        )
        gt_scores = torch.exp(-self.temperature * l2_err)

        score_err = F.l1_loss(refined_dets[:, -1], gt_scores, reduction="sum")

        return (
            {
                "mae_dets_center": det_center_err.sum(-1),
                "mae_refinement_center": ref_center_err.sum(-1),
                "mae_dets_size": det_size_err.sum(-1),
                "mae_refinement_size": ref_size_err.sum(-1),
                "mae_dets_rotation": det_rot_err,
                "mae_refinement_rotation": ref_rot_err,
                "mae_det_x": det_x_err,
                "mae_det_y": det_y_err,
                "mae_det_z": det_z_err,
                "mae_ref_x": ref_x_err,
                "mae_ref_y": ref_y_err,
                "mae_ref_z": ref_z_err,
            },
            {"mae_score": score_err},
            n_non_zero,
        )

    def get_yaw_err(self, pred_yaws, gt_yaws):
        """
        Assumes yaw in range [0, 2*pi].
        Transforms it to [-pi, pi]
        """

        yaw_err = pred_yaws - gt_yaws
        transformed_yaw_err = torch.norm(
            (yaw_err + math.pi) % (2 * math.pi) - math.pi, dim=1
        )

        return transformed_yaw_err

    def get_yaw_err_spot(self, predictions, gts):
        yaws_sincos_unsized = predictions
        yaws_sincos = yaws_sincos_unsized / torch.norm(
            yaws_sincos_unsized, dim=1, keepdim=True
        )
        yaws = torch.atan2(yaws_sincos[:, 0], yaws_sincos[:, 1]).view(-1, 1)

        yaws_sincos_unsized = gts
        yaws_sincos = yaws_sincos_unsized / torch.norm(
            yaws_sincos_unsized, dim=1, keepdim=True
        )
        gt_yaws = torch.atan2(yaws_sincos[:, 0], yaws_sincos[:, 1]).view(-1, 1)
        yaw_err = torch.norm((yaws - gt_yaws + np.pi) % (2 * np.pi) - np.pi, dim=1)

        return yaw_err
