
from zod.eval.detection._nuscenes_eval.common.data_classes import EvalBoxes
from zod.eval.detection._nuscenes_eval.detection.data_classes import DetectionBox
from smoother.data.zod_data import ZodTrackingResults
from smoother.io.config_utils import load_config
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
import numpy as np
from zod.eval.detection.eval_nuscenes_style import evaluate_nuscenes_style
from zod.data_classes.box import Box3D
from zod.constants import Lidar, EGO
from pyquaternion import Quaternion

from collections import Counter

import json

class_names = [
    'Vehicle']#, 'VulnerableVehicle', 'Pedestrian']


def evaluate(data_path:str, version:str, split:str, result_path:str, type:str, config:str, giou_thres:float, save_path:str, eval_type:str):
    conf = load_config(config)

    tracking_results = ZodTrackingResults(result_path, conf, version, split, data_path)

    gtsEval, predEval = create_eval_boxes(tracking_results, type, conf)
    if eval_type == 'nuscenes':
        evaluate_nuscenes_style(gtsEval, predEval, verbose=True, output_path=save_path, verify_coordinate_system=False)
    elif eval_type == 'giou':
        print(mean_giou(gtsEval, predEval, giou_thres))
    elif eval_type == 'fp':
        _, fp, fn = calculate_gious_fp_fn(gtsEval, predEval, giou_thres)
        precision, recall = calculate_precision_recall(fp, fn, len(predEval.all))
        print('precision', precision)
        print('recall', recall)
        print('Number false positives:', fp)
        print('Number false negatives:', fn)
        print('Total number predicted boxes:', len(predEval.all))
    elif eval_type == 'all':
        evaluate_nuscenes_style(gtsEval, predEval, verbose=True, output_path=save_path, verify_coordinate_system=False)
        giou, fp, fn = calculate_gious_fp_fn(gtsEval, predEval, giou_thres)
        precision, recall = calculate_precision_recall(fp, fn, len(predEval.all))
        print('Mean GIoU', sum(giou)/len(giou))
        print('precision', precision)
        print('recall', recall)
        print('Number false positives:', fp)
        print('Number false negatives:', fn)
        print('Total number predicted boxes:', len(predEval.all))

def create_eval_boxes(tracking_results, type, conf):
    gts = tracking_results.gt_boxes
    gt_frames = tracking_results.gt_frames
    preds = tracking_results.pred_boxes

    predEval, gtsEval = EvalBoxes(), EvalBoxes()
    count = 0

    for seq_id in gt_frames:
        gt_frame = gt_frames[seq_id]
        seq = tracking_results.zod[seq_id]
        track_boxes = []
        for box in tracking_results.get_pred_boxes_from_frame(gt_frame):
            if conf['data']['remove_bottom_center']:
                gravity_center = box["translation"][-1] + box["size"][-1]/2
                translation = [box["translation"][0]] + [box["translation"][1]] + [gravity_center]
            else:
                translation = box["translation"]
            score = box['detection_score'] if type == 'detection' else box['tracking_score']
            name = box['detection_name'] if type == 'detection' else box['tracking_name']
            if name not in class_names:
                continue
            #must set negative size to 0
            box['size'] = np.absolute(box['size'])
            box3d = Box3D(translation,box['size'], box['rotation'], Lidar.VELODYNE)
            if box3d.center[1] > 61.2:
                continue
            box3d.convert_to(EGO, seq.calibration)

            pred_box = DetectionBox(
                sample_token=seq_id,
                translation=tuple(box3d.center),
                size=tuple(box3d.size),
                rotation=tuple(box3d.orientation.elements),
                detection_name=name,
                detection_score=score,
            )
            track_boxes.append(pred_box)
        predEval.add_boxes(seq_id, track_boxes)

            
        gt_boxes = []
        for box in gts[seq_id]:
            if box['detection_name'] not in class_names:
                continue
            if conf['data']['annotations']['world_coord']:
                print('Change config for annotations to not be in world')
            
            box3d = Box3D(box['translation'],box['size'], box['rotation'], Lidar.VELODYNE)
            if box3d.center[1] > 61.2:
                continue
            box3d.convert_to(EGO, seq.calibration)

            gt_box = DetectionBox(
                sample_token=seq_id,
                translation=tuple(box3d.center),
                size=tuple(box3d.size),
                rotation=tuple(box3d.orientation.elements),
                detection_name=box['detection_name'],
                detection_score=box['detection_score'],
            )
            gt_boxes.append(gt_box)
        gtsEval.add_boxes(seq_id, gt_boxes)
    return gtsEval, predEval

def calculate_giou3d_matrix(gts, pred):
    dists = np.zeros(len(gts))
    for i, gt in enumerate(gts):
        gt_array = list(gt.translation) + list(gt.size) + list(gt.rotation)
        pred_array = list(pred.translation) + list(pred.size) + list(pred.rotation)
        giou = giou3d(gt_array, pred_array)
        dists[i] = giou
    return dists

def giou3d(gt, pred):
    center = np.array(pred[0:3])
    size = np.array(pred[3:6])
    rotation = Quaternion(np.array(pred[6:10]))

    gt_center = np.array(gt[0:3])
    gt_size = np.array(gt[3:6])
    gt_rotation = Quaternion(np.array(gt[6:10]))

    box = Box3D(
        center,
        size,
        rotation,
        Lidar
    )

    gt_box = Box3D(
        gt_center,
        gt_size,
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

def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area

def calculate_gious_fp_fn(gtsEval, predEval, giou_thres):
    gious = []
    false_pos = 0
    gt_assoc = Counter()

    for seq_id in gtsEval.boxes.keys():
        preds = predEval.boxes[seq_id]
        gts = gtsEval.boxes[seq_id]
        if len(gts) == 0:
            #print('No ground truth found for sequence', seq_id)
            false_pos += 1  
            continue
        if len(preds) == 0:
            #print('No predictions in center frame found for sequence', seq_id)
            continue

        for gt_i in range(len(gts)):
            gt_id = str(seq_id)+'_'+str(gt_i)
            gt_assoc.update({gt_id:0})
            
        for i, track_box in enumerate(preds):
            dists = calculate_giou3d_matrix(gts, track_box)
            gt_idx = np.argmax(dists)
            gt_dist = np.max(dists)
            valid_match = dists[gt_idx] >= giou_thres
            if valid_match:
                gious.append(gt_dist)
                gt_id = str(seq_id)+'_'+str(gt_idx)
                gt_assoc.update([gt_id])
            if not valid_match:
                false_pos += 1
    false_neg = len([key for key, count in gt_assoc.items() if count == 0])
    return gious, false_pos, false_neg


def mean_giou(gts, preds, giou_thres):
    gious, _, _= calculate_gious_fp_fn(gts, preds, giou_thres)
    return sum(gious)/len(gious)

def calculate_precision_recall(fp, fn, tot_pred):
    tp = tot_pred - fp
    precision = tp/ (tp+fp)
    recall = tp/ (tp+fn)
    return precision, recall

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/datasets/zod/zodv2")
    parser.add_argument('--version', type=str, default="mini")
    parser.add_argument('--split', type=str, default="val")

    parser.add_argument('--result-path', type=str, default="/staging/agp/masterthesis/2023autoann/storage/detection/CenterPoint/predictions/cp-zod-mini-results-val/pts_bbox/results_zod.json", help="Path to results")
    parser.add_argument('--type', type=str, default='detection', choices=['detection', 'tracking', 'smoothing'])
    parser.add_argument('--eval-type', type=str, default='detection', choices=['nuscenes', 'giou', 'fp', 'all'])
    parser.add_argument('--config', type=str, default='/home/s0001668/workspace/thesis-autoann/AutoAnnSmoother/configs/training_config.yaml')
    parser.add_argument('--giou', type=float, default='0.0')
    parser.add_argument('--save-dir', type=str, default="/home/s0001668/workspace/storage/smoothing/")
    
    args = parser.parse_args()

    evaluate(args.data_path, args.version, args.split, args.result_path, args.type, args.config, args.giou, args.save_dir, args.eval_type)