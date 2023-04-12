
from zod.eval.detection._nuscenes_eval.common.data_classes import EvalBoxes
from zod.eval.detection._nuscenes_eval.detection.data_classes import DetectionBox
from smoother.data.zod_data import ZodTrackingResults
from smoother.io.config_utils import load_config
from tools.utils.evaluation import giou3d
import numpy as np
from zod.eval.detection.eval_nuscenes_style import evaluate_nuscenes_style

import json

class_names = [
    'Vehicle']#, 'VulnerableVehicle', 'Pedestrian']


def main(data_path:str, version:str, split:str, result_path:str, type:str, config:str, save_path:str, eval_type:str):
    conf = load_config(config)
    giou_thres = conf["data"]["association_thresholds"]['giou']

    tracking_results = ZodTrackingResults(result_path, conf, version, split, data_path)

    gtsEval, predEval = create_eval_boxes(tracking_results, type)

    if eval_type == 'nuscenes':
        evaluate_nuscenes_style(gtsEval, predEval, verbose=True, output_path=save_path)
    elif eval_type == 'giou':
        print(mean_giou(gtsEval, predEval, giou_thres))
    elif eval_type == 'fp':
        _, fp = calculate_gious_fp(gtsEval, predEval, giou_thres)
        print('Number false positives:', fp)
        print('Total number predicted boxes:', len(predEval.all))
    elif eval_type == 'all':
        evaluate_nuscenes_style(gtsEval, predEval, verbose=True, output_path=save_path)
        giou, fp = calculate_gious_fp(gtsEval, predEval, giou_thres)
        print('Mean GIoU', sum(giou)/len(giou))
        print('Number false positives:', fp)
        print('Total number predicted boxes:', len(predEval.all))

def create_eval_boxes(tracking_results, type):
    gts = tracking_results.gt_boxes
    gt_frames = tracking_results.gt_frames
    preds = tracking_results.pred_boxes

    predEval, gtsEval = EvalBoxes(), EvalBoxes()

    for seq_id in gt_frames:
        gt_frame = gt_frames[seq_id]
        track_boxes = []

        for box in tracking_results.get_pred_boxes_from_frame(gt_frame):
            gravity_center = box["translation"][-1] + box["size"][-1]/2
            translation = [box["translation"][0]] + [box["translation"][1]] + [gravity_center]
            score = box['detection_score'] if type == 'detection' else box['tracking_score']
            name = box['detection_name'] if type == 'detection' else box['tracking_name']
            if name not in class_names:
                continue
            pred_box = DetectionBox(
                sample_token=seq_id,
                translation=tuple(translation),
                size=tuple(box['size']),
                rotation=tuple(box['rotation']),
                ego_translation = tuple(translation),
                detection_name=name,
                detection_score=score,
            )
            track_boxes.append(pred_box)
        predEval.add_boxes(seq_id, track_boxes)
            
        gt_boxes = []
        for box in gts[seq_id]:
            if box['detection_name'] not in class_names:
                continue
            gt_box = DetectionBox(
                sample_token=seq_id,
                translation=tuple(box['translation']),
                size=tuple(box['size']),
                rotation=tuple(box['rotation']),
                ego_translation = tuple(box['translation']),
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

def calculate_gious_fp(gtsEval, predEval, giou_thres):
    gious = []
    false_pos = 0

    for seq_id in gtsEval.boxes.keys():
        preds = predEval.boxes[seq_id]
        gts = gtsEval.boxes[seq_id]
        if len(gts) == 0:
            print('No ground truth found for sequence', seq_id)
            false_pos += 1  
            continue
        if len(preds) == 0:
            print('No predictions found for sequence', seq_id)
            continue
        for track_box in preds:
            dists = calculate_giou3d_matrix(gts, track_box)
            gt_idx = np.argmax(dists)
            gt_dist = np.max(dists)
            valid_match = dists[gt_idx] >= giou_thres
            if valid_match:
                gious.append(gt_dist)
            if not valid_match:
                false_pos += 1 
    return gious, false_pos


def mean_giou(gts, preds, giou_thres):
    gious, _ = calculate_gious_fp(gts, preds, giou_thres)
    return sum(gious)/len(gious)

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
    parser.add_argument('--save-dir', type=str, default="/home/s0001668/workspace/storage/smoothing/")
    
    args = parser.parse_args()

    main(args.data_path, args.version, args.split, args.result_path, args.type, args.config, args.save_dir, args.eval_type)