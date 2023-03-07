from smoother.data.loading.nuscenes_loader import load_gt_local
from smoother.data.common.box3d import l2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox
import os
import torch
import torch.nn.functional as F
from collections import defaultdict 
import numpy as np
from .common.sequence_data import TrackingResults

class NuscTrackingResults(TrackingResults):

    def __init__(self, tracking_results_path, config, version="v1.0-trainval", split="val", data_path="/data/nuscenes", nusc=None):
        print("Initializing NuscenesData class...")
        super(NuscTrackingResults, self).__init__(tracking_results_path, config, version, split, data_path)
        assert os.path.exists(tracking_results_path), 'Error: The result file does not exist!'

        assert version in ['v1.0-trainval', 'v1.0-mini'] 
        if version == 'v1.0-trainval':
            self.train_scenes = list(splits.train)
            self.val_scenes = list(splits.val)
        else: #args.version == 'v1.0-mini':
            self.train_scenes = list(splits.mini_train)
            self.val_scenes = list(splits.mini_val)

        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True) if not nusc else nusc

        # yields a list of all scene-tokens for the current split
        print("Splitting data ...")
        self.scene2split = {self.nusc.scene[i]['token']: self._get_scene2split(self.nusc.scene[i]['name']) for i in range(len(self.nusc.scene))}
        self.split_scene_token = [scene_token for scene_token in self.scene2split if self.scene2split[scene_token] == self.split]

        print("Loading prediction and ground-truths ...")
        self.max_boxes = 100000
        self.pred_boxes, self.meta = self.load_tracking_predictions(self.tracking_results_path)
        self.gt_boxes = self.load_gt_detections()

    def load_tracking_predictions(self, tracking_results_path):
        return load_prediction(tracking_results_path, self.max_boxes, TrackingBox, verbose=True)
    
    def load_gt_detections(self):
        return load_gt_local(self.nusc, self.split, DetectionBox, verbose=True)
    
    def get_sequence_id_from_index(self, index):
        return self.split_scene_token[index]
    
    #def get_sequence_from_id(self, id):
    #    return self.nusc.get("scene", id)
    
    #def get_frame_from_id(self, id):
    #    return self.nusc.get("sample", id)
    
    def get_frames_in_sequence(self, scene_token):
       seq = self.nusc.get("scene", id)
       seq_frames = []
       frame_token = seq["first_sample_token"]
       n_frames = seq["nbr_samples"]
       for i in range(n_frames):
           seq_frames.append(frame_token)
           frame = self.nusc.get("sample", frame_token)
           frame_token = frame["next"]
       return seq_frames
    
    def get_pred_boxes_from_frame(self, frame_token):
        return self.pred_boxes[frame_token]
    
    def get_gt_boxes_from_frame(self, frame_token):
        return self.gt_boxes[frame_token]
    
    #def get_first_frame_in_sequence(self, seq):
    #    return seq["first_sample_token"]
    
    def get_number_of_sequences(self):
        return len(self.split_scene_token)
    
    def get_length_of_sequence(self, seq):
        return seq["nbr_samples"]


    def _get_scene2split(self, scene_name):
        if scene_name in self.train_scenes:
            return "train"
        elif scene_name in self.val_scenes:
            return "val"
        else:
            return "test"