from smoother.data.loading.loader import load_gt_local
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

class ResultsData():

    def __init__(self, tracking_results_path, config, version="v1.0-trainval", split="val", data_path="/data/nuscenes", nusc=None):
        print("Initializing NuscenesData class...")
        self.tracking_results_path = tracking_results_path
        self.config = config
        self.version = version
        self.split = split
        self.data_path = data_path

        # CONFIG SPECIFIC
        self.window_size = self.config["data"]["window_size"]
        self.max_tracks = self.config["data"]["max_tracks"]
        self.gt_assoc_threshold = self.config["data"]["gt_assoc_threshold"]

        
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True) if not nusc else nusc

        assert version in ['v1.0-trainval', 'v1.0-mini'] 
        if version == 'v1.0-trainval':
            self.train_scenes = list(splits.train)
            self.val_scenes = list(splits.val)
        else: #args.version == 'v1.0-mini':
            self.train_scenes = list(splits.mini_train)
            self.val_scenes = list(splits.mini_val)

        assert os.path.exists(tracking_results_path), 'Error: The result file does not exist!'

        # yields a list of all scene-tokens for the current split
        print("Splitting data ...")
        self.scene2split = {self.nusc.scene[i]['token']: self._get_scene2split(self.nusc.scene[i]['name']) for i in range(len(self.nusc.scene))}
        self.split_scene_token = [scene_token for scene_token in self.scene2split if self.scene2split[scene_token] == self.split]

        print("Loading prediction and ground-truths ...")
        max_boxes = 100000
        self.pred_boxes, self.meta = load_prediction(self.tracking_results_path, max_boxes, TrackingBox, verbose=True)
        self.gt_boxes = load_gt_local(self.nusc, split, DetectionBox, verbose=True)

    def _get_scene2split(self, scene_name):
        if scene_name in self.train_scenes:
            return "train"
        elif scene_name in self.val_scenes:
            return "val"
        else:
            return "test"
        
    def __getitem__(self, index):
        return self.split_scene_token[index]

    def __len__(self):
        return len(self.split_scene_token)

class SequenceNuscData():

    def __init__(self, result_data: ResultsData, window: tuple, foi_index=20):
        self.result_data = result_data
        self.nusc = self.result_data.nusc
        self.start_ind, self.end_ind = window
        self.window_size = self.end_ind - self.start_ind
        self.foi_index = foi_index
        self.max_tracks = self.result_data.max_tracks
        self.gt_assoc_threshold = self.result_data.gt_assoc_threshold

    def __len__(self):
        return len(self.result_data)

    def __getitem__(self, index):
        scene_token = self.result_data[index]
        scene_tokens = self._get_samples_in_scene(scene_token)
        window_tokens = scene_tokens[self.start_ind:self.end_ind]
        foi_ind = self._get_frame_of_interest()
        foi_token = scene_tokens[foi_ind]
        pred_foi_boxes = self.result_data.pred_boxes[foi_token]
        track_ids = self._get_track_ids(pred_foi_boxes)

        gt_assoc = self._get_corresponding_foi_gt_boxes(foi_token) #get all gt boxes for foi for all boxes

        track_id_pc = defaultdict(list)
        for i, sample_token in enumerate(window_tokens, self.start_ind):
            sample_boxes = self.result_data.pred_boxes[sample_token]
            for box in sample_boxes:
                center = list(box.translation)
                size = list(box.size)
                rotation = list(box.rotation)
                temp_encoding = [i-foi_ind]
                point = center + size + rotation + temp_encoding
                track_id_pc[box.tracking_id].append(point)

        x, y = [], []
        for k,v in track_id_pc.items():
            if k in gt_assoc:
                if not gt_assoc[k]:
                    gt_box = [0]*10
                else:
                    gt_box = self._get_gt_box(gt_assoc, k)

                #if k in gt_assoc:
                pad = [[0]*11] * (self.window_size-len(v))
                padded_pc = v + pad
                x.append(padded_pc)
                y.append(gt_box)

        x, y = torch.tensor(x), torch.tensor(y)

        #Pad or truncate
        if len(x) < self.max_tracks:
            x = F.pad(x,(0,0,0,0,0,self.max_tracks-len(x)))
        else:
            x = x[:50,:,:]

        if len(y) < self.max_tracks:
            y = F.pad(y,(0,0,0,self.max_tracks-len(y)))
        else:
            y = y[:50,:] 

        return x,y

    def _get_gt_box(self, gt_assoc, track_id):
        box = gt_assoc[track_id]
        center = list(box.translation)
        size = list(box.size)
        rotation = list(box.rotation)
        return center + size + rotation

    def _get_samples_in_scene(self, scene_token):
        scene = self.nusc.get("scene", scene_token)
        scene_samples = []
        sample_token = scene["first_sample_token"]
        for i in range(scene["nbr_samples"]):
            scene_samples.append(sample_token)
            sample = self.nusc.get("sample", sample_token)
            sample_token = sample["next"]
        return scene_samples
    
    def _get_frame_of_interest(self):
        return self.foi_index
    
    def get_track_ids(self, scene_index):
        scene_token = self.result_data[scene_index]
        scene_tokens = self._get_samples_in_scene(scene_token)
        foi_ind = self._get_frame_of_interest()
        foi_token = scene_tokens[foi_ind]
        pred_foi_boxes = self.result_data.pred_boxes[foi_token]
        return [box.tracking_id for box in pred_foi_boxes]

    
    def _get_track_ids(self, pred_foi_boxes):
        return [box.tracking_id for box in pred_foi_boxes]
    
    def _get_corresponding_foi_gt_boxes(self, sample_token):
        sample_gt_boxes = self.result_data.gt_boxes[sample_token]
        sample_pred_boxes = self.result_data.pred_boxes[sample_token]
        gt_assoc = {}
        if len(sample_gt_boxes) == 0:
            for pred_idx, pred_box in enumerate(sample_pred_boxes):
                gt_assoc[pred_box.tracking_id] =  None
            return gt_assoc

        dists = l2(sample_gt_boxes, sample_pred_boxes)
        
        for pred_idx, pred_box in enumerate(sample_pred_boxes):
            # loop through all predictions
            this_dists = dists[:, pred_idx].copy()
            gt_idx = np.argmin(this_dists)
            valid_match = dists[gt_idx, pred_idx] <= self.gt_assoc_threshold
            if valid_match:
                gt_assoc[pred_box.tracking_id] =  sample_gt_boxes[gt_idx]
            else:
                gt_assoc[pred_box.tracking_id] =  None

        return gt_assoc
    
class WindowTracksNusc():

    def __init__(self, result_data: ResultsData, window):
        self.result_data = result_data
        self.window = window

        self.data = list(self._get_data_samples())

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    def _get_data_samples(self):
        seq_nusc_data = SequenceNuscData(self.result_data, self.window)
        for x_seq, y_seq in seq_nusc_data:
            for track, track_gt in zip(x_seq,y_seq):
                is_tracks = torch.count_nonzero(track) != torch.tensor(0)
                is_gt_track = torch.count_nonzero(track_gt) != torch.tensor(0)
                if is_tracks and is_gt_track:
                    yield (track, track_gt)

class SlidingWindowTracksNusc():

    def __init__(self, result_data: ResultsData, window_size=5, foi_index=20):
        self.result_data = result_data
        self.window_size = window_size
        self.foi_index = foi_index
        self.data = list(self._get_data_samples())

    def __getitem__(self,index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def _get_data_samples(self):
        for i in range(self.window_size):
            if self.foi_index - self.window_size + i < 0:
                continue
            start_ind = self.foi_index-self.window_size + i + 1
            end_ind = start_ind + self.window_size
            window = (start_ind, end_ind)
            wind_track_nusc = WindowTracksNusc(self.result_data,window)
            for track in wind_track_nusc:
                yield track