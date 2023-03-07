import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from .box3d import l2

# This class will be inherited for all different datatypes e.g. nuscenes and zod. 
class TrackingResults():

    def __init__(self, tracking_results_path, config, version="v1.0-trainval", split="val", data_path="/data/nuscenes"):
        self.tracking_results_path = tracking_results_path
        self.config = config
        self.version = version
        self.split = split
        self.data_path = data_path

        # CONFIG SPECIFIC
        self.feature_dim = self.config["data"]["feature_dim"]
        self.window_size = self.config["data"]["window_size"]
        self.gt_dim = self.config["data"]["gt_dim"]
        self.gt_assoc_threshold = self.config["data"]["gt_assoc_threshold"]

    def load_tracking_predictions(self, tracking_results_path):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def load_gt_detections(self):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def get_sequence_id_from_index(self, index):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    #def get_sequence_from_id(self, id):
    #    raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def get_frames_in_sequence(self, scene_token):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def get_pred_boxes_from_frame(self, frame_token):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def get_gt_boxes_from_frame(self, frame_token):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    #def get_first_frame_in_sequence(self, seq):
    #    raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def get_number_of_sequences(self):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def get_length_of_sequence(self, seq):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")



class SequenceData():

    def __init__(self, tracking_results: TrackingResults, window: tuple, foi_index=20):
        self.tracking_results = tracking_results
        #self.nusc = self.tracking_results.nusc
        self.start_ind, self.end_ind = window

        self.feature_dim = self.tracking_results.feature_dim
        self.window_size = self.end_ind - self.start_ind + 1
        self.foi_index = foi_index
        self.gt_dim = self.tracking_results.gt_dim
        self.gt_assoc_threshold = self.tracking_results.gt_assoc_threshold

    def __len__(self):
        return self.tracking_results.get_number_of_sequences()

    def __getitem__(self, index):
        seq_token = self.tracking_results.get_sequence_id_from_index(index)
        seq_tokens = self.tracking_results.get_frames_in_sequence(seq_token)
        window_tokens = seq_tokens[self.start_ind:self.end_ind]
        foi_ind = self._get_frame_of_interest()
        foi_token = seq_tokens[foi_ind]

        gt_assoc = self._get_corresponding_foi_gt_boxes(foi_token) #get all gt boxes for foi for all boxes

        track_id_pc = defaultdict(list)
        for i, frame_token in enumerate(window_tokens):
            frame_pred_boxes = self.tracking_results.get_pred_boxes_from_frame(frame_token)
            for box in frame_pred_boxes:
                if not box.tracking_id in track_id_pc:
                    t_encs = [i+self.start_ind-foi_ind for i in range(self.window_size)]
                    init_point = [[0]*(self.feature_dim-1) + [t_encs[i]] for i in range(self.window_size)]
                    track_id_pc[box.tracking_id] = init_point

                center = list(box.translation)
                size = list(box.size)
                rotation = list(box.rotation)
                temp_encoding = [i+self.start_ind-foi_ind]
                point = center + size + rotation + temp_encoding
                track_id_pc[box.tracking_id][i] = point

        x, y = [], []
        for track_id, point_cloud in track_id_pc.items():
            if track_id in gt_assoc: # track_id2 
                gt_box = self._get_gt_box(gt_assoc, track_id)
                x.append(point_cloud)
                y.append(gt_box)

        return x,y

    def _get_gt_box(self, gt_assoc, track_id):
        if gt_assoc[track_id] is None:
            point = [0]*self.gt_dim
        else:
            box = gt_assoc[track_id]
            center = list(box.translation)
            size = list(box.size)
            rotation = list(box.rotation)
            exist = [1]
            point = center + size + rotation + exist
            
        return point

    def _get_frames_in_sequence(self, scene_token):
        seq = self.tracking_results.get_sequence_from_id(scene_token)
        seq_frames = []
        frame_token = self.tracking_results.get_first_frame_in_sequence(seq)
        n_frames = self.tracking_results.get_length_of_sequence(seq)
        for i in range(n_frames):
            seq_frames.append(frame_token)
            frame = self.tracking_results.get_frame_from_id(frame_token)
            frame_token = self.tracking_results.get_next_frame(frame)
        return seq_frames
    
    def _get_frame_of_interest(self):
        return self.foi_index
    
    def _get_track_ids(self, pred_foi_boxes):
        return [box.tracking_id for box in pred_foi_boxes]
    
    def _get_corresponding_foi_gt_boxes(self, frame_token):
        frame_gt_boxes = self.tracking_results.get_gt_boxes_from_frame(frame_token)
        frame_pred_boxes = self.tracking_results.get_pred_boxes_from_frame(frame_token)
        gt_track_id_assocs = {}
        if len(frame_gt_boxes) == 0:
            for pred_idx, pred_box in enumerate(frame_pred_boxes):
                gt_track_id_assocs[pred_box.tracking_id] =  None
            
            return gt_track_id_assocs

        dists = l2(frame_gt_boxes, frame_pred_boxes)
        
        for pred_idx, pred_box in enumerate(frame_pred_boxes):
            # loop through all predictions
            this_dists = dists[:, pred_idx].copy()
            gt_idx = np.argmin(this_dists)
            valid_match = dists[gt_idx, pred_idx] <= self.gt_assoc_threshold
            if valid_match:
                gt_track_id_assocs[pred_box.tracking_id] =  frame_gt_boxes[gt_idx]
            else:
                gt_track_id_assocs[pred_box.tracking_id] =  None

        return gt_track_id_assocs
    
class WindowTracksData():

    def __init__(self, tracking_results: TrackingResults, window, means = None, stds = None):
        self.tracking_results = tracking_results
        self.window = window
        self.means = torch.tensor(means) if means else None
        self.stds = torch.tensor(stds) if stds else None

        self.data = list(self._get_data_samples())

    def __getitem__(self, index):
        track, track_gt = self.data[index]

        if self.means is not None and self.stds is not None:
           track = (track - self.means)/self.stds
           n_norm_features = len(track[0])
           track_gt[:n_norm_features] = (track_gt[:n_norm_features] - self.means) / self.stds

        return (track, track_gt)

    def __len__(self):
        return len(self.data)
    
    def _get_data_samples(self):
        seq_data = SequenceData(self.tracking_results, self.window)
        for x_seq, y_seq in seq_data:
            for track, track_gt in zip(x_seq,y_seq):
                track, track_gt = torch.tensor(track, dtype=torch.float32), torch.tensor(track_gt, dtype=torch.float32)
                yield (track, track_gt)

class SlidingWindowTracksData():

    def __init__(self, tracking_results: TrackingResults, window_size=5, foi_index=20, means=None, stds=None):
        self.tracking_results = tracking_results
        self.window_size = window_size
        self.foi_index = foi_index
        self.means = means
        self.stds = stds
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
            end_ind = start_ind + self.window_size - 1
            window = (start_ind, end_ind)
            print(window)
            wind_track_nusc = WindowTracksData(self.tracking_results,window,self.means, self.stds)
            for track in wind_track_nusc:
                yield track