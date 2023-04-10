import tqdm
from collections import defaultdict
from smoother.data.common.dataclasses import TrackingBox, Tracklet
from smoother.data.common.transformations import Transformation, ToTensor
import torch
import numpy as np
from smoother.data.common.transformations import CenterOffset, YawOffset, Normalize
import os
import copy
from smoother.data.common.utils import convert_to_sine_cosine, convert_to_quaternion
from pyquaternion import Quaternion


class TrackingData():

    def __init__(self, tracking_results, transformations=[]):
        self.tracking_results = tracking_results
        self.transformations = transformations
        self.score_dist_temp = self.tracking_results.score_dist_temp

        self.assoc_metric = self.tracking_results.assoc_metric
        self.assoc_thres = self.tracking_results.assoc_thres

        self.remove_bottom_center = self.tracking_results.remove_bottom_center

        self.use_pc = self.tracking_results.config["model"]["use_pc"]
        self.use_track = self.tracking_results.config["model"]["use_track"]

        self.data_samples = list(self._get_data_samples())
        self.max_track_length = 180

        self.center_offset_index = None
        self.yaw_offset_index = None
        self.normalize_index = None

        # set center_offset_transformation index for later look-up
        for i, transformations in enumerate(self.transformations):
            if type(transformations) == CenterOffset:
                self.center_offset_index = i
            if type(transformations) == YawOffset:
                self.yaw_offset_index = i
            if type(transformations) == Normalize:
                self.normalize_index = i

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        track = self.data_samples[index]

        gt_data = self._create_gt_data(track)
        track_data = self._create_track_data(track)
        track_data, gt_data = self._apply_transformations(track, track_data, gt_data)
        if self.use_pc:
            track_points = self._get_and_pad_track_points(track)
        else:
            track_points = torch.tensor([])
        return track_data, track_points, gt_data

    def _create_track_data(self, track):
        track_start_index = track.starting_frame_index
        track_end_index = track_start_index + len(track)

        track_data = [self._get_track_data_entry(i, track, track_start_index, track_end_index) for i in range(self.max_track_length)]

        return track_data

    def _get_track_data_entry(self, i, track, track_start_index, track_end_index):
        if i < track_start_index or i >= track_end_index:
            pad_data = [0] * 8 + [i - track.foi_index]
            return pad_data
        else:
            box = track[i - track_start_index]
            temporal_encoding = [box.frame_index - track.foi_index]
            rotation = convert_to_sine_cosine(Quaternion(box.rotation))
            return box.center + box.size + rotation + temporal_encoding

    def _get_and_pad_track_points(self, track):
        track_points = self._get_point_clouds(track)

        track_start_index = track.starting_frame_index
        track_end_index = track_start_index + len(track)

        padding_start = track_start_index
        padding_end = self.max_track_length - track_end_index

        if padding_start > 0:
            start_padding = torch.zeros((padding_start, *track_points.shape[1:]), dtype=track_points.dtype)
            track_points = torch.cat([start_padding, track_points], dim=0)

        if padding_end > 0:
            end_padding = torch.zeros((padding_end, *track_points.shape[1:]), dtype=track_points.dtype)
            track_points = torch.cat([track_points, end_padding], dim=0)

        return track_points

    def _create_gt_data(self, track):
        if track.has_gt:
            gt_box = track.gt_box
            center = list(gt_box['translation'])
            size = list(gt_box['size'])
            rotation = convert_to_sine_cosine(Quaternion(gt_box['rotation']))
            gt_data = center + size + rotation
        else:
            gt_data = [0] * 8

        return gt_data

    def _apply_transformations(self, track, track_data, gt_data):
        track_start_index = track.starting_frame_index
        track_end_index = track_start_index + len(track)

        for i, transformation in enumerate(self.transformations):
            if i == self.center_offset_index:
                foi_data = track_data[track.foi_index]
                transformation.set_offset(foi_data)
                track.set_center_offset(transformation.offset)
                transformation.set_start_and_end_index(track_start_index, track_end_index)
            if i == self.yaw_offset_index:
                foi_data = track_data[track.foi_index]
                transformation.set_offset(foi_data)
                track.set_yaw_offset(transformation.offset)
                transformation.set_start_and_end_index(track_start_index, track_end_index)
            elif i == self.normalize_index:
                transformation.set_start_and_end_index(track_start_index, track_end_index)

            track_data = transformation.transform(track_data)
            if track.has_gt or type(transformation) == ToTensor:
                gt_data = transformation.transform(gt_data)

        return track_data, gt_data

    def get(self, track_index):
        return self.data_samples[track_index]
    
    def get_foi_index(self, track_index):
        track = self.data_samples[track_index]
        return track.foi_index
    
    def get_pc(self, track_index):
        track = self.data_samples[track_index]
        track_points = self._get_point_clouds(track)
        return track_points

    def _get_point_clouds(self, track):
        #root_pc_path = '/staging/agp/masterthesis/2023autoann/storage/smoothing/autoannsmoothing/preprocessed_full_train'
        root_pc_path = self.tracking_results.config["data"]["pc_path"]

        pc_name = f"point_clouds_{track.sequence_id}_{track.tracking_id}.npy"
        pc_path = os.path.join(root_pc_path, pc_name)
        pc = torch.from_numpy(np.load(pc_path))
        return pc

    def _get_data_samples(self):
        tracking_data = self._format_tracking_data()
        for sequence, tracks in tracking_data.items():
            for track_id, track in tracks.items():
                yield track

    def _format_tracking_data(self):
        sequences = self.tracking_results.seq_tokens
        

        seq_tracking_ids = {}
        for sequence_token in tqdm.tqdm(sequences):
            sequence_frames = self.tracking_results.get_frames_in_sequence(sequence_token)

            track_ids = defaultdict(None)
            for frame_index, frame_token in enumerate(sequence_frames):
                frame_pred_boxes = self.tracking_results.get_pred_boxes_from_frame(frame_token)
                if frame_pred_boxes == []:
                    continue   
                for b in frame_pred_boxes:
                    box = copy.deepcopy(b)
                    box["is_foi"] = self.tracking_results.foi_indexes[sequence_token] == frame_index
                    box["frame_index"] = frame_index
                    box["frame_token"] = frame_token

                    if self.remove_bottom_center:
                        box["translation"][-1] = box["translation"][-1] + box["size"][-1]/2

                    tracking_box = TrackingBox.from_dict(box)
                    tracking_id = tracking_box.tracking_id

                    if tracking_id not in track_ids:
                        #foi_index = tracking_box.frame_index if tracking_box.is_foi else None
                        track_ids[tracking_id] = Tracklet(sequence_token, tracking_id, frame_index, self.assoc_metric, self.assoc_thres)
                    track_ids[tracking_id].add_box(tracking_box)

            # remove tracking_id which do not include FoI
            track_ids_filtered = defaultdict(None)
            for track_id, track in track_ids.items():
                if track.foi_index is not None:
                    track_ids_filtered[track_id] = track

            # Associate ground truth to each track
            gt_boxes = self.tracking_results.get_gt_boxes_from_frame(frame_token)
            track_ids_filtered_has_gt = defaultdict(None)
            for track_id, track in track_ids_filtered.items():
                track.associate(gt_boxes)
                if track.has_gt:
                    track_ids_filtered_has_gt[track_id] = track
            #seq_tracking_ids[sequence_token] = track_ids_filtered
            seq_tracking_ids[sequence_token] = track_ids_filtered_has_gt
        return seq_tracking_ids
    

class WindowTrackingData():

    def __init__(self, tracking_results, window_start, window_end, transformations=[], tracking_data=None):
        self.tracking_results = tracking_results
        self.window_start = window_start
        self.window_end = window_end
        
        self.use_pc = self.tracking_results.config["model"]["use_pc"]
        self.use_track = self.tracking_results.config["model"]["use_track"]

        if not tracking_data:
            print("Loading data samples")

        self.tracking_data = TrackingData(tracking_results, transformations) if not tracking_data else tracking_data

        if not tracking_data:
            print(f"Finished loading {len(self.tracking_data)} data samples!")

    def __len__(self):
        return len(self.tracking_data)

    def __getitem__(self, index):
        track_data, point_data, gt_data = self.tracking_data[index]
        foi_index = self.tracking_data.get_foi_index(index)
        start_index = foi_index + self.window_start
        end_index = foi_index + self.window_end

        wind_track_data = track_data[start_index:end_index+1]

        if self.use_pc:
            wind_point_data = point_data[start_index:end_index+1]
        else:
            wind_point_data = point_data # empty tensor

        return wind_track_data, wind_point_data, gt_data
    
    def get(self, track_index):
        return self.tracking_data.get(track_index)
    

class SlidingWindowTrackingData():

    def __init__(self, tracking_results, window_size, transformations=[]):
        self.tracking_results = tracking_results
        self.window_size = window_size
        self.transformations = transformations

        self.use_pc = self.tracking_results.config["model"]["use_pc"]
        self.use_track = self.tracking_results.config["model"]["use_track"]

        print("Loading sequences...")
        self.tracking_data = TrackingData(tracking_results, self.transformations)

        print(f"Finished loading {len(self.tracking_data) * self.window_size} data samples!")

    def __len__(self):
        return len(self.tracking_data) * self.window_size

    def __getitem__(self, index):
        track_index = index // self.window_size
        window_start_index = index % self.window_size - self.window_size + 1
        window_end_index = window_start_index + self.window_size - 1

        foi_index = self.tracking_data.get_foi_index(track_index)

        track_data, point_data, gt_data = self.tracking_data[track_index]

        start_index = foi_index + window_start_index
        end_index = foi_index + window_end_index

        wind_track_data = track_data[start_index:end_index+1]

        if self.use_pc:
            wind_point_data = point_data[start_index:end_index+1]
        else:
            wind_point_data = point_data # empty tensor

        return wind_track_data, wind_point_data, gt_data

    def get(self, sliding_track_index):
        track_index = sliding_track_index % len(self.tracking_data)
        return self.tracking_data.get(track_index)


        

