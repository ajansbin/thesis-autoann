import tqdm
from collections import defaultdict
from smoother.data.common.dataclasses import TrackingBox, Tracklet
from smoother.data.common.transformations import ToTensor, CenterOffset, YawOffset, Normalize
import torch
import numpy as np
import os
import copy
from smoother.data.common.utils import convert_to_sine_cosine
from pyquaternion import Quaternion


class TrackingData():

    def __init__(self, tracking_results, transformations=[], remove_non_foi_tracks=True, remove_non_gt_tracks=True):
        self.tracking_results = tracking_results
        self.transformations = transformations
        self.remove_non_foi_tracks = remove_non_foi_tracks
        self.remove_non_gt_tracks = remove_non_gt_tracks

        self.score_dist_temp = self.tracking_results.score_dist_temp
        self.assoc_metric = self.tracking_results.assoc_metric
        self.assoc_thres = self.tracking_results.assoc_thres

        self.remove_bottom_center = self.tracking_results.remove_bottom_center

        self.use_pc = self.tracking_results.config["model"]["pc"]["use_pc"]
        self.use_track = self.tracking_results.config["model"]["track"]["use_track"]

        self.pc_offset = self.tracking_results.config["data"]["pc_offset"]

        self.data_samples = list(self._get_data_samples())
        self.max_track_length = 180

        self.center_offset_index = None
        self.yaw_offset_index = None
        self.normalize_index = None

        # set center_offset_transformation index for later look-up
        for i, transformation in enumerate(self.transformations):
            if self._get_full_class_path(transformation) == self._get_full_class_path(CenterOffset):
                self.center_offset_index = i
            if self._get_full_class_path(transformation) == self._get_full_class_path(YawOffset):
                self.yaw_offset_index = i
            if self._get_full_class_path(transformation) == self._get_full_class_path(Normalize):
                self.normalize_index = i

    def _get_full_class_path(self, obj):
        if isinstance(obj, type):  # If the object is a class, not an instance
            return f"{obj.__module__}.{obj.__name__}"
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}"

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
            return box.center + box.size + box.rotation + temporal_encoding
        
    def _create_gt_data(self, track):
        if track.has_gt:
            gt_box = track.gt_box
            center = list(gt_box['translation'])
            size = list(gt_box['size'])
            #rotation = list(convert_to_sine_cosine(gt_box['rotation']))
            rotation = gt_box['rotation']
            #target_confidence = [0]#[np.exp(-self.score_dist_temp*gt_box["distance"])]
            gt_data = center + size + rotation #+ target_confidence
        else:
            gt_data = [0] * 8

        return gt_data

    def _get_and_pad_track_points(self, track):
        track_points = self._get_point_clouds(track)
        if self.pc_offset:
            track_points = self._get_offset_pc(track, track_points)

        track_start_index = track.starting_frame_index
        track_end_index = track_start_index + len(track)
        if track_end_index == 181:
            track_end_index = 180

        # Initialize a tensor with the required dimensions
        full_track_points = torch.zeros((self.max_track_length, track_points.shape[1], track_points.shape[2] + 1), dtype=torch.float32)

        # Fill in the track_points
        full_track_points[track_start_index:track_end_index, :, :-1] = track_points[0:track_end_index-track_start_index]

        # Fill in the temporal encoding for all points
        temporal_encoding = torch.arange(180) - 89

        # Unsqueezing the temporal encoding to match the dimensions of full_track_points
        temporal_encoding = temporal_encoding.unsqueeze(1).unsqueeze(2)
        # Adding the temporal encoding as a new column in the last dimension of full_track_points
        full_track_points[:, :, -1:] = full_track_points[:, :, -1:] + temporal_encoding

        return full_track_points.float()

    
    def _get_point_clouds(self, track):
        if "pc_path" in self.tracking_results.config["data"]:
            root_pc_path = self.tracking_results.config["data"]["pc_path"]
        else:
            root_pc_path = '/staging/agp/masterthesis/2023autoann/storage/smoothing/autoannsmoothing/preprocessed/preprocessed_full_train'

        pc_name = f"point_clouds_{track.sequence_id}_{track.tracking_id}.npy"
        pc_path = os.path.join(root_pc_path, pc_name)
        pc = torch.from_numpy(np.load(pc_path)).float()
        
        track_start_index = track.starting_frame_index
        track_end_index = track_start_index + len(track)
        #expected_size = track_end_index - track_start_index
        if pc.shape[0] == 181:
            pc = pc[:180,:]
        return pc

    def _get_offset_pc(self, track: Tracklet, track_points: np.ndarray):
        foi_center = np.array(track.get_foi_box().center)
        offset_points = track_points - foi_center
        return offset_points

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

                    #convert Quaternion to polar angle representation
                    box['rotation'] = convert_to_sine_cosine(box['rotation'])

                    tracking_box = TrackingBox.from_dict(box)
                    tracking_id = tracking_box.tracking_id

                    if tracking_id not in track_ids:
                        #foi_index = tracking_box.frame_index if tracking_box.is_foi else None
                        track_ids[tracking_id] = Tracklet(sequence_token, tracking_id, frame_index, self.assoc_metric, self.assoc_thres)
                    track_ids[tracking_id].add_box(tracking_box)

            # remove tracking_id which do not include FoI
            if self.remove_non_foi_tracks:
                track_ids_filtered = defaultdict(None)
                for track_id, track in track_ids.items():
                    if track.foi_index is not None:
                        track_ids_filtered[track_id] = track
            else:
                track_ids_filtered = copy.deepcopy(track_ids)

            # Associate ground truth to each track
            gt_boxes = self.tracking_results.get_gt_boxes_from_frame(frame_token)
            for gt_box in gt_boxes:
                gt_box['rotation'] = convert_to_sine_cosine(gt_box['rotation'])

            track_ids_filtered_has_gt = defaultdict(None)
            for track_id, track in track_ids_filtered.items():
                if track.foi_index:
                    track.associate(gt_boxes)

                if not self.remove_non_gt_tracks: #or track.has_gt:
                    track_ids_filtered_has_gt[track_id] = track
                elif track.has_gt:
                    track_ids_filtered_has_gt[track_id] = track


            #seq_tracking_ids[sequence_token] = track_ids_filtered
            seq_tracking_ids[sequence_token] = copy.deepcopy(track_ids_filtered_has_gt)
        return seq_tracking_ids
    

class WindowTrackingData():

    def __init__(self, tracking_results, window_start, window_end, transformations=[], tracking_data=None, remove_non_foi_tracks=True, remove_non_gt_tracks=True):
        self.tracking_results = tracking_results
        self.window_start = window_start
        self.window_end = window_end
        self.remove_non_foi_tracks = remove_non_foi_tracks
        self.remove_non_gt_tracks = remove_non_gt_tracks
        
        self.use_pc = self.tracking_results.config["model"]["pc"]["use_pc"]
        self.use_track = self.tracking_results.config["model"]["track"]["use_track"]

        if not tracking_data:
            print("Loading data samples")

        self.tracking_data = TrackingData(tracking_results, transformations, self.remove_non_foi_tracks, self.remove_non_gt_tracks) if not tracking_data else tracking_data

        if not tracking_data:
            print(f"Finished loading {len(self.tracking_data)} data samples!")

    def __len__(self):
        return len(self.tracking_data)

    def __getitem__(self, index):
        track_data, point_data, gt_data = self.tracking_data[index]
        foi_index = self.tracking_data.get_foi_index(index)
        start_index = foi_index + self.window_start
        end_index = foi_index + self.window_end

        if start_index < 0:
            pad_start = -start_index
            start_index = 0
        else:
            pad_start = 0

        if end_index >= len(track_data):
            pad_end = end_index - len(track_data) + 1
            end_index = len(track_data) - 1
        else:
            pad_end = 0

        wind_track_data = track_data[start_index:end_index+1]
        wind_track_data = torch.cat((torch.zeros(pad_start, wind_track_data.shape[1]), wind_track_data, torch.zeros(pad_end, wind_track_data.shape[1])), dim=0)

        if self.use_pc:
            wind_point_data = point_data[start_index:end_index+1]
            wind_point_data = torch.cat((torch.zeros(pad_start, wind_point_data.shape[1], wind_point_data.shape[2]), wind_point_data, torch.zeros(pad_end, wind_point_data.shape[1], wind_point_data.shape[2])), dim=0)
        else:
            wind_point_data = point_data # empty tensor

        return wind_track_data, wind_point_data, gt_data
    
    def get(self, track_index):
        return self.tracking_data.get(track_index)
    

class SlidingWindowTrackingData():

    def __init__(self, tracking_results, window_size, transformations=[], remove_non_foi_tracks=True, remove_non_gt_tracks=True):
        self.tracking_results = tracking_results
        self.window_size = window_size
        self.transformations = transformations
        self.remove_non_foi_tracks = remove_non_foi_tracks
        self.remove_non_gt_tracks = remove_non_gt_tracks

        self.use_pc = self.tracking_results.config["model"]["pc"]["use_pc"]
        self.use_track = self.tracking_results.config["model"]["track"]["use_track"]

        print("Loading sequences...")
        self.tracking_data = TrackingData(tracking_results, self.transformations, self.remove_non_foi_tracks, self.remove_non_gt_tracks)

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

        if start_index < 0:
            pad_start = -start_index
            start_index = 0
        else:
            pad_start = 0

        if end_index >= len(track_data):
            pad_end = end_index - len(track_data) + 1
            end_index = len(track_data) - 1
        else:
            pad_end = 0

        wind_track_data = track_data[start_index:end_index+1]
        wind_track_data = torch.cat((torch.zeros(pad_start, wind_track_data.shape[1]), wind_track_data, torch.zeros(pad_end, wind_track_data.shape[1])), dim=0)

        if self.use_pc:
            wind_point_data = point_data[start_index:end_index+1]
            wind_point_data = torch.cat((torch.zeros(pad_start, wind_point_data.shape[1], wind_point_data.shape[2]), wind_point_data, torch.zeros(pad_end, wind_point_data.shape[1], wind_point_data.shape[2])), dim=0)
        else:
            wind_point_data = point_data # empty tensor

        return wind_track_data, wind_point_data, gt_data

    def get(self, sliding_track_index):
        track_index = sliding_track_index % len(self.tracking_data)
        return self.tracking_data.get(track_index)


        

