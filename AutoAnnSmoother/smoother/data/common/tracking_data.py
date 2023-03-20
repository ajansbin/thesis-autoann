import tqdm
from collections import defaultdict
from smoother.data.common.dataclasses import TrackingBox, Tracklet
from smoother.data.common.transformations import Transformation
import torch
import numpy as np
from smoother.data.common.transformations import CenterOffset, Normalize


class TrackingData():

    def __init__(self, tracking_results, transformations=[]):
        self.tracking_results = tracking_results
        self.transformations = transformations
        self.score_dist_temp = self.tracking_results.score_dist_temp
        self.data_samples = list(self._get_data_samples())
        self.max_track_length = 180

        # set center_offset_transformation index for later look-up
        for i, transformations in enumerate(self.transformations):
            if type(transformations) == CenterOffset:
                self.center_offset_index = i
            if type(transformations) == Normalize:
                self.normalize_index = i


    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        track = self.data_samples[index]
        foi_data_index = track.foi_index-track.starting_frame_index
        if track.has_gt:
            gt_box = track.gt_box
            center = list(gt_box['translation'])
            size = list(gt_box['size'])
            rotation = list(gt_box['rotation'])
            target_confidence = [0]#[np.exp(-self.score_dist_temp*gt_box["distance"])]
            gt_data = center + size + rotation + target_confidence
        else:
            gt_data = [0]*11


        track_start_index = track.starting_frame_index
        track_end_index = track.starting_frame_index + len(track)-1
        track_data = []
        for i in range(self.max_track_length):
            if i < track_start_index or i > track_end_index:
                track_data.append([0]*11)
            else:
                box = track[i-track_start_index]
                temporal_encoding = [box.frame_index - track.foi_index]
                track_data.append(box.center + box.size + box.rotation + temporal_encoding)

        for i, transformation in enumerate(self.transformations):
            if i == self.center_offset_index:
                foi_data = track_data[foi_data_index]
                transformation.set_offset(foi_data)
                transformation.set_start_and_end_index(track_start_index, track_end_index)
            elif i == self.normalize_index:
                transformation.set_start_and_end_index(track_start_index, track_end_index)
            track_data = transformation.transform(track_data)
            gt_data = transformation.transform(gt_data)

        # update target confidence after transformations
        foi_center = track_data[foi_data_index,0:3]
        gt_center = gt_data[0:3]
        new_dist = torch.norm(gt_center-foi_center)
        gt_data[10] = np.exp(-self.score_dist_temp*new_dist)

        return track_data, gt_data
    
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
                for box in frame_pred_boxes:
                    box["is_foi"] = self.tracking_results.foi_indexes[sequence_token] == frame_index
                    box["frame_index"] = frame_index

                    tracking_box = TrackingBox.from_dict(box)
                    tracking_id = tracking_box.tracking_id

                    if tracking_id not in track_ids:
                        #foi_index = tracking_box.frame_index if tracking_box.is_foi else None
                        track_ids[tracking_id] = Tracklet(sequence_token, tracking_id, frame_index)
                    track_ids[tracking_id].add_box(tracking_box)

            # remove tracking_id which do not include FoI
            track_ids_filtered = defaultdict(None)
            for track_id, track in track_ids.items():
                if track.foi_index is not None:
                    track_ids_filtered[track_id] = track

            # Associate ground truth to each track
            gt_boxes = self.tracking_results.get_gt_boxes_from_frame(frame_token)
            for track_id, track in track_ids_filtered.items():
                track.associate(gt_boxes)

            seq_tracking_ids[sequence_token] = track_ids_filtered
        return seq_tracking_ids
    

class WindowTrackingData():

    def __init__(self, tracking_results, window_start, window_end, transformations=[], tracking_data=None):
        self.tracking_results = tracking_results
        self.window_start = window_start
        self.window_end = window_end
        self.tracking_data = TrackingData(tracking_results, transformations) if not tracking_data else tracking_data

    def __len__(self):
        return len(self.tracking_data)

    def __getitem__(self, index):
        track_data, gt_data = self.tracking_data[index]
        foi_index = self.tracking_data.get_foi_index(index)
        start_index = foi_index + self.window_start
        end_index = foi_index + self.window_end

        wind_track_data = track_data[start_index:end_index+1]

        return wind_track_data, gt_data
    

class SlidingWindowTrackingData():

    def __init__(self, tracking_results, window_size, transformations=[]):
        self.tracking_results = tracking_results
        self.window_size = window_size
        self.transformations = transformations

        print("Loading sequences...")
        self.tracking_data = TrackingData(tracking_results, self.transformations)

        print("Loading data samples")
        self.data = list(self._get_data_samples())
        print(f"Finished loading {len(self.data)} data samples!")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        track_data, gt_data = self.data[index]

        return track_data, gt_data

    def _get_data_samples(self):
        for i in range(self.window_size):
            start_ind = i - self.window_size + 1
            end_ind = start_ind + self.window_size - 1
            wind_track_nusc = WindowTrackingData(self.tracking_results,start_ind, end_ind, self.transformations, self.tracking_data)
            for track, track_gt in wind_track_nusc:
                yield track, track_gt

        

