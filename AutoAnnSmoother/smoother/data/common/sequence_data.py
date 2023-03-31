import tqdm
from collections import defaultdict
from smoother.data.common.dataclasses import TrackingBox, Tracklet
from smoother.data.common.transformations import Transformation, ToTensor
import torch
import numpy as np
from smoother.data.common.transformations import CenterOffset, Normalize


class SequenceData():

    def __init__(self, tracking_results, sequence_id, transformations=[]):
        self.tracking_results = tracking_results
        self.sequence_id = sequence_id
        self.transformations = transformations
        self.score_dist_temp = self.tracking_results.score_dist_temp

        self.assoc_metric = self.tracking_results.assoc_metric
        self.assoc_thres = self.tracking_results.assoc_thres

        self.remove_bottom_center = self.tracking_results.remove_bottom_center

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

        return track
    
    # returs the track object. Useful for retrieving information about the track.
    def get(self, track_index):
        return self.data_samples[track_index]
    
    def get_foi_index(self, track_index):
        track = self.data_samples[track_index]
        return track.foi_index

    def _get_data_samples(self):
        sequence_data = self._format_sequence_data()
        for track_id, track in sequence_data.items():
                yield track

    def _format_sequence_data(self):

        sequence_frames = self.tracking_results.get_frames_in_sequence(self.sequence_id)
        track_ids = {}
        for frame_index, frame_token in enumerate(sequence_frames):
            frame_pred_boxes = self.tracking_results.get_pred_boxes_from_frame(frame_token)
            if frame_pred_boxes == []:
                continue
            for box in frame_pred_boxes:
                box["is_foi"] = False #self.tracking_results.foi_indexes[self.sequence_id] == frame_index
                box["frame_index"] = frame_index

                if self.remove_bottom_center:
                    box["translation"][-1] = box["translation"][-1] + box["size"][-1]/2

                tracking_box = TrackingBox.from_dict(box)
                tracking_id = tracking_box.tracking_id

                if tracking_id not in track_ids:
                    #foi_index = tracking_box.frame_index if tracking_box.is_foi else None
                    track_ids[tracking_id] = Tracklet(self.sequence_id, tracking_id, frame_index, self.assoc_metric, self.assoc_thres)
                track_ids[tracking_id].add_box(tracking_box)

        return track_ids
    