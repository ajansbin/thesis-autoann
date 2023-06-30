from collections import defaultdict
from smoother.data.common.dataclasses import TrackingBox, Tracklet
from smoother.data.common.transformations import CenterOffset, Normalize, YawOffset
from smoother.data.common.utils import convert_to_sine_cosine
import copy


class SequenceData:
    def __init__(self, tracking_results, sequence_id, transformations=[]):
        self.tracking_results = tracking_results
        self.sequence_id = sequence_id
        self.transformations = transformations
        self.score_dist_temp = self.tracking_results.score_dist_temp

        self.use_pc = self.tracking_results.config["model"]["pc"]["use_pc"]
        self.data_conf = self.tracking_results.config["data"]
        self.pc_offset = self.tracking_results.config["data"]["pc_offset"]

        self.assoc_metric = self.tracking_results.assoc_metric
        self.assoc_thres = self.tracking_results.assoc_thres

        self.remove_bottom_center = self.tracking_results.remove_bottom_center

        self.data_samples = list(self._get_data_samples())
        self.max_track_length = 180

        self.center_offset_index = None
        self.yaw_offset_index = None
        self.normalize_index = None

        # set center_offset_transformation index for later look-up
        for i, transformation in enumerate(self.transformations):
            if self._get_full_class_path(transformation) == self._get_full_class_path(
                CenterOffset
            ):
                self.center_offset_index = i
            if self._get_full_class_path(transformation) == self._get_full_class_path(
                YawOffset
            ):
                self.yaw_offset_index = i
            if self._get_full_class_path(transformation) == self._get_full_class_path(
                Normalize
            ):
                self.normalize_index = i

    def _get_full_class_path(self, obj):
        if isinstance(obj, type):  # If the object is a class, not an instance
            return f"{obj.__module__}.{obj.__name__}"
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}"

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
        track_ids = defaultdict(None)
        for frame_index, frame_token in enumerate(sequence_frames):
            frame_pred_boxes = self.tracking_results.get_pred_boxes_from_frame(
                frame_token
            )
            if frame_pred_boxes == []:
                continue
            for b in frame_pred_boxes:
                box = copy.deepcopy(b)
                box["is_foi"] = False
                box["frame_index"] = frame_index
                box["frame_token"] = frame_token

                if self.remove_bottom_center:
                    box["translation"][-1] = (
                        box["translation"][-1] + box["size"][-1] / 2
                    )

                # convert Quaternion to polar angle representation
                box["rotation"] = convert_to_sine_cosine(box["rotation"])

                tracking_box = TrackingBox.from_dict(box)
                tracking_id = tracking_box.tracking_id

                if tracking_id not in track_ids:
                    track_ids[tracking_id] = Tracklet(
                        self.sequence_id,
                        tracking_id,
                        frame_index,
                        self.assoc_metric,
                        self.assoc_thres,
                    )
                track_ids[tracking_id].add_box(tracking_box)

        return track_ids
