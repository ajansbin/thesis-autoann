import tqdm
from collections import defaultdict
from smoother.data.common.dataclasses import TrackingBox, Tracklet
from smoother.data.common.transformations import (
    ToTensor,
    CenterOffset,
    YawOffset,
    Normalize,
    PointsShift,
)
import torch
import numpy as np
import os
import copy
from smoother.data.common.utils import convert_to_yaw
from pyquaternion import Quaternion
import random

TRACK_FEAT_DIM = 8
POINT_FEAT_DIM = 4
GT_FEAT_DIM = 9


class TrackingData:
    def __init__(
        self,
        tracking_results,
        transformations=[],
        points_transformations=[],
        remove_non_foi_tracks=True,
        remove_non_gt_tracks=True,
        seq_tokens=None,
        use_pc=True,
    ):
        self.tracking_results = tracking_results
        self.use_pc = use_pc
        self.transformations = transformations
        self.points_transformations = points_transformations
        self.remove_non_foi_tracks = remove_non_foi_tracks
        self.remove_non_gt_tracks = remove_non_gt_tracks
        self.seqs = self.tracking_results.seq_tokens if not seq_tokens else seq_tokens

        data_conf = self.tracking_results.config["data"]
        self.assoc_metric = data_conf["association_metric"]
        self.assoc_thres = data_conf["association_thresholds"][self.assoc_metric]
        self.remove_bottom_center = data_conf["remove_bottom_center"]

        self.data_samples = list(self._get_data_samples())
        self.max_track_length = 180

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        track = copy.deepcopy(self.data_samples[index])

        track_data = self._get_track_data(track)
        track_points = self._get_track_points(track)
        gt_data = self._get_gt_data(track)

        """
        track_data      (180,7)         [x,y,z,l,w,h,r]
        track_points    (180, 1000, 3)  [x,y,z]
        gt_data         (8)             [x,y,z,l,w,h,r,s]
        """
        return track_data, track_points, gt_data

    def get(self, track_index) -> Tracklet:
        return self.data_samples[track_index]

    def get_foi_index(self, track_index) -> int:
        track = self.data_samples[track_index]
        return track.foi_index

    def _get_track_data(self, track):
        track_start_index = track.starting_frame_index
        track_end_index = track_start_index + len(track)

        track_data = [
            self._get_track_data_entry(i, track, track_start_index, track_end_index)
            for i in range(self.max_track_length)
        ]

        return torch.tensor(track_data).float()

    def _get_track_data_entry(self, i, track, track_start_index, track_end_index):
        if i < track_start_index or i >= track_end_index:
            pad_data = [0] * 7
            return pad_data
        else:
            box = track[i - track_start_index]
            return box.center + box.size + box.rotation

    def _get_gt_data(self, track):
        if track.has_gt:
            gt_box = track.gt_box
            center = list(gt_box["translation"])
            size = list(gt_box["size"])
            rotation = gt_box["rotation"]
            has_gt = [1]  # has gt indicator
            gt_data = center + size + rotation + has_gt
        else:
            gt_data = [0] * 8

        return torch.tensor(gt_data).float()

    def _get_track_points(self, track):
        if not self.use_pc:
            return torch.tensor([])

        track_points = self._load_track_points(track)
        track_points = self._pad_track_points(track_points, track)
        track_points = self._transform_track_points(track_points, track)
        return track_points

    def _load_track_points(self, track):
        if "pc_path" in self.tracking_results.config["data"]:
            root_pc_path = self.tracking_results.config["data"]["pc_path"]
        else:
            root_pc_path = "/staging/agp/masterthesis/2023autoann/storage/smoothing/autoannsmoothing/preprocessed_world/full_train"

        pc_name = f"point_clouds_{track.sequence_id}_{track.tracking_id}.npy"
        pc_path = os.path.join(root_pc_path, pc_name)
        pc = torch.from_numpy(np.load(pc_path)).float()

        if pc.shape[0] > self.max_track_length:
            pc = pc[: self.max_track_length, :]
        return pc

    def _pad_track_points(self, track_points, track):
        track_start_index = track.starting_frame_index
        track_end_index = track_start_index + len(track)
        if track_end_index > self.max_track_length:
            track_end_index = self.max_track_length

        # Initialize a tensor with the required dimensions
        full_track_points = torch.zeros(
            (self.max_track_length, track_points.shape[1], track_points.shape[2]),
            dtype=torch.float32,
        )

        # Fill in the track_points
        try:
            full_track_points[track_start_index:track_end_index] = track_points[
                0 : track_end_index - track_start_index
            ]
        except Exception as e:
            print("full track points", full_track_points.shape)
            print("start and end", track_start_index, track_end_index)
            print("track_points", track_points.shape)
            print(f"Exception occurred: {e}")
            raise

        return full_track_points.float()

    def _transform_track_points(self, track_points, track):
        if len(self.points_transformations) == 0:
            return track_points

        track_start_index = track.starting_frame_index
        track_end_index = track_start_index + len(track)
        for p_transformation in self.points_transformations:
            if self._get_full_class_path(p_transformation) == self._get_full_class_path(
                PointsShift
            ):
                p_transformation.set_start_and_end_index(
                    track_start_index, track_end_index
                )
            track_points = p_transformation.transform(track_points)
        return track_points

    def _get_data_samples(self):
        tracking_data = self._format_tracking_data()
        for sequence, tracks in tracking_data.items():
            for track_id, track in tracks.items():
                yield track

    def _format_tracking_data(self):
        seq_tracking_ids = {}
        for sequence_token in tqdm.tqdm(self.seqs, position=0, leave=True):
            sequence_frames = self.tracking_results.get_frames_in_sequence(
                sequence_token
            )

            track_ids = defaultdict(None)
            for frame_index, frame_token in enumerate(sequence_frames):
                frame_pred_boxes = self.tracking_results.get_pred_boxes_from_frame(
                    frame_token
                )
                if frame_pred_boxes == []:
                    continue
                for b in frame_pred_boxes:
                    box = copy.deepcopy(b)
                    box["is_foi"] = (
                        self.tracking_results.foi_indexes[sequence_token] == frame_index
                    )
                    box["frame_index"] = frame_index
                    box["frame_token"] = frame_token

                    if self.remove_bottom_center:
                        box["translation"][-1] = (
                            box["translation"][-1] + box["size"][-1] / 2
                        )

                    # convert Quaternion to yaw
                    box["rotation"] = convert_to_yaw(box["rotation"])

                    tracking_box = TrackingBox.from_dict(box)
                    tracking_id = tracking_box.tracking_id

                    if tracking_id not in track_ids:
                        # foi_index = tracking_box.frame_index if tracking_box.is_foi else None
                        track_ids[tracking_id] = Tracklet(
                            sequence_token,
                            tracking_id,
                            frame_index,
                            self.assoc_metric,
                            self.assoc_thres,
                        )
                    track_ids[tracking_id].add_box(tracking_box)
                    # if tracking_id == "Vehicle_0_13":
                    #    print(
                    #        tracking_id,
                    #        [box.center for box in track_ids[tracking_id].boxes],
                    #    )

            # remove tracking_id which do not include FoI
            if self.remove_non_foi_tracks:
                track_ids_filtered = defaultdict(None)
                for track_id, track in track_ids.items():
                    if track.foi_index is not None:
                        track_ids_filtered[track_id] = track
            else:
                track_ids_filtered = copy.deepcopy(track_ids)

            # Associate ground truth to each track
            gt_boxes = copy.deepcopy(
                self.tracking_results.get_gt_boxes_from_frame(frame_token)
            )
            for gt_box in gt_boxes:
                gt_box["rotation"] = convert_to_yaw(gt_box["rotation"])

            track_ids_filtered_has_gt = defaultdict(None)
            for track_id, track in track_ids_filtered.items():
                if track.foi_index:
                    track.associate(gt_boxes)

                if not self.remove_non_gt_tracks or track.has_gt:
                    track_ids_filtered_has_gt[track_id] = track

            seq_tracking_ids[sequence_token] = copy.deepcopy(track_ids_filtered_has_gt)
        return seq_tracking_ids


class WindowTrackingData:
    def __init__(
        self,
        tracking_results,
        window_size,
        times,
        random_slides,
        use_pc=True,
        transformations=[],
        points_transformations=[],
        remove_non_foi_tracks=True,
        remove_non_gt_tracks=True,
        seqs=None,
    ):
        self.tracking_results = tracking_results
        self.window_size = window_size
        self.times = times
        self.random_slides = random_slides
        self.use_pc = use_pc
        self.transformations = transformations
        self.points_transformations = points_transformations
        self.remove_non_foi_tracks = remove_non_foi_tracks
        self.remove_non_gt_tracks = remove_non_gt_tracks
        self.seqs = None

        print("Loading sequences...")
        self.tracking_data = TrackingData(
            tracking_results,
            self.transformations,
            self.points_transformations,
            self.remove_non_foi_tracks,
            self.remove_non_gt_tracks,
            seqs,
        )

        print(f"Finished loading {len(self.tracking_data) * self.times} data samples!")

    def __len__(self):
        return len(self.tracking_data) * self.times

    def __getitem__(self, index):
        track_index = index // self.times
        track_data, point_data, gt_data = self.tracking_data[track_index]
        """
        track_data      (180,7)         [x,y,z,l,w,h,r]
        track_points    (180, 1000, 3)  [x,y,z]
        gt_data         (8)             [x,y,z,l,w,h,r,s]
        """

        start_index, end_index, rel_foi_index = self._get_absolute_window_range(
            index, track_index, self.random_slides
        )

        start_index, end_index, pad_start, pad_end = self._get_pad_range(
            start_index, end_index, len(track_data)
        )

        wind_track_data = self._get_window_track_data(
            track_data, start_index, end_index, pad_start, pad_end
        )

        wind_point_data = self._get_window_point_data(
            point_data, start_index, end_index, pad_start, pad_end
        )

        # Add foi_index marker last to gt_data
        gt_data = torch.cat((gt_data, rel_foi_index))

        offset_track_data, offset_point_data, offset_gt_data = self._get_offset_data(
            wind_track_data, wind_point_data, gt_data
        )

        """
        offset_track_data   (W,8)           [x,y,z,l,w,h,r,t]
        offset_point_data   (180, 1000, 4)  [x,y,z,t]
        offset_gt_data      (9)             [x,y,z,l,w,h,r,s,t]
        """
        return offset_track_data, offset_point_data, offset_gt_data

    def get(self, index) -> Tracklet:
        track_index = index // self.times
        return self.tracking_data.get(track_index)

    def _get_absolute_window_range(self, index, track_index, random_slides):
        foi_index = self.tracking_data.get_foi_index(track_index)

        if random_slides:
            # Randomly take a window surrounding foi
            window_start = random.randint(-self.window_size + 1, 0)
            window_end = window_start + self.window_size - 1
        else:
            # Always set foi as middle-frame in window
            window_start = int(-(self.window_size - 1) / 2)
            window_end = int((self.window_size - 1) / 2)

        rel_foi_index = torch.tensor([-copy.copy(window_start)])

        start_index = foi_index + window_start
        end_index = foi_index + window_end

        return start_index, end_index, rel_foi_index

    def _get_pad_range(self, start_index, end_index, track_length):
        if start_index < 0:
            pad_start = -start_index
            start_index = 0
        else:
            pad_start = 0

        if end_index >= track_length:
            pad_end = end_index - track_length + 1
            end_index = track_length - 1
        else:
            pad_end = 0
        return start_index, end_index, pad_start, pad_end

    def _get_window_track_data(
        self, track_data, start_index, end_index, pad_start, pad_end
    ):
        # Extract the window
        wind_track_data = track_data[start_index : end_index + 1]

        # Pad
        wind_track_data = torch.cat(
            (
                torch.zeros(pad_start, wind_track_data.shape[1]),
                wind_track_data,
                torch.zeros(pad_end, wind_track_data.shape[1]),
            ),
            dim=0,
        )

        # Add temporal encoding to tracks
        wind_track_data_temp = torch.zeros((self.window_size, 8))
        wind_track_data_temp[:, :-1] = wind_track_data
        wind_track_data_temp[:, -1] = torch.arange(self.window_size)
        return wind_track_data_temp

    def _get_window_point_data(
        self, point_data, start_index, end_index, pad_start, pad_end
    ):
        if not self.use_pc:
            return point_data  # empty tensor

        # Extract, pad and add temporal encoding for point clouds
        wind_point_data = point_data[start_index : end_index + 1]
        wind_point_data = torch.cat(
            (
                torch.zeros(
                    pad_start, wind_point_data.shape[1], wind_point_data.shape[2]
                ),
                wind_point_data,
                torch.zeros(
                    pad_end, wind_point_data.shape[1], wind_point_data.shape[2]
                ),
            ),
            dim=0,
        )
        wind_point_data_temp = torch.zeros(
            (wind_point_data.shape[0], wind_point_data.shape[1], 4)
        )
        wind_point_data_temp[:, :, :-1] = wind_point_data
        wind_point_data_temp[:, :, -1] = (
            torch.arange(self.window_size)
            .unsqueeze(1)
            .expand(-1, wind_point_data.shape[1])
        )

        return wind_point_data_temp

    def _get_offset_data(self, wind_track_data, wind_point_data, gt_data):
        nonzero_indices = torch.nonzero(wind_track_data[:, :-1])  # (W,8)

        # Select first and last index where non-zero element in all but last element
        first_index = nonzero_indices[0, 0]
        last_index = nonzero_indices[-1, 0]

        offset_center = wind_track_data[first_index, 0:3]
        offset_rotation = wind_track_data[first_index, 6:7]
        rotation_matrix = self._get_rotation_matrix(offset_rotation)

        # Offset center and rotation for tracks
        offset_track_data = copy.deepcopy(wind_track_data)

        # print("track_centers", track_centers.shape)

        offset_track_data[first_index : last_index + 1, 0:3] = self.transform(
            offset_track_data[first_index : last_index + 1, 0:3].unsqueeze(1),
            offset_center,
            rotation_matrix,
        ).squeeze(1)
        offset_track_data[first_index : last_index + 1, 6:7] = (
            wind_track_data[first_index : last_index + 1, 6:7] - offset_rotation
        )

        # Offset all points using center
        offset_point_data = copy.deepcopy(wind_point_data)
        offset_point_data[first_index : last_index + 1, :, 0:3] = self.transform(
            wind_point_data[first_index : last_index + 1, :, 0:3],
            offset_center,
            rotation_matrix,
        )

        # Offset center and yaw for gt_data
        offset_gt_data = copy.deepcopy(gt_data)
        if gt_data[-2] == 1:
            # Compute offset if there is a GT associated
            offset_gt_data[0:3] = (
                self.transform(
                    gt_data[0:3].unsqueeze(0).unsqueeze(0),
                    offset_center,
                    rotation_matrix,
                )
                .squeeze(0)
                .squeeze(0)
            )
            offset_gt_data[6:7] = gt_data[6:7] - offset_rotation

        # Offset temporal encoding for window
        offset_track_data[first_index : last_index + 1, -1] = (
            offset_track_data[first_index : last_index + 1, -1] - first_index
        )
        offset_track_data[:first_index, -1] = 0
        offset_track_data[last_index + 1 :, -1] = 0

        offset_point_data[first_index : last_index + 1, :, -1] = (
            offset_point_data[first_index : last_index + 1, :, -1] - first_index
        )
        offset_point_data[:first_index, :, -1] = 0
        offset_point_data[last_index + 1 :, :, -1] = 0

        return offset_track_data, offset_point_data, offset_gt_data

    def _get_rotation_matrix(self, yaw):
        c, s = torch.cos(yaw), torch.sin(yaw)
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def transform(self, points, center, rotation_matrix):
        """
        points (W,N,3)
        center (3)
        rotation_matrix (3,3)
        """

        print(points.shape)
        if points.shape[1] == 1:
            print("points")
            print(points)

        rot_mat_transpose = rotation_matrix.T
        inverse_translate = -rot_mat_transpose @ center.unsqueeze(-1)
        trans = torch.cat((rot_mat_transpose, inverse_translate), dim=1)
        trans = torch.cat(
            (trans, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)), dim=0
        )

        points_homogeneous = torch.cat(
            (
                points.permute(0, 2, 1),
                torch.ones((points.shape[0], 1, points.shape[1]), dtype=torch.float32),
            ),
            dim=1,
        )
        local_points = trans @ points_homogeneous

        # Remove the homogeneous coordinate
        local_points = local_points.permute(0, 2, 1)[:, :, :3]

        if points.shape[1] == 1:
            print("local_points")
            print(local_points)

        return local_points
