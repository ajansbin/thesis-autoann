from zod.constants import (
    Camera,
    Lidar,
    Anonymization,
    AnnotationProject,
    EGO,
    CoordinateFrame,
)
import torch
import copy
from smoother.data.common.utils import convert_yaw_to_quat
from matplotlib import pyplot as plt
from zod.data_classes.box import Box3D
from zod.data_classes.sensor import LidarData
from zod.visualization.lidar_on_image import visualize_lidar_on_image
from zod.visualization.object_visualization import overlay_object_3d_box_on_image
from zod.data_classes.geometry import Pose
import numpy as np


class VisualizerTools:
    def __init__(self, result_data, track_data, trained_model):
        self.result_data = result_data
        self.track_data = track_data
        self.trained_model = trained_model

        self.window_size = self.track_data.window_size

        self.det_color = (0, 0, 255)  # Blue
        self.ref_color = (255, 255, 0)  # Yellow
        self.gt_color = (255, 0, 0)  # Red
        self.line_thickness = 3

        self.trained_model

    def set_trained_model(self, trained_model):
        self.trained_model = trained_model

    def plot_frame(
        self,
        seq,
        tracks_same_seq,
        lidar_index,
        camera_frame,
        lidar_frame,
        show_lidar=True,
        show_det=True,
        show_ref=True,
        show_gt=True,
        score_thresh=0.0,
    ):
        image = camera_frame.read()

        for track_index, track_same_seq in tracks_same_seq:
            if (
                lidar_index < track_same_seq.starting_frame_index
                or lidar_index
                >= track_same_seq.starting_frame_index + len(track_same_seq.boxes)
            ):
                continue

            # print(track_index, track_same_seq)
            frame_track_index = lidar_index - track_same_seq.starting_frame_index
            track_box = track_same_seq.boxes[frame_track_index]

            track_box3d = self._create_track_box(track_box, lidar_frame, seq)
            # print("track_box3d", track_box3d)
            if show_det:
                image = self._add_box_to_image(
                    image,
                    track_box3d,
                    seq,
                    color=self.det_color,
                    line_thickness=self.line_thickness,
                )

            if show_lidar:
                _, points, _ = self.track_data[track_index]
                points = points[:, :, :3]
                masked_lidar = self._get_masked_lidar(
                    track_box3d,
                    track_box.frame_token,
                    track_box.frame_index,
                    lidar_frame,
                    seq,
                )
                if (
                    masked_lidar.points.shape[0] == 1
                    or masked_lidar.points.shape[0] > 50000
                ):
                    continue
                image = self._add_lidar_to_image(image, masked_lidar, seq)

            if show_ref:
                ref_box, score = self._get_refined_box(
                    self.trained_model,
                    self.track_data,
                    track_index,
                    frame_track_index,
                    lidar_frame,
                    seq,
                    score_thresh,
                )
                if ref_box:
                    image = self._add_box_to_image(
                        image,
                        ref_box,
                        seq,
                        color=self.ref_color,
                        line_thickness=self.line_thickness,
                    )

            if lidar_index == track_same_seq.foi_index and show_gt:
                if track_same_seq.has_gt:
                    gt_box = self._get_gt_box(track_same_seq, lidar_frame, seq)
                    image = self._add_box_to_image(
                        image,
                        gt_box,
                        seq,
                        color=self.gt_color,
                        line_thickness=self.line_thickness,
                    )
                else:
                    print("Track does not have gt, skipping gt-box")

        return image

    def _get_tracks_in_seq(self, seq_id):
        tracks_same_seq = []
        for i in range(len(self.track_data)):
            t = self.track_data.get(i)
            if t.sequence_id == seq_id:
                tracks_same_seq.append((i, t))
        return tracks_same_seq

    def _extract_track_and_sequence_info(
        self, track_index, frame_track_index, track_data
    ):
        track = track_data.get(track_index)
        seq = self.result_data.zod[track.sequence_id]
        camera_lidar_map = self._get_camera_lidar_index_map(seq)

        track_box_index = (
            track.foi_index - track.starting_frame_index
            if frame_track_index == "foi"
            else frame_track_index
        )
        track_box = track.boxes[track_box_index]

        track_lidar_index = track.starting_frame_index + track_box_index
        track_camera_index = camera_lidar_map.index(track_lidar_index)
        return track, seq, track_camera_index, track_box

    def _get_camera_lidar_index_map(self, seq):
        frames = seq.info.get_camera_lidar_map(
            Anonymization.BLUR, Camera.FRONT, Lidar.VELODYNE
        )
        camera_lidar_map = []
        lidar_i = 0
        _, lidar_frame = next(frames)
        lidar_name = lidar_frame.filepath
        camera_lidar_map.append(lidar_i)
        for _, lidar_frame in frames:
            if lidar_frame.filepath != lidar_name:
                lidar_i += 1
                lidar_name = lidar_frame.filepath
            camera_lidar_map.append(lidar_i)

        return camera_lidar_map

    def _create_track_box(self, track_box, lidar_frame, seq):
        track_box = self._get_box(
            track_box.center,
            track_box.size,
            convert_yaw_to_quat(track_box.rotation),
            lidar=False,
            lidar_frame=lidar_frame,
            seq=seq,
        )

        return track_box

    def _get_gt_box(self, track, lidar_frame, seq):
        # function code
        gt_box_dict = track.gt_box
        gt_translation = gt_box_dict["translation"]
        gt_size = gt_box_dict["size"]
        gt_rotation = convert_yaw_to_quat(gt_box_dict["rotation"])

        gt_box = self._get_box(
            gt_translation,
            gt_size,
            gt_rotation,
            lidar=False,
            lidar_frame=lidar_frame,
            seq=seq,
        )

        return gt_box

    def _get_refined_box(
        self,
        model,
        track_data,
        track_index,
        frame_track_index,
        lidar_frame,
        seq,
        score_thresh,
    ):
        track_obj = track_data.get(track_index)
        old_foi_index = copy.copy(track_obj.foi_index)

        frame_track_index = (
            track.foi_index - track.starting_frame_index
            if frame_track_index == "foi"
            else frame_track_index
        )

        track_obj.foi_index = frame_track_index + track_obj.starting_frame_index
        track, point, gt = track_data[track_index]
        center_out, size_out, rot_out, score_out = model.forward(
            track.unsqueeze(0), point.unsqueeze(0)
        )

        center_out = center_out.squeeze(0)
        size_out = size_out.squeeze(0)
        rot_out = rot_out.squeeze(0)
        score_out = score_out.squeeze(0)

        mid_wind = track.shape[0] // 2 + 1
        c_hat = track[mid_wind, 0:3] + center_out[mid_wind]
        s_hat = track[mid_wind, 3:6] + size_out
        r_hat = track[mid_wind, 6:7] + rot_out[mid_wind]

        score = score_out[mid_wind]

        if score < score_thresh:
            return (None, score)

        model_out = torch.cat((c_hat, s_hat, r_hat, score), dim=-1).squeeze().detach()

        # Remove offset
        # Select first and last index where non-zero element in all but last element
        absolute_starting_index = frame_track_index - (self.window_size // 2 + 1)
        window_starting_box_index = max(0, absolute_starting_index)

        center_offset = torch.tensor(
            track_obj.boxes[window_starting_box_index].center
        ).float()
        rotation_offset = torch.tensor(
            track_obj.boxes[window_starting_box_index].rotation
        ).float()

        rot_matrix = self._get_rotation_matrix(rotation_offset)
        local_center = model_out[0:3].reshape(1, 3)
        center_offset = center_offset.reshape(1, 3)
        center_world = self.transform(local_center, center_offset, rot_matrix)

        model_out[0:3] = center_world
        model_out[6:7] += rotation_offset

        # create refined box and add to image
        center = model_out[0:3].numpy()
        size = model_out[3:6].numpy()
        rotation = convert_yaw_to_quat(model_out[6:7].numpy())
        ref_box = self._get_box(
            center, size, rotation, lidar=False, lidar_frame=lidar_frame, seq=seq
        )

        track_obj.foi_index = old_foi_index

        return (ref_box, score)

    def _get_box(self, center, size, rotation, lidar=False, lidar_frame=None, seq=None):
        if lidar:
            return Box3D(center, size, rotation, Lidar.VELODYNE)
        box = Box3D(np.array(center), np.array(size), rotation, EGO)
        core_timestamp = lidar_frame.time.timestamp()
        core_ego_pose = Pose(seq.ego_motion.get_poses(core_timestamp))
        box._transform_inv(core_ego_pose, EGO)
        box.convert_to(Lidar.VELODYNE, seq.calibration)

        return box

    def _get_masked_lidar(
        self, track_box3d, frame_token, frame_index, lidar_frame, seq
    ):
        core_lidar = self.result_data.get_lidar_data_in_frame(
            frame_token, frame_index, lidar=True
        )
        # core_lidar = lidar_frame.read()

        points = core_lidar.points

        # masked_points = track_box3d.get_points_in_bbox(points)
        # print(masked_points.shape)

        masked_lidar = LidarData(
            points,
            core_lidar.timestamps,
            core_lidar.intensity,
            core_lidar.diode_idx,
            core_lidar.core_timestamp,
        )
        # masked_lidar.transform(seq.calibration.get_extrinsics(EGO))

        return core_lidar

    def _plot_image(self, image):
        plt.rcParams["figure.figsize"] = [20, 10]
        plt.axis("off")
        plt.imshow(image)
        plt.show()

    def _add_lidar_to_image(self, image, masked_lidar, seq):
        return visualize_lidar_on_image(
            masked_lidar,
            seq.calibration,
            image,
        )

    def _add_box_to_image(self, image, box3d, seq, color=(0, 255, 0), line_thickness=2):
        return overlay_object_3d_box_on_image(
            image, box3d, seq.calibration, color, line_thickness
        )

    def _get_rotation_matrix(self, yaw):
        c, s = torch.cos(yaw), torch.sin(yaw)
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def transform(self, local_center, center_offset, rotation_matrix):
        """
        local_center (1,3)
        center (1,3)
        rotation_matrix (3,3)
        """

        trans = torch.cat((rotation_matrix, center_offset.transpose(1, 0)), dim=-1)
        trans = torch.cat(
            (trans, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)), dim=0
        )

        ones = torch.ones((1, 1), dtype=torch.float32)

        points_homogeneous = torch.cat((local_center, ones), dim=-1)

        transformed_points = trans @ points_homogeneous.T

        # Remove the homogeneous coordinate
        transformed_points = transformed_points[0:3].squeeze(-1)

        return transformed_points
