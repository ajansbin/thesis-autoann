from smoother.io.config_utils import load_config
from smoother.io.logging_utils import configure_loggings
from tools.utils.training_utils import TrainingUtils
from smoother.data.common.utils import convert_yaw_to_quat
import copy, mmcv, tqdm
from smoother.data.common.tracking_data import (
    WindowTrackingData,
    TRACK_FEAT_DIM,
    POINT_FEAT_DIM,
)
from smoother.data.common.transformations import (
    ToTensor,
    CenterOffset,
    YawOffset,
    Normalize,
    PointsShift,
    PointsScale,
)
from smoother.models.pc_track_net import (
    PCTrackNet,
    TrackNet,
    PCNet,
    PCTrackEarlyFusionNet,
)
import torch
from torch import optim
from torch.utils.data import random_split, DataLoader
import os
import json

from tools.utils.evaluation import giou3d
from zod import ZodSequences
from zod.constants import Lidar, EGO
from zod.data_classes.box import Box3D
from zod.data_classes.geometry import Pose


class SmoothingInferer:
    def __init__(
        self,
        tracking_results_path,
        conf_path,
        data_path,
        version,
        split,
        pc_name,
        save_path,
    ):
        print("---Initializing SmoothingInferer class---")
        self.tracking_results_path = tracking_results_path
        self.conf_path = conf_path
        self.data_path = data_path
        self.version = version
        self.split = split
        self.pc_name = pc_name
        self.save_path = save_path

        self.conf = self._get_config(self.conf_path)

        # Update conf with pc path
        # self.conf["data"]["pc_path"] = '/staging/agp/masterthesis/2023autoann/storage/smoothing/autoannsmoothing/preprocessed_world/full_train'
        # self.conf["data"][
        #    "pc_path"
        # ] = "/staging/agp/masterthesis/2023autoann/storage/smoothing/autoannsmoothing/preprocessed_world_gravity/full_train"
        # self.conf["data"]["pc_path"] = str(os.path.join("preprocessed", self.pc_name))

        self.data_type = self.conf["data"]["type"]  # nuscenes / zod
        self.window_size = self.conf["data"]["window_size"]
        self.times = self.conf["data"]["times"]
        self.random_slides = self.conf["data"]["random_slides"]
        self.remove_non_gt_tracks = self.conf["data"]["remove_non_gt_tracks"]

        self.zod = ZodSequences(self.data_path, self.version)

        self.use_pc = self.conf["model"]["pc"]["use_pc"]
        self.use_track = self.conf["model"]["track"]["use_track"]

        self.lr = self.conf["train"]["learning_rate"]
        self.wd = self.conf["train"]["weight_decay"]
        self.n_epochs = self.conf["train"]["n_epochs"]
        self.seed = self.conf["train"]["seed"]
        self.train_size = self.conf["train"]["train_size"]
        self.batch_size = self.conf["train"]["batch_size"]
        self.n_workers = self.conf["train"]["n_workers"]

        self.center_dim = 3
        self.size_dim = 3
        self.rotation_dim = 1
        self.score_dim = 1

        torch.manual_seed(self.seed)

        self.tracking_results = None
        self.transformations = None
        self.points_transformations = None
        self.track_data = None
        self.model = None
        self.trained_model = None
        self.result_dict = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_config(self, conf_path):
        return load_config(conf_path)

    def load_data(self):
        print("---Loading data---")
        # Load datatype specific results from tracking
        if self.data_type == "nuscenes":
            from smoother.data.nuscenes_data import NuscTrackingResults

            self.tracking_results = NuscTrackingResults(
                self.tracking_results_path,
                self.conf,
                self.version,
                self.split,
                self.data_path,
            )
        elif self.data_type == "zod":
            from smoother.data.zod_data import ZodTrackingResults

            self.tracking_results = ZodTrackingResults(
                self.tracking_results_path,
                self.conf,
                self.version,
                self.split,
                self.data_path,
            )
        else:
            raise NotImplementedError(
                f"Dataclass of type {self.data_type} is not implemented. Please use 'nuscenes' or 'zod'."
            )

        transformations, points_transformations = self._add_transformations(
            self.conf["data"]["transformations"]
        )

        # Get sequences
        seqs = self.tracking_results.seq_tokens

        # Load data model
        self.track_data = WindowTrackingData(
            tracking_results=self.tracking_results,
            window_size=self.window_size,
            times=self.times,
            random_slides=self.random_slides,
            use_pc=self.use_pc,
            transformations=transformations,
            points_transformations=points_transformations,
            remove_non_foi_tracks=True,
            remove_non_gt_tracks=self.remove_non_gt_tracks,
            seqs=seqs,
        )

    def _get_train_val_seq_split(self, seqs: list):
        # Create a list of indices
        indices = torch.arange(len(seqs))

        # Shuffle the indices
        shuffled_indices = indices[torch.randperm(len(indices))]

        # Calculate the lengths of the train and validation sets
        train_len = int(len(seqs) * self.train_size)
        val_len = len(seqs) - train_len

        # Split the indices into train and validation sets
        train_indices = shuffled_indices[:train_len].tolist()
        val_indices = shuffled_indices[train_len:].tolist()

        # Get the sequences corresponding to the respective indices
        train_seqs = [seqs[i] for i in train_indices]
        val_seqs = [seqs[i] for i in val_indices]

        return train_seqs, val_seqs

    def _add_transformations(self, transformations_dict):
        transformations = [ToTensor()]

        if transformations_dict["center_offset"]:
            transformations.append(CenterOffset())
        if transformations_dict["yaw_offset"]:
            transformations.append(YawOffset())

        if transformations_dict["normalize"]["normalize"]:
            center_mean = transformations_dict["normalize"]["center"]["mean"]
            center_stdev = transformations_dict["normalize"]["center"]["stdev"]
            size_mean = transformations_dict["normalize"]["size"]["mean"]
            size_stdev = transformations_dict["normalize"]["size"]["stdev"]
            rotation_mean = transformations_dict["normalize"]["rotation"]["mean"]
            rotation_stdev = transformations_dict["normalize"]["rotation"]["stdev"]

            means = center_mean + size_mean + rotation_mean
            stdev = center_stdev + size_stdev + rotation_stdev
            transformations.append(Normalize(means, stdev))
        if transformations_dict["center_offset"]:
            transformations.append(CenterOffset())
        if transformations_dict["yaw_offset"]:
            transformations.append(YawOffset())

        points_transformations = []
        if transformations_dict["points"]["points_shift"]:
            shift_max_size = transformations_dict["points"]["shift_max_size"]
            points_transformations.append(PointsShift(shift_max_size))
        if transformations_dict["points"]["points_scale"]:
            scale_min = transformations_dict["points"]["scale_min"]
            scale_max = transformations_dict["points"]["scale_max"]
            points_transformations.append(PointsScale(scale_min, scale_max))

        return transformations, points_transformations

    def load_model(self, model_path):
        print("---Loading model---")
        self.early_fuse = self.conf["model"]["early_fuse"]["early_fuse"]
        fuse_encoder = self.conf["model"]["early_fuse"]["fuse_encoder"]
        encoder_out_size = self.conf["model"]["early_fuse"]["encoder_out_size"]

        self.use_pc = self.conf["model"]["pc"]["use_pc"]
        pc_encoder = self.conf["model"]["pc"]["pc_encoder"]
        pc_out_size = self.conf["model"]["pc"]["pc_out_size"]

        self.use_track = self.conf["model"]["track"]["use_track"]
        track_encoder = self.conf["model"]["track"]["track_encoder"]
        track_out_size = self.conf["model"]["track"]["track_out_size"]

        temporal_encoder = self.conf["model"]["temporal_encoder"]
        dec_out_size = self.conf["model"]["dec_out_size"]

        # decoder_name = self.conf["model"]["decoder"]["name"]
        # dec_out_size = self.conf["model"]["decoder"]["dec_out_size"]

        if self.early_fuse:
            self.model = PCTrackEarlyFusionNet(
                fuse_encoder,
                encoder_out_size,
                temporal_encoder,
                dec_out_size,
                track_feat_dim=TRACK_FEAT_DIM,
                pc_feat_dim=POINT_FEAT_DIM,
                window_size=self.window_size,
            )
        elif self.use_pc and self.use_track:
            self.model = PCTrackNet(
                track_encoder_name=track_encoder,
                pc_encoder_name=pc_encoder,
                temporal_encoder_name=temporal_encoder,
                track_feat_dim=TRACK_FEAT_DIM,
                pc_feat_dim=POINT_FEAT_DIM,
                track_out=track_out_size,
                pc_out=pc_out_size,
                dec_out_size=dec_out_size,
                window_size=self.window_size,
            )

        self.trained_model = self.model
        checkpoint = torch.load(model_path)
        self.trained_model.load_state_dict(checkpoint)
        self.trained_model.eval()
        self.trained_model.to(self.device)
        print("---Finished loading trained model---")

    def _get_model(self, early_fuse, use_pc, use_track):
        if early_fuse:
            return PCTrackEarlyFusionNet
        if use_pc and use_track:
            return PCTrackNet
        if use_pc:
            return PCNet
        if use_track:
            return TrackNet
        raise NotImplementedError("Model without point-cloud nor tracks not available")

    def infer(self):
        print("---Starting inference---")

        self.trained_model.to(self.device)

        self.result_dict = {}
        self.result_dict["results"] = {}
        self.result_dict["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        for scene_index, seq_id in tqdm.tqdm(enumerate(self.zod.get_split(self.split))):
            seq = self.zod[seq_id]
            lidar_frames = seq.info.get_lidar_frames(lidar=Lidar.VELODYNE)

            tracks_same_seq = self._get_tracks_in_seq(seq_id)

            for lidar_index, lidar_frame in enumerate(lidar_frames[:-1]):
                key_lidar_frame = seq.info.get_key_lidar_frame(lidar=Lidar.VELODYNE)
                if lidar_frame != key_lidar_frame:
                    continue
                lidar_path = lidar_frame.to_dict()["filepath"]
                frame_token = os.path.basename(lidar_path)

                if frame_token not in self.result_dict["results"]:
                    self.result_dict["results"][frame_token] = []

                for track_index, track_same_seq in tracks_same_seq:
                    if (
                        lidar_index < track_same_seq.starting_frame_index
                        or lidar_index
                        >= track_same_seq.starting_frame_index
                        + len(track_same_seq.boxes)
                    ):
                        continue

                    frame_track_index = (
                        lidar_index - track_same_seq.starting_frame_index
                    )
                    track_box = track_same_seq.boxes[frame_track_index]

                    assert frame_token == track_box.frame_token

                    ref_box, score = self._get_refined_box(
                        self.trained_model,
                        self.track_data,
                        track_index,
                        frame_track_index,
                        lidar_frame,
                        seq,
                        0.0,
                    )

                    refined_box = {
                        "sample_token": track_box.frame_token,
                        "translation": ref_box.center.tolist(),
                        "size": ref_box.size.tolist(),
                        "rotation": ref_box.orientation.elements.tolist(),
                        "velocity": [0, 0],
                        "tracking_id": track_box.tracking_id,
                        "tracking_name": track_box.tracking_name,
                        "tracking_score": float(score),
                    }

                    self.result_dict["results"][track_box.frame_token].append(
                        refined_box
                    )
        mmcv.dump(self.result_dict, self.save_path)
        print("Inference saved at", self.save_path)

    def _get_tracks_in_seq(self, seq_id):
        tracks_same_seq = []
        for i in range(len(self.track_data)):
            t = self.track_data.get(i)
            if t.sequence_id == seq_id:
                tracks_same_seq.append((i, t))
        return tracks_same_seq

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
        track, point, gt = (
            track.to(self.device),
            point.to(self.device),
            gt.to(self.device),
        )
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
        absolute_starting_index = frame_track_index - (self.window_size // 2 + 1)
        window_starting_box_index = max(0, absolute_starting_index)
        center_offset = (
            torch.tensor(track_obj.boxes[window_starting_box_index].center)
            .float()
            .to(self.device)
        )
        rotation_offset = (
            torch.tensor(track_obj.boxes[window_starting_box_index].rotation)
            .float()
            .to(self.device)
        )

        rot_matrix = self._get_rotation_matrix(rotation_offset)
        local_center = model_out[0:3].reshape(1, 3)
        center_offset = center_offset.reshape(1, 3)
        center_world = self.transform(local_center, center_offset, rot_matrix)

        model_out[0:3] = center_world
        model_out[6:7] += rotation_offset

        # create refined box and add to image
        center = model_out[0:3].cpu().detach().numpy()
        size = model_out[3:6].cpu().detach().numpy()
        rotation = convert_yaw_to_quat(model_out[6:7].cpu().detach().numpy())
        ref_box = self._get_box(
            center, size, rotation, lidar=False, lidar_frame=lidar_frame, seq=seq
        )

        track_obj.foi_index = old_foi_index

        return (ref_box, score)

    def _get_rotation_matrix(self, yaw):
        c, s = torch.cos(yaw), torch.sin(yaw)
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]]).to(self.device)

    def transform(self, local_center, center_offset, rotation_matrix):
        """
        local_center (1,3)
        center (1,3)
        rotation_matrix (3,3)
        """

        trans = torch.cat(
            (rotation_matrix, center_offset.transpose(1, 0)), dim=-1
        )  # (3,4)
        trans = torch.cat(
            (trans, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).to(self.device)),
            dim=0,
        )

        ones = torch.ones((1, 1), dtype=torch.float32).to(self.device)

        points_homogeneous = torch.cat((local_center, ones), dim=-1)

        transformed_points = trans @ points_homogeneous.T

        # Remove the homogeneous coordinate
        transformed_points = transformed_points[0:3].squeeze(-1)

        return transformed_points

    def _get_box(self, center, size, rotation, lidar=False, lidar_frame=None, seq=None):
        if lidar:
            return Box3D(center, size, rotation, Lidar.VELODYNE)
        box = Box3D(center, size, rotation, EGO)
        core_timestamp = lidar_frame.time.timestamp()
        core_ego_pose = Pose(seq.ego_motion.get_poses(core_timestamp))
        box._transform_inv(core_ego_pose, EGO)
        box.convert_to(Lidar.VELODYNE, seq.calibration)

        return box
