from smoother.io.config_utils import load_config
from smoother.io.logging_utils import configure_loggings
from tools.utils.training_utils import TrainingUtils
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
from torch.optim.lr_scheduler import ExponentialLR


class SmoothingTrainer:
    def __init__(
        self,
        tracking_results_path,
        tracking_preprocessed_path,
        conf_path,
        data_path,
        pc_name,
        save_dir,
        run_name,
    ):
        print("---Initializing SmoothingTrainer class---")
        self.tracking_results_path = tracking_results_path
        self.tracking_preprocessed_path = tracking_preprocessed_path
        self.conf_path = conf_path
        self.data_path = data_path
        self.pc_name = pc_name
        self.save_dir = save_dir
        self.run_name = run_name

        self.conf = self._get_config(self.conf_path)

        # Update conf with pc path
        self.conf["data"]["pc_path"] = str(os.path.join("preprocessed", self.pc_name))

        self.data_type = self.conf["data"]["type"]  # nuscenes / zod
        self.data_version = self.conf["data"]["version"]
        self.split = self.conf["data"]["split"]
        self.window_size = self.conf["data"]["window_size"]
        self.times = self.conf["data"]["times"]
        self.random_slides = self.conf["data"]["random_slides"]
        self.remove_non_gt_tracks = self.conf["data"]["remove_non_gt_tracks"]

        # Needed for both model and dataclass
        self.use_pc = self.conf["model"]["pc"]["use_pc"]

        self.lr = self.conf["train"]["learning_rate"]
        self.wd = self.conf["train"]["weight_decay"]
        self.n_epochs = self.conf["train"]["n_epochs"]
        self.seed = self.conf["train"]["seed"]
        self.train_size = self.conf["train"]["train_size"]
        self.batch_size = self.conf["train"]["batch_size"]
        self.n_workers = self.conf["train"]["n_workers"]

        torch.manual_seed(self.seed)

        self.tracking_results = None
        self.train_data_model = None
        self.val_data_model = None
        self.model = None
        self.trained_model = None
        self.result_dict = {}

        print("---Setting up wandb-logger---")
        self.log_out, self.run = configure_loggings(
            self.run_name, self.save_dir, self.conf
        )

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
                self.data_version,
                self.split,
                self.data_path,
            )
        elif self.data_type == "zod":
            from smoother.data.zod_data import ZodTrackingResults

            self.tracking_results = ZodTrackingResults(
                self.tracking_results_path,
                self.conf,
                self.data_version,
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

        # Get train and val split sequences
        train_seqs, val_seqs = self._get_train_val_seq_split(
            self.tracking_results.seq_tokens
        )
        # Load data model
        self.train_data_model = WindowTrackingData(
            tracking_results=self.tracking_results,
            window_size=self.window_size,
            times=self.times,
            random_slides=self.random_slides,
            use_pc=self.use_pc,
            transformations=transformations,
            points_transformations=points_transformations,
            remove_non_foi_tracks=True,
            remove_non_gt_tracks=self.remove_non_gt_tracks,
            seqs=train_seqs,
            tracking_preprocessed_path=self.tracking_preprocessed_path,
        )

        self.val_data_model = WindowTrackingData(
            tracking_results=self.tracking_results,
            window_size=self.window_size,
            times=self.times,
            random_slides=self.random_slides,
            use_pc=self.use_pc,
            transformations=transformations,
            points_transformations=points_transformations,
            remove_non_foi_tracks=True,
            remove_non_gt_tracks=self.remove_non_gt_tracks,
            seqs=val_seqs,
            tracking_preprocessed_path=self.tracking_preprocessed_path,
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

    def load_model(self):
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
        elif self.use_pc:
            self.model = PCNet(
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
        else:
            self.model = TrackNet(
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

    def train(self):
        print("---Starting training---")

        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        scheduler = ExponentialLR(optimizer, gamma=0.98, verbose=True)

        tu = TrainingUtils(self.conf, self.log_out, scheduler)

        train_dataloader = DataLoader(
            self.train_data_model,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
        )
        val_dataloader = DataLoader(
            self.val_data_model,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
        )

        self.trained_model, train_losses, val_losses = tu.training_loop(
            self.model, optimizer, self.n_epochs, train_dataloader, val_dataloader
        )
        self.result_dict = {"train_losses": train_losses, "val_losses": val_losses}
        print("---Finished training---")

    def save_results(self):
        # Save to file
        if self.use_pc and self.use_track:
            model_type = "pc-track"
        elif self.use_pc:
            model_type = "pc"
        elif self.use_track:
            model_type = "track"

        save_dir_full = os.path.join(self.save_dir, model_type)
        if not os.path.exists(save_dir_full):
            os.mkdir(save_dir_full)

        # save model
        model_path = os.path.join(save_dir_full, self.run_name + "_model.pth")
        torch.save(self.trained_model.state_dict(), model_path)

        # save config
        conf_path = os.path.join(save_dir_full, self.run_name + "_conf.json")
        with open(conf_path, "w") as f:
            json.dump(self.conf, f)
