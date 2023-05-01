from smoother.io.config_utils import load_config
from smoother.io.logging_utils import configure_loggings
from tools.utils.training_utils import TrainingUtils
from smoother.data.common.utils import convert_yaw_to_quat
import copy, mmcv
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
from smoother.models.pc_track_net import PCTrackNet, TrackNet, PCNet
import torch
from torch import optim
from torch.utils.data import random_split, DataLoader
import os
import json

from tools.utils.evaluation import giou3d


class SmoothingInferer():

    def __init__(
            self, 
            tracking_results_path, 
            conf_path, 
            data_path, 
            pc_name,
            save_path, 
        ):
        print("---Initializing SmoothingInferer class---")
        self.tracking_results_path = tracking_results_path
        self.conf_path = conf_path
        self.data_path = data_path
        self.pc_name = pc_name
        self.save_path = save_path

        self.conf = self._get_config(self.conf_path)

        # Update conf with pc path
        # self.conf["data"]["pc_path"] = '/staging/agp/masterthesis/2023autoann/storage/smoothing/autoannsmoothing/preprocessed_world/full_train'
        self.conf["data"]["pc_path"] = "storage/smoothing/autoannsmoothing/preprocessed_world_gravity/full_train"
        #self.conf["data"]["pc_path"] = str(os.path.join("preprocessed", self.pc_name))

        self.data_type = self.conf["data"]["type"]  # nuscenes / zod
        self.data_version = self.conf["data"]["version"]
        self.split = self.conf["data"]["split"]
        self.window_size = self.conf["data"]["window_size"]
        self.n_slides = self.conf["data"]["n_slides"]
        self.remove_non_gt_tracks = self.conf["data"]["remove_non_gt_tracks"]
        
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
        self.data_model = None
        self.model = None
        self.trained_model = None
        self.result_dict = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                        else "cpu")

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

        self.transformations, self.points_transformations = self._add_transformations(
            self.conf["data"]["transformations"]
        )

        # Get sequences
        seqs = self.tracking_results.seq_tokens

        # Load data model
        self.data_model = WindowTrackingData(
            tracking_results=self.tracking_results,
            window_size=self.window_size,
            n_slides=self.n_slides,
            use_pc=self.use_pc,
            transformations=self.transformations,
            points_transformations=self.points_transformations,
            remove_non_foi_tracks=True,
            remove_non_gt_tracks=self.remove_non_gt_tracks,
            seqs=seqs,
        )

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
        self.use_pc = self.conf["model"]["pc"]["use_pc"]
        self.use_track = self.conf["model"]["track"]["use_track"]

        pc_encoder = self.conf["model"]["pc"]["pc_encoder"]
        pc_out_size = self.conf["model"]["pc"]["pc_out_size"]
        track_encoder = self.conf["model"]["track"]["track_encoder"]
        track_out_size = self.conf["model"]["track"]["track_out_size"]
        decoder_name = self.conf["model"]["decoder"]["name"]
        dec_out_size = self.conf["model"]["decoder"]["dec_out_size"]

        model_class = self._get_model(self.use_pc, self.use_track)

        self.model = model_class(
            track_encoder=track_encoder,
            pc_encoder=pc_encoder,
            decoder=decoder_name,
            pc_feat_dim=POINT_FEAT_DIM,
            track_feat_dim=TRACK_FEAT_DIM,
            pc_out=pc_out_size,
            track_out=track_out_size,
            dec_out=dec_out_size,
        )


        self.trained_model = self.model
        checkpoint = torch.load(model_path)
        self.trained_model.load_state_dict(checkpoint)
        print("---Finished loading trained model---")

    def _get_model(self, use_pc, use_track):
        if use_pc and use_track:
            return PCTrackNet
        if use_pc:
            return PCNet
        if use_track:
            return TrackNet
        raise NotImplementedError("Model without point-cloud nor tracks not available")

    def infer(self):
        print("---Starting inference---")
        self.trained_model.eval()
        
        data_loader = DataLoader(
            self.data_model, 
            batch_size=1, 
            shuffle=True, 
            num_workers=self.n_workers
        )
        
        self.trained_model.to(self.device)

        self.result_dict = {}
        self.result_dict['results'] = {}
        self.result_dict['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False
        }

        for batch_index, (x1, x2, y) in enumerate(data_loader):
            track, points, gt_anns = (
                    x1.to(self.device),
                    x2.to(self.device),
                    y.to(self.device),
            )


            track_obj = self.data_model.get(batch_index)
            old_foi_index = copy.copy(track_obj.foi_index)
            
            #only running inference on FOI now
            frame_track_index = 'foi'

            frame_track_index = (
                track_obj.foi_index - track_obj.starting_frame_index
                if frame_track_index == "foi"
                else frame_track_index
            )

            track_obj.foi_index = frame_track_index + track_obj.starting_frame_index

            center_out, size_out, rot_out, score_out = self.trained_model.forward(
                track, points
            )

            center_out = center_out.squeeze(0)
            size_out = size_out.squeeze(0)
            rot_out = rot_out.squeeze(0)
            score_out = score_out.squeeze(0)

            track = track.squeeze(0)
            mid_wind = track.shape[0] // 2 + 1

            c_hat = track[mid_wind, 0:3] + center_out[mid_wind]
            s_hat = track[mid_wind, 3:6] + size_out
            r_hat = track[mid_wind, 6].unsqueeze(-1) + rot_out[mid_wind]

            score = score_out[mid_wind]
            
            model_out = torch.cat((c_hat, s_hat, r_hat, score), dim=-1).squeeze().detach()
            
            # Remove offset
            absolute_starting_index = frame_track_index - (self.window_size // 2 + 1)
            window_starting_box_index = max(0, absolute_starting_index)
            center_offset = torch.tensor(
                track_obj.boxes[window_starting_box_index].center
            ).float().to(self.device)
            rotation_offset = torch.tensor(
                track_obj.boxes[window_starting_box_index].rotation
            ).float().to(self.device)
            model_out[0:3] = model_out[0:3] + center_offset
            model_out[6:7] = model_out[6:7] + rotation_offset

            # create refined box and add to image
            center = model_out[0:3].cpu().numpy()
            size = model_out[3:6].cpu().numpy()
            rotation = convert_yaw_to_quat(model_out[6:7].cpu().numpy())

            track_obj.foi_index = old_foi_index

            track_id = track_obj.tracking_id
            track_box = track_obj.boxes[window_starting_box_index]
                    
            refined_box = {
                'sample_token': track_box.frame_token,
                'translation': center.tolist(),
                'size': size.tolist(),
                'rotation': rotation.elements,
                'velocity': [0,0],
                'tracking_id': track_id,
                'tracking_name': track_box.tracking_name,
                'tracking_score': float(score),
            }

            if track_box.frame_token not in self.result_dict['results']:
                self.result_dict['results'][track_box.frame_token] = []

            self.result_dict['results'][track_box.frame_token].append(refined_box)
        mmcv.dump(self.result_dict, self.save_path)
