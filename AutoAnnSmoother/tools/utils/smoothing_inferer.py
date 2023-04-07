from smoother.io.config_utils import load_config
from smoother.data.common.sequence_data import SequenceData
from smoother.data.common.transformations import ToTensor, CenterOffset, YawOffset, Normalize
from smoother.models.pointnet import PointNet
from smoother.models.transformer import PointTransformer
from tools.utils.inference_utils import WindowInferer, SlidingWindowInferer
import tqdm
import mmcv
import os
from collections import defaultdict


import torch
from torch import optim
from torch.utils.data import random_split, DataLoader

from tools.utils.evaluation import giou3d


class SmoothingInferer():

    def __init__(self, tracking_results_path, conf_path, data_path, save_dir, seq_id):
        print("---Initializing SmoothingInferer class---")
        self.tracking_results_path = tracking_results_path
        self.conf_path = conf_path
        self.data_path = data_path
        self.save_dir = save_dir
        self.seq_id = seq_id



        self.conf = self._get_config(self.conf_path)
        
        self.conf["data"] = self.conf["data"]["value"]
        self.conf["train"] = self.conf["train"]["value"]
        self.conf["test"] = self.conf["test"]["value"]
        self.conf["model"] = self.conf["model"]["value"]

        #print(self.conf["data"].keys())
        self.data_type = self.conf["data"]["type"] # nuscenes / zod
        self.data_version = self.conf["test"]["data"]["version"]
        self.split = self.conf["test"]["data"]["split"]

        self.foi_index = self.conf["data"]["foi_index"] # index in sequence where annotation exist
        self.window_size = self.conf["data"]["window_size"]
        self.sliding_window = self.conf["data"]["sliding_window"]
        
        self.model_type = self.conf["model"]["type"]

        self.batch_size = self.conf["train"]["batch_size"]
        self.n_workers = self.conf["train"]["n_workers"]

        self.tracking_results = None
        self.transformations = None
        self.data_model = None
        self.model = None
        self.trained_model = None
        self.smoothing_result = []
        self.gt = []
        self.frame_tokens = []

        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                        else "cpu")

    def _get_config(self, conf_path):
        return load_config(conf_path)

    def load_data(self):
        print("---Loading data---")
        # Load datatype specific results from tracking
        if self.data_type == 'nuscenes':
            from smoother.data.nuscenes_data import NuscTrackingResults
            self.tracking_results = NuscTrackingResults(self.tracking_results_path, self.conf, self.data_version, self.split, self.data_path)
        elif self.data_type == 'zod':
            from smoother.data.zod_data import ZodTrackingResults
            self.tracking_results = ZodTrackingResults(self.tracking_results_path, self.conf, self.data_version, self.split, self.data_path)
        else:
            raise NotImplementedError(f"Dataclass of type {self.data_type} is not implemented. Please use 'nuscenes' or 'zod'.")
        
        self.transformations = self._add_transformations(self.conf["data"]["transformations"])
        
        self.frame_tokens = self.tracking_results.get_frames_in_sequence(self.seq_id)
        
        # Load data model
        self.data_model = SequenceData(self.tracking_results, self.seq_id, self.transformations)

    def _add_transformations(self, transformations_dict):
        transformations = [ToTensor(self.device)]

        if transformations_dict["normalize"]["normalize"]:
            center_mean = transformations_dict["normalize"]["center"]["mean"]
            center_stdev = transformations_dict["normalize"]["center"]["stdev"]
            size_mean = transformations_dict["normalize"]["size"]["mean"]
            size_stdev = transformations_dict["normalize"]["size"]["stdev"]
            rotation_mean = transformations_dict["normalize"]["rotation"]["mean"]
            rotation_stdev = transformations_dict["normalize"]["rotation"]["stdev"]
            score_mean = transformations_dict["normalize"]["score"]["mean"]
            score_stdev = transformations_dict["normalize"]["score"]["stdev"]

            means = center_mean + size_mean + rotation_mean + score_mean
            stdev = center_stdev + size_stdev + rotation_stdev + score_stdev
            transformations.append(Normalize(means,stdev, self.device))
        if transformations_dict["center_offset"]:
            transformations.append(CenterOffset())

        return transformations
    
    def load_model(self, model_path):
        print("---Loading trained model---")
        if self.model_type == 'pointnet':
            input_dim = self.conf["model"][self.model_type]["input_dim"]
            out_size = self.conf["model"][self.model_type]["out_size"]
            mlp1_sizes = self.conf["model"][self.model_type]["mlp1_sizes"]
            mlp2_sizes = self.conf["model"][self.model_type]["mlp2_sizes"]
            mlp3_sizes = self.conf["model"][self.model_type]["mlp3_sizes"]
            self.model = PointNet(input_dim, out_size, mlp1_sizes, mlp2_sizes, mlp3_sizes, self.window_size)
        elif self.model_type == 'transformer':
            input_dim = self.conf["model"][self.model_type]["input_dim"]
            out_size = self.conf["model"][self.model_type]["out_size"]
            mlp_sizes = self.conf["model"][self.model_type]["mlp_sizes"]
            num_heads = self.conf["model"][self.model_type]["num_heads"]
            self.model = PointTransformer(input_dim, out_size, mlp1_sizes, num_heads)

        self.trained_model = self.model
        checkpoint = torch.load(model_path)
        self.trained_model.load_state_dict(checkpoint)
        print("---Finished loading trained model---")


    def infer(self):
        print("---Starting inference---")
        self.trained_model.eval()
        #test_dataloader = DataLoader(self.data_model, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)

        #device = "cpu"
        self.trained_model.to(self.device)

        frame_results = {}
        frame_results['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False
        }


        window_half = int((self.window_size-1)/2)

        frame_results['results'] = {}
        for foi_index in tqdm.tqdm(range(self.data_model.max_track_length)):
            frame_token = self.frame_tokens[foi_index]
            frame_results['results'][frame_token] = []

            #inferer_data = WindowInferer(self.data_model, foi_index, -window_half, window_half)
            inferer_data = SlidingWindowInferer(self.data_model, foi_index, self.window_size)

            #infer_dataloader = DataLoader(inferer_data, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)
            #refined_boxes = []
            track_ids = {}
            for i, x in enumerate(inferer_data):

                #if x == torch.tensor([]):
                #    continue
                track = x.to(self.device)
                track_object = inferer_data.get(i)

                refined_track = self.trained_model.forward(track.unsqueeze(0)).squeeze()

                for transformation in reversed(self.transformations):
                    if type(transformation) == CenterOffset:
                        offset = torch.tensor(track_object.offset, dtype=torch.float32).to(self.device)
                        transformation.set_offset(offset)
                        transformation.set_start_and_end_index(0, -1)
                    elif type(transformation) == YawOffset:
                        offset = torch.tensor(track_object.offset, dtype=torch.float32).to(self.device)
                        transformation.set_offset(offset)
                        transformation.set_start_and_end_index(0, -1)
                    if type(transformation) == Normalize:
                        transformation.set_start_and_end_index(0, -1)

                    refined_track = transformation.untransform(refined_track)

                tracking_score = float(refined_track[-1])
                if tracking_score <= 0:
                    continue


                track_id = track_object.tracking_id
                if track_id in track_ids:
                    track_ids[track_id] = torch.cat((track_ids[track_id], refined_track.unsqueeze(0)), dim=0)
                else:
                    #print("new Track id", track_id)
                    track_ids[track_id] = refined_track.unsqueeze(0)
                # if track_id in track_ids:
                #     track_ids[track_id] = torch.stack((track_ids[track_id], refined_track.unsqueeze(0)))
                # else:
                #     track_ids[track_id] = refined_track
                #track_ids[track_object.tracking_id] = track_ids.get(track_object.tracking_id, torch.tensor([])).stack(refined_track)

            #print("LENGTH", len(track_ids.keys()))
                
            for track_id, refined_tracks in track_ids.items():

            
                averaged_track = torch.mean(refined_tracks, dim=0)
                #else: 
                #    averaged_track = refined_tracks

                refined_box = {
                    'sample_token': frame_token,
                    'translation': averaged_track[:3].tolist(),
                    'size': averaged_track[3:6].tolist(),
                    'rotation': averaged_track[6:10].tolist(),
                    'velocity': [0,0],
                    'tracking_id': track_id,
                    'tracking_name': 'Vehicle',
                    'tracking_score': float(averaged_track[-1]),
                }


                frame_results['results'][frame_token].append(refined_box)

                #refined_boxes.append(refined_box)

            #frame_results['results'][frame_token].append(refined_boxes)
        
        save_path = os.path.join(self.save_dir, 'results_smoothing.json')
        mmcv.dump(frame_results, save_path)

             







                
