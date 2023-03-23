from smoother.io.config_utils import load_config
from tools.utils.test_utils import Evaluator
from smoother.data.common.tracking_data import SlidingWindowTrackingData, WindowTrackingData
from smoother.data.common.transformations import ToTensor, CenterOffset, Normalize
from smoother.models.pointnet import PointNet
from smoother.models.transformer import PointTransformer

import torch
from torch import optim
from torch.utils.data import random_split, DataLoader

from tools.utils.evaluation import giou3d


class SmoothingTester():

    def __init__(self, tracking_results_path, conf_path, data_path, save_dir):
        print("---Initializing SmoothingTester class---")
        self.tracking_results_path = tracking_results_path
        self.conf_path = conf_path
        self.data_path = data_path
        self.save_dir = save_dir

        self.conf = self._get_config(self.conf_path)
        self.data_type = self.conf["data"]["type"] # nuscenes / zod
        self.data_version = self.conf["test"]["data"]["version"]
        self.split = self.conf["test"]["data"]["split"]

        self.foi_index = self.conf["data"]["foi_index"] # index in sequence where annotation exist
        self.window_size = self.conf["data"]["window_size"]
        self.sliding_window = self.conf["data"]["sliding_window"]
        
        self.model_type = self.conf["model"]["type"]

        #self.lr = self.conf["train"]["learning_rate"]
        #self.wd = self.conf["train"]["weight_decay"]
        #self.n_epochs = self.conf["train"]["n_epochs"]
        #self.seed = self.conf["train"]["seed"]
        #self.train_size = self.conf["train"]["train_size"]
        self.batch_size = self.conf["train"]["batch_size"]
        self.n_workers = self.conf["train"]["n_workers"]

        self.tracking_results = None
        self.data_model = None
        self.model = None
        self.trained_model = None
        self.smoothing_result = []
        self.gt = []


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
        
        transformations = self._add_transformations(self.conf["data"]["transformations"])
        # Load data model
        start_ind = int(-(self.window_size-1)/2)
        end_ind = int((self.window_size-1)/2)
        self.data_model = WindowTrackingData(self.tracking_results,start_ind, end_ind, transformations)

    def _add_transformations(self, transformations_dict):
        transformations = [ToTensor()]

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
            transformations.append(Normalize(means,stdev))
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


    def test(self):
        print("---Running evaluation---")
        self.trained_model.eval()
        #test_dataloader = DataLoader(self.data_model, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)
        #device = torch.device("cuda" if torch.cuda.is_available() 
        #                                else "cpu")
        device = "cpu"
        self.trained_model.to(device)

        #out_size = self.conf["model"][self.model_type]["out_size"]
        #evaluator = Evaluator(self.conf, self.model_type)
        gious = []
        gious_refined = []
        count_false_pos = 0
        count_true_pos = 0
        improved = 0
        worse = 0
        for i, (x, y) in enumerate(self.data_model):
            tracks, gt = x.to(device), y.to(device)

            track = self.data_model.get(i)
            track_start_index = track.starting_frame_index
            track_end_index = track.starting_frame_index + len(track)-1

            has_gt = track.has_gt
            foi_box = track.get_foi_box()
            temporal_encoding = 0
            det = foi_box.center + foi_box.size + foi_box.rotation + [temporal_encoding]

            #extracting foi after refinement
            refined_det = self.trained_model(tracks.unsqueeze(0))
            refined_det = refined_det.squeeze()

            
            transformations = self.data_model.tracking_data.transformations

            for transformation in transformations:
                if type(transformation) == ToTensor:
                    det = transformation.transform(det).to(device)
            
            for transformation in reversed(transformations):
                if type(transformation) == CenterOffset:
                    transformation.set_offset(track.offset)
                    transformation.set_start_and_end_index(track_start_index, track_end_index)
                if type(transformation) == Normalize:
                    transformation.set_start_and_end_index(track_start_index, track_end_index)
                gt = transformation.untransform(gt)
                refined_det = transformation.untransform(refined_det)

            #Calculate GIoU for boxes which has an associated GT
            if has_gt:
                giou = giou3d(gt.tolist(), det.tolist())
                gious.append(giou)

                giou_r = giou3d(gt.tolist(), refined_det.tolist())
                gious_refined.append(giou_r)

                if giou_r < giou:
                    worse += 1
                else:
                    improved += 1
               
                count_true_pos += 1
            else:
                count_false_pos += 1
    
        print('giou', sum(gious)/len(gious))
        print('giou refined', sum(gious_refined)/len(gious_refined))
        print('FP', count_false_pos)
        print('TP', count_true_pos)
        print('improved', improved)
        print('worse', worse)
