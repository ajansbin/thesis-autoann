from smoother.io.load_config import load_config
from tools.utils.training_utils import Trainer
from smoother.data.common.sequence_data import SlidingWindowTracksData, WindowTracksData
from smoother.models.pointnet import WaymoPointNet
import torch
from torch import optim
from torch.utils.data import random_split, DataLoader


class SmoothingTrainer():

    def __init__(self, tracking_results_path, conf_path, data_path, save_dir):
        print("---Initializing SmoothingTrainer class---")
        self.tracking_results_path = tracking_results_path
        self.conf_path = conf_path
        self.data_path = data_path
        self.save_dir = save_dir

        self.conf = self._get_config(self.conf_path)
        self.data_type = self.conf["data"]["type"] # nuscenes / zod
        self.data_version = self.conf["data"]["version"]
        self.split = self.conf["data"]["split"]
        self.foi_index = self.conf["data"]["foi_index"] # index in sequence where annotation exist
        self.window_size = self.conf["data"]["window_size"]
        self.sliding_window = self.conf["data"]["sliding_window"]

        self.model_type = self.conf["model"]["type"]

        self.lr = self.conf["train"]["learning_rate"]
        self.wd = self.conf["train"]["weight_decay"]
        self.n_epochs = self.conf["train"]["n_epochs"]
        self.seed = self.conf["train"]["seed"]
        self.train_size = self.conf["train"]["train_size"]
        self.batch_size = self.conf["train"]["batch_size"]
        self.n_workers = self.conf["train"]["n_workers"]

        self.tracking_results = None
        self.data_model = None
        self.model = None
        self.trained_model = None
        self.result_dict = {}


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
            raise NotImplementedError(f"Dataclass of type {self.data_type} is not implemented. Please use 'nuscenes'")
        else:
            raise NotImplementedError(f"Dataclass of type {self.data_type} is not implemented. Please use 'nuscenes' or 'zod'.")
        
        # Load data model
        if self.sliding_window:
            self.data_model = SlidingWindowTracksData(self.tracking_results,self.window_size, self.foi_index)
        else:
            start_ind = self.foi_index - (self.window_size-1)/2
            end_ind = self.foi_index + (self.window_size-1)/2
            window = (start_ind, end_ind)
            self.data_model = WindowTracksData(self.tracking_results,window)


    def load_model(self):
        print("---Loading model---")
        if self.model_type == 'pointnet':
            input_dim = self.conf["model"][self.model_type]["input_dim"]
            out_size = self.conf["model"][self.model_type]["out_size"]
            mlp1_sizes = self.conf["model"][self.model_type]["mlp1_sizes"]
            mlp2_sizes = self.conf["model"][self.model_type]["mlp2_sizes"]
            self.model = WaymoPointNet(input_dim, out_size, mlp1_sizes, mlp2_sizes, self.window_size)


    def train(self):
        print("---Starting training---")
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,  weight_decay=self.wd)
        trainer = Trainer(self.conf, self.model_type)
        loss_fn = trainer.box_regression_loss

        torch.manual_seed(self.seed)
        size = len(self.data_model)
        n_train = int(self.train_size*size)
        n_val = size - n_train

        train_dataset, val_dataset = random_split(self.data_model, [n_train, n_val])

        batch_size = self.batch_size

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)

        self.trained_model, train_losses, val_losses = trainer.training_loop(self.model,optimizer,loss_fn,self.n_epochs, train_dataloader, val_dataloader)
        self.result_dict = {"train_losses": train_losses, "val_losses":val_losses}
        print("---Finished training---")



    def save_results(self):
        import os
        import json
        # Save to file
        save_dir_full = os.path.join(self.save_dir, self.model_type)
        if not os.path.exists(save_dir_full):
            os.mkdir(save_dir_full)
        
        # save model
        model_path = os.path.join(save_dir_full, "model.pth")
        torch.save(self.trained_model.state_dict(), model_path)

        # save losses
        losses_path = os.path.join(save_dir_full, "losses.sjon")
        json_object = json.dumps(self.result_dict)
        with open(losses_path, "w") as outfile:
            outfile.write(json_object)