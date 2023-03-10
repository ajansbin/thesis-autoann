from smoother.io.config_utils import load_config
from smoother.io.logging_utils import configure_loggings
from tools.utils.training_utils import TrainingUtils
from smoother.data.common.sequence_data import SlidingWindowTracksData, WindowTracksData
from smoother.models.pointnet import PointNet
from smoother.models.transformer import PointTransformer
from smoother.models.box_refinement_loss import BoxRefinementLoss
import torch
from torch import optim
from torch.utils.data import random_split, DataLoader


class SmoothingTrainer():

    def __init__(self, tracking_results_path, conf_path, data_path, save_dir, run_name):
        print("---Initializing SmoothingTrainer class---")
        self.tracking_results_path = tracking_results_path
        self.conf_path = conf_path
        self.data_path = data_path
        self.save_dir = save_dir
        self.run_name = run_name

        self.conf = self._get_config(self.conf_path)
        self.data_type = self.conf["data"]["type"] # nuscenes / zod
        self.data_version = self.conf["data"]["version"]
        self.split = self.conf["data"]["split"]
        self.foi_index = self.conf["data"]["foi_index"] # index in sequence where annotation exist
        self.window_size = self.conf["data"]["window_size"]
        self.sliding_window = self.conf["data"]["sliding_window"]
        
        self.normalize = self.conf["data"]["normalize"]["normalize"]

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
        self.n_train_batches = 0
        self.n_val_batches = 0
        self.trained_model = None
        self.result_dict = {}

        print("---Setting up wandb-logger---")
        self.log_out = configure_loggings(self.run_name, self.save_dir, self.conf)



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
        
        #Normalization
        if self.normalize:
            center_mean = self.conf["data"]["normalize"]["center"]["mean"]
            center_stdev = self.conf["data"]["normalize"]["center"]["stdev"]
            size_mean = self.conf["data"]["normalize"]["size"]["mean"]
            size_stdev = self.conf["data"]["normalize"]["size"]["stdev"]
            rotation_mean = self.conf["data"]["normalize"]["rotation"]["mean"]
            rotation_stdev = self.conf["data"]["normalize"]["rotation"]["stdev"]
            score_mean = self.conf["data"]["normalize"]["score"]["mean"]
            score_stdev = self.conf["data"]["normalize"]["score"]["stdev"]

            means = center_mean + size_mean + rotation_mean + score_mean
            stdev = center_stdev + size_stdev + rotation_stdev + score_stdev
        else:
            means, stdev = None, None


        # Load data model
        if self.sliding_window:
            self.data_model = SlidingWindowTracksData(self.tracking_results,self.window_size, self.foi_index, means, stdev)
        else:
            start_ind = int(self.foi_index - (self.window_size-1)/2)
            end_ind = int(self.foi_index + (self.window_size-1)/2)
            window = (start_ind, end_ind)
            self.data_model = WindowTracksData(self.tracking_results,window, means, stdev)


    def load_model(self):
        print("---Loading model---")
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
           

    def train(self):
        print("---Starting training---")

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,  weight_decay=self.wd)
        tu = TrainingUtils(self.conf, self.model_type, self.log_out)


        loss_fn = tu.brl

        torch.manual_seed(self.seed)
        size = len(self.data_model)
        n_train = int(self.train_size*size)
        n_val = size - n_train

        train_dataset, val_dataset = random_split(self.data_model, [n_train, n_val])

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)

        self.n_train_batches = len(train_dataloader)
        self.n_val_batches = len(val_dataloader)

        self.trained_model, train_losses, val_losses = tu.training_loop(self.model,optimizer,self.n_epochs, train_dataloader, val_dataloader)
        self.result_dict = {"train_losses": train_losses, "val_losses":val_losses}
        print("---Finished training---")



    def save_results(self):
        import os
        import json
        # Save to file
        save_dir_full = os.path.join(self.save_dir, self.model_type)
        if not os.path.exists(save_dir_full):
            os.mkdir(save_dir_full)

        model_name = f"{self.model_type}_{self.data_type}_{self.split}"
        
        # save model
        model_path = os.path.join(save_dir_full, model_name + "_model.pth")
        torch.save(self.trained_model.state_dict(), model_path)

        # save losses
        losses_path = os.path.join(save_dir_full, model_name + "_losses.json")
        json_object = json.dumps(self.result_dict)
        with open(losses_path, "w") as outfile:
            outfile.write(json_object)

        plot_path = os.path.join(save_dir_full, model_name + "_plot.png")
        self._plot_results(plot_path)

    def _plot_results(self, save_dir):
        import matplotlib.pyplot as plt
        import numpy as np

        def moving_average(a, n=3) :
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n
        
        train_losses = self.result_dict["train_losses"]
        val_losses = self.result_dict["val_losses"]

        window = 100
        train_loss_ma = moving_average(train_losses, window)

        batches_per_epoch = self.n_train_batches
        x = range(1,len(train_losses)-window+2)
        x_val = np.arange(1, len(val_losses)+1) * batches_per_epoch

        plt.plot(x, train_loss_ma, label="Training")
        plt.plot(x_val, val_losses, label = "Validation")
        #plt.xlim((200,20000))
        #plt.ylim((500,4000))
        plt.xlabel("Batch number")
        plt.ylabel("Loss")
        plt.legend(loc= "upper right")
        plt.savefig(save_dir)