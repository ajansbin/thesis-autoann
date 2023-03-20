import torch
from torch import nn
import numpy as np
from smoother.io.logging_utils import log_batch_stats, log_epoch_stats
from smoother.models.box_refinement_loss import BoxRefinementLoss
import tqdm

class TrainingUtils():

    def __init__(self, conf, model_type:str, log_out:str):
        self.conf = conf
        self.model_type = model_type
        self.log_out = log_out
        self.out_size = conf["model"][model_type]["out_size"]
        self.loss_params = conf["loss"]

        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                        else "cpu")

        loss_conf = self.conf["loss"]
        self.brl = BoxRefinementLoss(loss_conf)
        self.loss_fn = self.brl.loss


    def training_loop(self, model, optimizer, n_epochs, train_loader, val_loader):

        print("Training started!")

        model.to(self.device)
        train_losses, val_losses = [], []

        #for epoch in tqdm.tqdm()
        for epoch in range(1,n_epochs+1):
            print("Epoch nr", epoch)
            model, train_loss = self._train_epoch(model,
                                            optimizer,
                                            train_loader,
                                            epoch)
            print("Epoch training finished! Starting validation")
            val_loss, improvements = self._validate(model, val_loader)

            log_train_loss = round(sum(train_loss)/len(train_loader),3)
            log_val_loss = round(val_loss/len(val_loader), 3)

            print(f"Epoch {epoch}/{n_epochs}: "
                f"Train loss: {log_train_loss}, "
                f"Val. loss: {log_val_loss}, ")
            train_losses.extend(train_loss)
            val_losses.append(val_loss)

            losses = {"train_loss": log_train_loss, "val_loss": log_val_loss}
            log_epoch_stats(losses, improvements, epoch, mode='TRAIN', log_out=self.log_out)
        return model, train_losses, val_losses

    def _train_epoch(self, model, optimizer, train_loader, epoch):
        model.train()
        train_loss_batches = []
        num_batches = len(train_loader)
        
        for batch_index, (x, y) in enumerate(train_loader, 1):
            tracks, gt_anns = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            model_output = model.forward(tracks)

            loss = self.loss_fn(model_output.view(-1, self.out_size) , gt_anns.float())

            loss.backward()
            optimizer.step()
            train_loss_batches.append(loss.item())

            if batch_index % 50 == 0:
                log_batch_stats(loss,None,epoch, batch_index, num_batches, 'TRAIN', self.log_out)
            
        return model, train_loss_batches

    def _validate(self, model, val_loader):
        val_loss_cum = 0
        val_acc_cum = 0
        model.eval()
        with torch.no_grad():
            for batch_index, (x, y) in enumerate(val_loader, 1):
                tracks, gt_anns = x.to(self.device), y.to(self.device)
                model_output = model.forward(tracks)
                loss = self.loss_fn(model_output.view(-1, self.out_size) , gt_anns.float())
                val_loss_cum += loss.item()
                #foi_index = self._find_foi_index(tracks)
                #foi_dets = tracks[:,foi_index]
                #improvements = self.brl.evaluate_model(foi_dets.view(-1, self.out_size),model_output.view(-1, self.out_size),gt_anns.float())
                improvements = None

        return val_loss_cum, improvements
    
    def _find_foi_index(self, tracks):
        indices = (tracks[:,:,10] == 0).nonzero()
        rows_of_10 = indices[:, 1]
        return rows_of_10[0].item()