import torch
from torch import nn
import numpy as np
from smoother.io.logging_utils import log_batch_stats, log_epoch_stats
from smoother.models.box_refinement_loss import BoxRefinementLoss
import tqdm

class TrainingUtils():

    def __init__(self, conf, log_out:str):
        self.conf = conf
        self.log_out = log_out
        self.out_size = 8
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
            val_loss, metrics = self._validate(model, val_loader)

            log_train_loss = round(sum(train_loss)/len(train_loader),3)
            log_val_loss = round(val_loss/len(val_loader), 3)

            print(f"Epoch {epoch}/{n_epochs}: "
                f"Train loss: {log_train_loss}, "
                f"Val. loss: {log_val_loss}, ")
            train_losses.extend(train_loss)
            val_losses.append(val_loss)

            losses = {"train_loss": log_train_loss, "val_loss": log_val_loss}
            log_epoch_stats(losses, metrics, epoch, mode='TRAIN', log_out=self.log_out)
        return model, train_losses, val_losses

    def _train_epoch(self, model, optimizer, train_loader, epoch):
        model.train()
        train_loss_batches = []
        num_batches = len(train_loader)
        
        for batch_index, (x1, x2, y) in enumerate(train_loader, 1):
            tracks, points, gt_anns = x1.to(self.device), x2.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            model_output = model.forward(tracks, points)

            loss = self.loss_fn(model_output.view(-1, self.out_size) , gt_anns.float())

            loss.backward()
            optimizer.step()
            train_loss_batches.append(loss.item())

            #if batch_index % 50 == 0:
            #    log_batch_stats(loss,None,epoch, batch_index, num_batches, 'train', self.log_out)
            
        return model, train_loss_batches

    def _validate(self, model, val_loader):
        val_loss_cum = 0
        val_acc_cum = 0
        model.eval()
        val_metrics = {}
        total_samples = 0

        with torch.no_grad():
            for batch_index, (x1, x2, y) in enumerate(val_loader, 1):
                tracks, points, gt_anns = x1.to(self.device), x2.to(self.device), y.to(self.device)
                model_output = model.forward(tracks, points)
                loss = self.loss_fn(model_output.view(-1, self.out_size), gt_anns.float())
                val_loss_cum += loss.item()

                foi_indexes = self._find_foi_indexes(tracks)
                foi_dets = tracks[torch.arange(tracks.shape[0]), foi_indexes, :-1] #removes temporal encoding
                metrics, n_non_zero = self.brl.evaluate_model(foi_dets.view(-1, self.out_size), model_output.view(-1, self.out_size), gt_anns.float())
                total_samples += n_non_zero
                for metric, sos in metrics.items():
                    val_metrics[metric] = val_metrics.get(metric,0) + sos.sum()
        for metric, sos in val_metrics.items():
            val_metrics[metric] = (sos / total_samples) ** (1 / 2)

        return val_loss_cum, val_metrics
    
    def _find_foi_indexes(self, tracks):
        is_foi = tracks[:,:,-1] == 0
        foi_index = is_foi.nonzero()
        return foi_index[:,-1]
