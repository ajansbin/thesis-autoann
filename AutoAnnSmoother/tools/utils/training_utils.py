import torch
from torch import nn
import numpy as np

class Trainer():

    def __init__(self, conf, model_type:str):
        self.conf = conf
        self.out_size = conf["model"][model_type]["out_size"]
        self.loss_params = conf["loss"]


    def training_loop(self, model, optimizer, loss_fn, n_epochs, train_loader, val_loader):
        print("Training started!")
        device = torch.device("cuda" if torch.cuda.is_available() 
                                        else "cpu")
        #device="cpu"
        model.to(device)
        train_losses, val_losses = [], []

        for epoch in range(1,n_epochs+1):
            print("Epoch nr", epoch)
            model, train_loss = self._train_epoch(model,
                                            optimizer,
                                            loss_fn,
                                            train_loader,
                                            val_loader,
                                            device)
            print("Epoch training finished! Starting validation")
            val_loss = self._validate(model, loss_fn, val_loader, device)
            print(f"Epoch {epoch}/{n_epochs}: "
                f"Train loss: {sum(train_loss)/len(train_loss):.3f}, "
                f"Val. loss: {val_loss:.3f}, ")
            train_losses.extend(train_loss)
            val_losses.append(val_loss)
        return model, train_losses, val_losses

    def _train_epoch(self, model, optimizer, loss_fn, train_loader, val_loader, device):
        model.train()
        train_loss_batches = []
        num_batches = len(train_loader)
        
        for batch_index, (x, y) in enumerate(train_loader, 1):
            #if batch_index % 40 == 0 or batch_index==1:
            #    print("Batch index", batch_index)
            tracks, gt_anns = x.to(device), y.to(device).long()
            optimizer.zero_grad()
            model_output = model.forward(tracks)

            loss = loss_fn(model_output.view(-1, self.out_size) , gt_anns.float())

            loss.backward()
            optimizer.step()
            train_loss_batches.append(loss.item())
            
        return model, train_loss_batches

    def _validate(self, model, loss_fn, val_loader, device):
        val_loss_cum = 0
        val_acc_cum = 0
        model.eval()
        with torch.no_grad():
            for batch_index, (x, y) in enumerate(val_loader, 1):
                tracks, gt_anns = x.to(device), y.to(device)
                model_output = model.forward(tracks)
                loss = loss_fn(model_output.view(-1, self.out_size) , gt_anns.float())
                val_loss_cum += loss.item()

        return val_loss_cum/len(val_loader)
    
    def compute_score_loss(self, predictions, gts):
        scores = predictions[:,10]
        gt_score = gts[:,10]
        m = nn.Sigmoid()
        loss = nn.BCELoss()
        out = loss(m(scores),gt_score)
        return out



    def compute_loss(self, predictions, gts):
        #print("Loss", predictions.shape, gts.shape)
        centers = predictions[:,:3]
        gt_centers = gts[:,:3]
        center_loss = torch.linalg.norm(gt_centers-centers)

        sizes = predictions[:,3:6]
        gt_sizes = gts[:,3:6]
        size_loss = torch.linalg.norm(gt_sizes-sizes)

        rotation = predictions[:,6:10]
        gt_rotation = gts[:,6:10]
        rotation_loss = torch.linalg.norm(gt_rotation-rotation)

        return center_loss, size_loss, rotation_loss

    def box_regression_loss(self, predictions, gt):
        center_loss, size_loss, rotation_loss = self.compute_loss(predictions, gt)
        score_loss = self.compute_score_loss(predictions,gt)
        center_loss *= self.loss_params["weight"]["center"]
        size_loss *= self.loss_params["weight"]["size"]
        rotation_loss *= self.loss_params["weight"]["rotation"]
        score_loss *= self.loss_params["weight"]["score"]
        #print(center_loss, size_loss, rotation_loss, score_loss)
        return center_loss + size_loss + rotation_loss + score_loss