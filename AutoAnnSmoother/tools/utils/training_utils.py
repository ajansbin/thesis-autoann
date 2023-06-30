import torch
from smoother.io.logging_utils import log_epoch_stats
from smoother.models.box_refinement_loss import BoxRefinementLoss


class TrainingUtils:
    def __init__(self, conf, log_out: str, scheduler):
        self.conf = conf
        self.log_out = log_out
        self.scheduler = scheduler
        self.center_dim = 3
        self.size_dim = 3
        self.rotation_dim = 1
        self.score_dim = 1
        self.loss_params = conf["loss"]
        self.eval_every = conf["train"]["eval_every"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loss_conf = self.conf["loss"]
        normalize_conf = self.conf["data"]["transformations"]["normalize"]
        self.brl = BoxRefinementLoss(loss_conf, normalize_conf, self.device)
        self.loss_fn = self.brl.loss

    def training_loop(self, model, optimizer, n_epochs, train_loader, val_loader):
        print("Training started!")

        model.to(self.device)
        train_losses, val_losses = [], []

        for epoch in range(1, n_epochs + 1):
            print("Epoch nr", epoch)
            model, train_loss = self._train_epoch(model, optimizer, train_loader, epoch)

            print("Epoch training finished! Starting validation")
            val_loss, metrics = self._validate(model, val_loader, epoch)

            log_train_loss = round(sum(train_loss) / len(train_loader), 3)
            log_val_loss = round(val_loss / len(val_loader), 3)

            print(
                f"Epoch {epoch}/{n_epochs}: "
                f"Train loss: {log_train_loss}, "
                f"Val. loss: {log_val_loss}, "
            )
            train_losses.extend(train_loss)
            val_losses.append(val_loss)

            losses = {"train_loss": log_train_loss, "val_loss": log_val_loss}
            log_epoch_stats(losses, metrics, epoch, mode="TRAIN", log_out=self.log_out)
            if epoch % 5 == 0:
                self.scheduler.step()
        return model, train_losses, val_losses

    def _train_epoch(self, model, optimizer, train_loader, epoch):
        model.train()
        train_loss_batches = []
        num_batches = len(train_loader)

        for batch_index, (x1, x2, y) in enumerate(train_loader, 1):
            tracks, points, gt_anns = (
                x1.to(self.device),
                x2.to(self.device),
                y.to(self.device),
            )
            """
            tracks  (B,W,8)
            points  (B,W,N,4)
            gt_anns (B,9)
            
            """
            batch_size, window_size, _ = tracks.shape

            optimizer.zero_grad()

            center_out, size_out, rotation_out, score_out = model.forward(
                tracks, points
            )

            # Take out indices that corresponds to foi box
            foi_indexes = gt_anns[:, -1].long()

            batch_range = torch.arange(batch_size)

            # Take out corresponding foi_box from tracks, centers, rotations and score
            foi_tracks = tracks[batch_range, foi_indexes]  # (B,8)
            foi_centers = center_out[batch_range, foi_indexes]  # (B,3)
            foi_rotations = rotation_out[batch_range, foi_indexes]  # (B,1)
            foi_score = score_out[batch_range, foi_indexes]  # (B,1)

            # Add output to original track
            c_hat = foi_tracks[:, 0:3] + foi_centers
            s_hat = foi_tracks[:, 3:6] + size_out
            r_hat = foi_tracks[:, 6:7] + foi_rotations

            loss = self.loss_fn(
                c_hat,
                s_hat,
                r_hat,
                foi_score,
                gt_anns.float(),
            )

            loss.backward()
            optimizer.step()
            train_loss_batches.append(loss.item())

        return model, train_loss_batches

    def _validate(self, model, val_loader, epoch):
        val_loss_cum = 0
        val_acc_cum = 0
        model.eval()
        val_metrics = {}
        total_non_zero = 0
        score_metrics = {}

        with torch.no_grad():
            for batch_index, (x1, x2, y) in enumerate(val_loader, 1):
                tracks, points, gt_anns = (
                    x1.to(self.device),
                    x2.to(self.device),
                    y.to(self.device),
                )

                center_out, size_out, rotation_out, score_out = model.forward(
                    tracks, points
                )

                foi_indexes = gt_anns[:, -1].long()
                foi_tracks = tracks[torch.arange(tracks.size(0)), foi_indexes]
                foi_centers = center_out[torch.arange(center_out.size(0)), foi_indexes]
                foi_rotations = rotation_out[
                    torch.arange(rotation_out.size(0)), foi_indexes
                ]
                foi_score = score_out[torch.arange(score_out.size(0)), foi_indexes]

                c_hat = foi_tracks[:, 0:3] + foi_centers
                s_hat = foi_tracks[:, 3:6] + size_out
                r_hat = foi_tracks[:, 6:7] + foi_rotations

                loss = self.loss_fn(
                    c_hat,
                    s_hat,
                    r_hat,
                    foi_score,
                    gt_anns.float(),
                )
                val_loss_cum += loss.item()
                if epoch % self.eval_every == 0:
                    # Remove Temporal encoding
                    foi_dets = tracks[torch.arange(tracks.shape[0]), foi_indexes, :-1]

                    out_size = (
                        self.center_dim
                        + self.size_dim
                        + self.rotation_dim
                        + self.score_dim
                    )
                    box_out = torch.cat((c_hat, s_hat, r_hat, foi_score), dim=-1)
                    metrics, scores, n_non_zero = self.brl.evaluate_model(
                        foi_dets.view(-1, out_size - 1),
                        box_out.view(-1, out_size),
                        gt_anns.float(),
                        self.device,
                    )
                    total_non_zero += n_non_zero
                    for metric, sos in metrics.items():
                        val_metrics[metric] = val_metrics.get(metric, 0) + sos.sum()
                    for metric, score in scores.items():
                        score_metrics[metric] = score_metrics.get(metric, 0) + score

        if epoch % self.eval_every == 0:
            for metric, sos in val_metrics.items():
                val_metrics[metric] = sos / total_non_zero
            for metric, score in score_metrics.items():
                val_metrics[metric] = score / len(val_loader.dataset)

        return val_loss_cum, val_metrics
