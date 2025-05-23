import matplotlib

matplotlib.use("Agg")

import numpy as np
import wandb
import os


def configure_loggings(run_name, out_dir, conf):
    out_dir_path = os.path.join(out_dir, "wandb")
    run = wandb.init(
        name=run_name, project="AutoAnnSmoothing-train", config=conf, dir=out_dir_path
    )
    log_out = os.path.join(out_dir_path, "train_log.txt")
    log(log_out, conf)
    return log_out, run


def log(log_out, write_str):
    with open(log_out, "a") as f:
        f.write(str(write_str) + "\n")
    print(write_str)


def print_batch_stats(
    log_out, epoch, cur_batch, num_batches, losses, metrics, type_id="TRAIN"
):
    log(log_out, "-------------------------")
    log(
        log_out, "[Epoch %d: Batch %d/%d] %s" % (epoch, cur_batch, num_batches, type_id)
    )
    log(log_out, "             %s: %f" % ("epoch_loss", losses))


def print_epoch_stats(log_out, epoch, losses, metrics, type_id="TRAIN"):
    log(log_out, "==================================================================")
    log(log_out, "[Epoch %d: Summary] %s" % (epoch, type_id))
    for k, loss in losses.items():
        log(log_out, "             %s: %f" % (k, loss))
    for k, metric in metrics.items():
        log(log_out, "             %s: %f" % (k, metric))
    log(log_out, "==================================================================")


def log_batch_stats(losses, metrics, epoch, batch_idx, num_batches, mode, log_out):
    losses_formatted, metrics_formatted = (
        losses,
        metrics,
    )
    if mode == "train":
        wandb.log({**losses_formatted, **metrics_formatted})


def log_epoch_stats(losses, metrics, epoch, mode, log_out):
    losses_formatted, metrics_formatted = format_stats(losses, metrics)
    wandb.log({**losses_formatted, **metrics_formatted})
    print_epoch_stats(log_out, epoch, losses_formatted, metrics_formatted, mode)


def format_stats(losses, metrics):
    losses_formatted = losses
    metrics_formatted = metrics

    return losses_formatted, metrics_formatted
