import matplotlib
matplotlib.use('Agg')

import numpy as np
import wandb
import os

def configure_loggings(run_name, out_dir, conf):
    out_dir_path = os.path.join(out_dir, "wandb")
    wandb.init(name=run_name, project="AutoAnnSmoothing-train", config=conf, dir=out_dir_path)
    log_out = os.path.join(out_dir_path, 'train_log.txt')
    log(log_out, conf)
    return log_out


def log(log_out, write_str):
    with open(log_out, 'a') as f:
        f.write(str(write_str) + '\n')
    print(write_str)

def print_batch_stats(log_out, epoch, cur_batch, num_batches, losses, metrics, type_id='TRAIN'):
    log(log_out, '-------------------------')
    log(log_out, '[Epoch %d: Batch %d/%d] %s' % (epoch, cur_batch, num_batches, type_id))
    #for k, loss in losses.items():
    log(log_out, '             %s: %f' % ("epoch_loss", losses))
    #for k, metric in metrics.items():
    #    log(log_out, '             %s: %f' % (k, metric))


def print_epoch_stats(log_out, epoch, losses, metrics, type_id='TRAIN'):
    log(log_out, '==================================================================')
    log(log_out, '[Epoch %d: Summary] %s' % (epoch, type_id))
    for k, loss in losses.items():
        log(log_out, '             %s: %f' % (k, loss))
    for k, metric in metrics.items():
        log(log_out, '             %s: %f' % (k, metric))
    log(log_out, '==================================================================')


def log_batch_stats(losses, metrics, epoch, batch_idx, num_batches, mode, log_out):
    losses_formatted, metrics_formatted = losses, metrics #format_stats(losses, metrics, mode)
    #print_batch_stats(log_out, epoch, batch_idx, num_batches, losses_formatted, metrics_formatted, mode)
    if mode == "train":
        wandb.log({**losses_formatted, **metrics_formatted})


def log_epoch_stats(losses, metrics, epoch, mode, log_out):
    losses_formatted, metrics_formatted = format_stats(losses, metrics)
    wandb.log({**losses_formatted, **metrics_formatted})
    print_epoch_stats(log_out, epoch, losses_formatted, metrics_formatted, mode)

def format_stats(losses, metrics):

    losses_formatted = losses
    metrics_formatted = metrics

    # mae_center = [metrics["mae_dets_center"], metrics["mae_refinement_center"]]
    # mae_size = [metrics["mae_dets_size"], metrics["mae_refinement_size"]]
    # mae_rotation = [metrics["mae_dets_rotation"], metrics["mae_refinement_rotation"]]

    # mse_center = [metrics["mse_dets_center"], metrics["mse_refinement_center"]]
    # mse_size = [metrics["mse_dets_size"], metrics["mse_refinement_size"]]
    # mse_rotation = [metrics["mse_dets_rotation"], metrics["mse_refinement_rotation"]]

    # metrics_formatted = {
    #     "mae_center": mae_center,
    #     "mae_size": mae_size,
    #     "mae_rotation": mae_rotation,
    #     "mse_center":mse_center,
    #     "mse_size":mse_size,
    #     "mse_rotation":mse_rotation
    # }

    return losses_formatted, metrics_formatted




