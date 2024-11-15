import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
from einops import rearrange

import utils
from metrics import compute_eval_metrics, generate_predicted_videos
from utils import AverageMeter


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    patch_size: int = 16,
    lr_scheduler=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    tublet_size=2,
    den_factor: int = 100,
    wandb=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    loss_avg = AverageMeter()
    mask_type = data_loader.dataset.mask_type
    for step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = (
            bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        )

        with torch.no_grad():
            videos_patch = rearrange(
                videos,
                "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
                p0=tublet_size,  # tubelet size
                p1=patch_size,
                p2=patch_size,
            )

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs = model(videos, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels * den_factor)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

        # For epoch loss
        loss_avg.update(loss.item(), len(videos))

    if wandb is not None:
        wandb.log(
            {
                "train/loss": loss_avg.avg,
                f"train/{mask_type}": loss_avg.avg,
            },
            step=epoch,
        )

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    patch_size: int = 16,
    tublet_size: int = 2,
    batch_size: int = 512,
    num_frames: int = 20,
    input_size: int = 80,
    obs_frames: int = 8,
    den_factor: int = 100,
    wandb=None,
    eval_stats={},
):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    loss_avg = AverageMeter()
    adkl_avg = AverageMeter()
    adrkl_avg = AverageMeter()
    adjs_avg = AverageMeter()
    fdkl_avg = AverageMeter()
    fdrkl_avg = AverageMeter()
    fdjs_avg = AverageMeter()
    for _, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = (
            bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        )

        with torch.no_grad():
            videos_patch = rearrange(
                videos,
                "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
                p0=tublet_size,  # tubelet size
                p1=patch_size,
                p2=patch_size,
            )

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

            with torch.cuda.amp.autocast():
                outputs = model(videos, bool_masked_pos)
                loss = loss_func(input=outputs, target=labels * den_factor)

            predicted_videos = generate_predicted_videos(
                outputs,
                videos_patch,
                bool_masked_pos,
                B,
                input_size,
                patch_size,
                tublet_size,
                num_frames,
            )

            adkl, adrkl, adjs, fdkl, fdrkl, fdjs = compute_eval_metrics(
                videos, predicted_videos / den_factor, obs_frames
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        torch.cuda.synchronize()

        metric_logger.update(test_loss=loss_value)
        # For epoch loss
        loss_avg.update(loss.item(), len(videos))
        adkl_avg.update(adkl.item(), len(videos))
        adrkl_avg.update(adrkl.item(), len(videos))
        adjs_avg.update(adjs.item(), len(videos))
        fdkl_avg.update(fdkl.item(), len(videos))
        fdrkl_avg.update(fdrkl.item(), len(videos))
        fdjs_avg.update(fdjs.item(), len(videos))

    min_updated = False
    if epoch == 0 or eval_stats["adjs_min"] > adjs_avg.avg:
        eval_stats["test_loss_min"] = loss_avg.avg
        eval_stats["adkl_min"] = adkl_avg.avg
        eval_stats["arkl_min"] = adrkl_avg.avg
        eval_stats["adjs_min"] = adjs_avg.avg
        eval_stats["fdkl_min"] = fdkl_avg.avg
        eval_stats["fdrkl_min"] = fdrkl_avg.avg
        eval_stats["fdjs_min"] = fdjs_avg.avg
        min_updated = True

    if wandb is not None:
        wandb.log(
            {
                "test/loss": loss_avg.avg,
                "test/adkl": adkl_avg.avg,
                "test/adrkl": adrkl_avg.avg,
                "test/adjs": adjs_avg.avg,
            },
            step=epoch,
        )
        for key, val in eval_stats.items():
            wandb.log(
                {
                    f"test_min/{key[:-4]}": val,
                },
                step=epoch,
            )

    print("Averaged stats:", metric_logger)
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        eval_stats,
        min_updated,
    )
