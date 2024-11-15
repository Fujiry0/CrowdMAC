# Standard Library
import datetime
import json
import os
import random
import time
import warnings

# Third Party Library
import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
from omegaconf import DictConfig, OmegaConf

# First Party Library
import utils
from datasets import build_test_dataset, build_train_dataset
from early_stopping import EarlyStopping
from helper import test_one_epoch, train_one_epoch
from masking_generator import choose_mask_type
from model import get_model
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import set_seed

warnings.simplefilter("ignore")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.dataset.device)
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    num_frames = cfg.dataset.num_frames
    input_size = cfg.model.input_size
    tublet_size = cfg.model.tublet_size
    obs_frames = cfg.forecast.obs_frames
    obs_mask_type = cfg.training.obs_mask.mask_type
    pred_mask_type = cfg.training.pred_mask.mask_type
    warmup_epochs = cfg.optimizer.warmup_epochs
    use_wandb = cfg.training.use_wandb
    after_pretraining = cfg.training.after_pretraining.apply
    pretraining_epochs = cfg.training.after_pretraining.epochs
    after_pretraining_obs_mask_type = cfg.training.after_pretraining.obs_mask.mask_type
    after_pretraining_pred_mask_type = (
        cfg.training.after_pretraining.pred_mask.mask_type
    )
    params_tuning = cfg.params_tuning.apply
    params_tuning_epochs = cfg.params_tuning.epochs
    obs_mask_warmup_epochs = cfg.training.obs_mask.warmup_epochs
    obs_mask_curriculum_learning = cfg.training.obs_mask.curriculum_learning
    if obs_mask_curriculum_learning:
        obs_mask_schedule = np.linspace(
            cfg.training.obs_mask.min_lambda,
            cfg.training.obs_mask.max_lambda,
            obs_mask_warmup_epochs,
        )

    assert obs_mask_type in [
        "none",
        "linear",
        "cubic",
        "square",
        "square_root",
        "sine",
        "exp",
        "exp_fix",
        "tube",
        "random",
    ]

    assert pred_mask_type in [
        "forecastfuture",
        "forecast_inv",
        "forecast_interpolate",
        "forecast_inv_interpolate",
    ]

    # fix the seed for reproducibility
    seed = cfg.training.seed
    set_seed(seed)

    cudnn.benchmark = True

    model = get_model(cfg)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))

    # get train dataset
    dataset_train = build_train_dataset(cfg)
    num_training_steps_per_epoch = len(dataset_train) // batch_size

    # get test dataset
    dataset_test = build_test_dataset(cfg)

    print(f"The dataset is {cfg.dataset.datasets}")
    print(f"The number of train tajectories is {len(dataset_train)}.")
    print(f"The number of test tajectories is {len(dataset_test)}.")

    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if use_wandb:
        import wandb

        wandb.init(
            config=config_dict,
            project=cfg.training.project,
        )
        project_name = wandb.run.project
        run_name = wandb.run.name

        output_dir = os.path.join(cfg.logger.output_dir, project_name, run_name)
        os.makedirs(output_dir, exist_ok=True)

        wandb.watch(model, log="all")
    else:
        wandb = None
        output_dir = None
    best_val_checkpoint_path = None
    earlystopping = EarlyStopping(
        patience=cfg.training.early_stopping.patience,
        verbose=True,
        path=best_val_checkpoint_path,
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker,
        shuffle=cfg.dataset.shuffle,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker,
    )

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print("number of params: {} M".format(n_parameters / 1e6))

    lr = cfg.optimizer.lr * batch_size / 256
    min_lr = cfg.optimizer.min_lr * batch_size / 256
    print("LR = %.8f" % lr)
    print("Batch size = %d" % batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print(
        "Number of training examples per epoch = %d"
        % (batch_size * num_training_steps_per_epoch)
    )

    model = torch.nn.DataParallel(model).to(device)
    if cfg.model.checkpoint:
        checkpoint = torch.load(
            cfg.model.checkpoint,
            map_location=device,
        )
        model.load_state_dict(checkpoint["model"])
    optimizer = create_optimizer(cfg, model)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        lr,
        min_lr,
        epochs,
        num_training_steps_per_epoch,
        warmup_epochs=warmup_epochs,
        warmup_steps=cfg.optimizer.warmup_steps,
    )
    weight_decay_end = cfg.optimizer.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        cfg.optimizer.weight_decay,
        weight_decay_end,
        epochs,
        num_training_steps_per_epoch,
    )
    print(
        "Max WD = %.7f, Min WD = %.7f"
        % (max(wd_schedule_values), min(wd_schedule_values))
    )

    if cfg.training.test_per:
        test_per = cfg.training.test_per
    else:
        test_per = epochs // 20

    torch.cuda.empty_cache()
    print(f"Start training for {epochs} epochs")
    start_time = time.time()
    eval_stats = {}
    for epoch in range(epochs):
        data_loader_train.dataset.mask_type = choose_mask_type(
            obs_mask_type, pred_mask_type, epoch
        )
        if after_pretraining and pretraining_epochs and epoch > pretraining_epochs:
            data_loader_train.dataset.mask_type = choose_mask_type(
                after_pretraining_obs_mask_type, after_pretraining_pred_mask_type, epoch
            )
        if obs_mask_curriculum_learning and epoch < obs_mask_warmup_epochs:
            data_loader_train.dataset.obs_labmbda = random.uniform(
                cfg.training.obs_mask.min_lambda, obs_mask_schedule[epoch]
            )
        else:
            data_loader_train.dataset.obs_labmbda = random.uniform(
                cfg.training.obs_mask.min_lambda, cfg.training.obs_mask.max_lambda
            )
        if obs_mask_type == "exp":
            data_loader_train.dataset.obs_labmbda = random.uniform(
                cfg.training.obs_mask.min_lambda, cfg.training.obs_mask.max_lambda
            )
        elif obs_mask_type == "exp_fix":
            data_loader_train.dataset.obs_labmbda = cfg.training.obs_mask.max_lambda

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            cfg.optimizer.clip_grad,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            tublet_size=tublet_size,
            den_factor=cfg.model.den_factor,
            wandb=wandb,
        )

        if epoch % test_per == 0:
            test_stats, eval_stats, min_updated = test_one_epoch(
                model,
                data_loader_test,
                device,
                epoch,
                patch_size=patch_size[0],
                tublet_size=tublet_size,
                batch_size=batch_size,
                num_frames=num_frames,
                input_size=input_size,
                obs_frames=obs_frames,
                wandb=wandb,
                eval_stats=eval_stats,
                den_factor=cfg.model.den_factor,
            )

            if min_updated:
                checkpoint_path = os.path.join(output_dir, "checkpoint-best.pth")
                to_save = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": cfg,
                }
                torch.save(to_save, checkpoint_path)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            earlystopping(eval_stats["adjs_min"], model)
            if earlystopping.early_stop:
                print("Early stopping")
                break
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if output_dir:
            with open(
                os.path.join(output_dir, "log.txt"),
                mode="a",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if params_tuning and epoch == params_tuning_epochs:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
