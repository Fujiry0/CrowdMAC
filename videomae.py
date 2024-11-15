import random

import joblib
import numpy as np
import torch
from PIL import Image

from utils_videomae import (
    create_template,
    generate_heatmap_from_templates,
    get_trajectories,
)


def randomshift(traj, img_max_size, shift_limit=0.0625):
    shift_x = int(img_max_size * random.uniform(0, shift_limit))
    shift_y = int(img_max_size * random.uniform(0, shift_limit))
    if np.random.uniform() < 0.5:
        shift_x = -shift_x
    if np.random.uniform() < 0.5:
        shift_x = -shift_y
    traj[:, 2] += shift_x
    traj[:, 3] += shift_y
    return traj


def randomscale(traj, scale_limit=0.1):
    resize = random.uniform(0, scale_limit)
    if np.random.uniform() < 0.5:
        resize = -resize
    resize = 1 - resize
    traj[:, 2] *= resize
    traj[:, 3] *= resize
    return traj


def randomrotate(traj, img_max_size):
    degree = np.random.rand(1)[0] * 360
    angle = np.deg2rad(degree)

    rotation_matrix = np.zeros((2, 2))
    rotation_matrix[0, 0] = np.cos(angle)
    rotation_matrix[0, 1] = -np.sin(angle)
    rotation_matrix[1, 0] = np.sin(angle)
    rotation_matrix[1, 1] = np.cos(angle)

    center = np.array([img_max_size // 2, img_max_size // 2])
    traj[:, 2:] -= center
    traj[:, 2:] = np.dot(traj[:, 2:], rotation_matrix)
    traj[:, 2:] += center
    return traj


class VideoMAE(torch.utils.data.Dataset):
    def __init__(
        self,
        root="data",
        datasets=["stanford"],
        input_size=80,
        sigma=3,
        transform=None,
        template_size=501,
        normalize=False,
        obs_frames=8,
        pred_frames=12,
        split="train",
        obs_mask_ratio=0.4,
        obs_labmbda=5,
        obs_mask_type="none",
        mask_type="forecastfuture",
        shift_p=0.5,
        shift_limit=0.0625,
        scale_p=0.5,
        scale_limit=0.1,
        method="sum",
    ):
        super(VideoMAE, self).__init__()
        (
            self.input_size,
            self.sigma,
            self.transform,
            self.normalize,
            self.obs_frames,
            self.pred_frames,
            self.split,
            self.obs_mask_ratio,
            self.obs_labmbda,
            self.obs_mask_type,
            self.mask_type,
            self.shift_p,
            self.shift_limit,
            self.scale_p,
            self.scale_limit,
            self.method,
        ) = (
            input_size,
            sigma,
            transform,
            normalize,
            obs_frames,
            pred_frames,
            split,
            obs_mask_ratio,
            obs_labmbda,
            obs_mask_type,
            mask_type,
            shift_p,
            shift_limit,
            scale_p,
            scale_limit,
            method,
        )
        self.trajectories = []
        for dataset in datasets:
            self.trajectories.extend(
                get_trajectories(
                    root,
                    dataset,
                    split,
                    seq_len=self.obs_frames + self.pred_frames,
                    obs_frames=self.obs_frames,
                )
            )
        np.random.shuffle(self.trajectories)

        self.templates = create_template(template_size, sigma)

        assert self.split in ["train", "test"]

    def __getitem__(self, index):
        trajectory = self.trajectories[index]
        dataset, traj, _, img_size, frames = (
            trajectory["dataset"],
            trajectory["trajectory"].copy(),
            trajectory["dir_name"],
            trajectory["img_size"],
            trajectory["frames"],
        )  # traj: trackId, frame, x, y
        img_max_size = max(img_size[0], img_size[1])
        if dataset == "crowdflow":
            data_paths = traj
            gaussmaps = [
                Image.fromarray(np.load(data_path)) for data_path in data_paths
            ]
            seq_len = len(gaussmaps)
        else:
            seq_len = len(frames)
            if self.split == "train":
                traj = randomrotate(traj, img_max_size)
                if self.shift_p > 0 and np.random.uniform() < self.shift_p:
                    traj = randomscale(traj, self.scale_limit)
                if self.scale_p > 0 and np.random.uniform() < self.scale_p:
                    traj = randomshift(traj, img_max_size, self.shift_limit)

            gaussmaps = joblib.Parallel(n_jobs=-1, verbose=0)(
                joblib.delayed(generate_heatmap_from_templates)(
                    traj=traj[traj[:, 1] == int(frame)],
                    fmap_size=self.input_size,
                    img_max_size=img_max_size,
                    templates=self.templates,
                    normalize=self.normalize,
                    method=self.method,
                )
                for _, frame in enumerate(frames)
            )

        if self.split == "train":
            process_data, mask = self.transform(
                (gaussmaps, None),
                self.obs_frames,
                self.pred_frames,
                self.obs_mask_ratio,
                self.obs_labmbda,
                self.obs_mask_type,
                self.mask_type,
            )  # T*C,H,W
        elif self.split == "test":
            process_data, mask = self.transform((gaussmaps, None))

        process_data = process_data.view(
            (seq_len, 1) + process_data.size()[-2:]
        ).transpose(
            0, 1
        )  # T*C,H,W -> T,C,H,W -> C,T,H,W

        return (process_data, mask)

    def __len__(self):
        return len(self.trajectories)


if __name__ == "__main__":
    import time

    from omegaconf import OmegaConf

    from datasets import TransformTest, TransformTrain

    cfg = OmegaConf.create(
        {
            "model": {
                "input_size": 80,
                "patch_size": 8,
                "tublet_size": 4,
            },
            "training": {"sampling": True, "temperature": 500},
            "augmentation": {
                "train": {
                    "RandomHorizontalFlip": {"apply": True, "p": 0.5},
                    "RandomVerticalFlip": {"apply": True, "p": 0.5},
                    "Randomscale": {"scale_limit": 1, "p": 0.5},
                    "Randomshift": {"shift_limit": 1, "p": 0.5},
                },
            },
            "dataset": {"num_frames": 20},
            "forecast": {"obs_frames": 8, "pred_frames": 12},
        }
    )

    transform = TransformTrain(cfg)
    start_time = time.time()
    dataset = VideoMAE(
        datasets=["stanford", "eth"],
        transform=transform,
        split="train",
        mask_type="forecastfuture",
        obs_labmbda=5,
    )
    print(f"train init: {time.time()-start_time}s")
    print(len(dataset))
    start_time = time.time()
    process_data, mask = dataset[0]
    print(f"train: {time.time()-start_time}s")
    print(process_data.shape)  # C,T,H,W
    print(sum(mask) / len(mask))
    process_data, mask = dataset[0]
    print(sum(mask) / len(mask))

    transform = TransformTest(cfg)
    start_time = time.time()
    dataset = VideoMAE(
        datasets=["stanford"],
        transform=transform,
        split="test",
    )
    print(f"test init: {time.time()-start_time}s")
    print(len(dataset))
    start_time = time.time()
    process_data, mask = dataset[0]
    print(f"test: {time.time()-start_time}s")
    print(process_data.shape)  # C,T,H,W
    print(sum(mask) / len(mask))
    process_data, mask = dataset[0]
    print(f"test: {time.time()-start_time}s")
    print(process_data.shape)  # C,T,H,W
    print(sum(mask) / len(mask))
