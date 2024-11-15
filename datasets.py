from torchvision import transforms

from masking_generator import (
    test_masked_position_generator,
    train_masked_position_generator,
)
from transforms import create_test_augmentation_list, create_train_augmentation_list
from videomae import VideoMAE


class TransformTrain(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.transform = transforms.Compose(create_train_augmentation_list(cfg))
        self.patch_size, self.tublet_size = cfg.model.patch_size, cfg.model.tublet_size
        self.window_size = (
            cfg.dataset.num_frames // cfg.model.tublet_size,  # tublet_size
            cfg.model.input_size // cfg.model.patch_size,
            cfg.model.input_size // cfg.model.patch_size,
        )
        self.tublet_size = self.cfg.model.tublet_size
        self.sampling = cfg.training.sampling
        self.temperature = cfg.training.temperature

    def __call__(
        self,
        images,
        obs_frames,
        pred_frames,
        obs_mask_ratio,
        obs_labmbda,
        obs_mask_type,
        pred_mask_type,
    ):
        process_data, mask = self.transform(images)
        mask = train_masked_position_generator(
            process_data,
            self.window_size,
            obs_frames,
            pred_frames,
            self.patch_size,
            self.tublet_size,
            obs_mask_ratio,
            obs_labmbda,
            obs_mask_type,
            pred_mask_type,
            self.sampling,
            self.temperature,
        )
        return process_data, mask

    def __repr__(self):
        repr = "(TransformTrain,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


class TransformTest(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.transform = transforms.Compose(create_test_augmentation_list(cfg))
        self.window_size = (
            cfg.dataset.num_frames // cfg.model.tublet_size,  # tublet_size
            cfg.model.input_size // cfg.model.patch_size,
            cfg.model.input_size // cfg.model.patch_size,
        )
        self.obs_frames, self.pred_frames, self.tublet_size = (
            cfg.forecast.obs_frames,
            cfg.forecast.pred_frames,
            cfg.model.tublet_size,
        )

    def __call__(self, images):
        process_data, mask = self.transform(images)
        mask = test_masked_position_generator(
            self.window_size, self.obs_frames, self.pred_frames, self.tublet_size
        )
        return process_data, mask

    def __repr__(self):
        repr = "(TransformTest,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


def build_train_dataset(cfg):
    transform = TransformTrain(cfg)

    dataset = VideoMAE(
        datasets=cfg.dataset.datasets,
        input_size=cfg.model.input_size,
        sigma=cfg.training.sigma,
        transform=transform,
        template_size=cfg.training.template_size,
        obs_frames=cfg.forecast.obs_frames,
        pred_frames=cfg.forecast.pred_frames,
        split="train",
        obs_mask_type=cfg.training.obs_mask.mask_type,
        obs_mask_ratio=cfg.training.obs_mask.mask_ratio,
        shift_p=cfg.augmentation.train.Randomshift.p,
        shift_limit=cfg.augmentation.train.Randomshift.shift_limit,
        scale_p=cfg.augmentation.train.Randomscale.p,
        scale_limit=cfg.augmentation.train.Randomscale.scale_limit,
    )

    print("Data Aug = %s" % str(transform))
    return dataset


def build_test_dataset(cfg):
    transform = TransformTest(cfg)

    dataset = VideoMAE(
        datasets=cfg.dataset.datasets,
        input_size=cfg.model.input_size,
        sigma=cfg.training.sigma,
        transform=transform,
        template_size=cfg.training.template_size,
        obs_frames=cfg.forecast.obs_frames,
        pred_frames=cfg.forecast.pred_frames,
        split="test",
        obs_mask_ratio=cfg.training.obs_mask.mask_ratio,
    )

    print("Data Aug = %s" % str(transform))
    return dataset
