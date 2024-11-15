import numpy as np
import torch

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def choose_mask_type(obs_mask_type, pred_mask_type, epoch):
    if pred_mask_type == "forecastfuture":
        if obs_mask_type == "none":
            return "forecastfuture"
        elif obs_mask_type == "tube":
            return "forecastfuturefrompasttube"
        else:
            return "forecastfuturefrompastwithscheduling"
    if pred_mask_type == "forecast_inv":
        if epoch % 2 == 0:
            if obs_mask_type == "none":
                return "forecastfuture"
            elif obs_mask_type == "tube":
                return "forecastfuturefrompasttube"
            else:
                return "forecastfuturefrompastwithscheduling"
        else:
            if obs_mask_type == "none":
                return "forecastpast"
            elif obs_mask_type == "tube":
                return "forecastpastfromfuturetube"
            else:
                return "forecastpastfromfuturewithscheduling"
    if pred_mask_type == "forecast_interpolate":
        if epoch % 2 == 0:
            if obs_mask_type == "none":
                return "forecastfuture"
            elif obs_mask_type == "tube":
                return "forecastfuturefrompasttube"
            else:
                return "forecastfuturefrompastwithscheduling"
        else:
            return "interpolatetube"
    if pred_mask_type == "forecast_inv_interpolate":
        if epoch % 3 == 0:
            if obs_mask_type == "none":
                return "forecastfuture"
            elif obs_mask_type == "tube":
                return "forecastfuturefrompasttube"
            else:
                return "forecastfuturefrompastwithscheduling"
        if epoch % 3 == 1:
            if obs_mask_type == "none":
                return "forecastpast"
            elif obs_mask_type == "tube":
                return "forecastpastfromfuturetube"
            else:
                return "forecastpastfromfuturewithscheduling"
        if epoch % 3 == 2:
            return "interpolatetube"


def train_masked_position_generator(
    process_data,
    window_size,
    obs_frames,
    pred_frames,
    patch_size,
    tublet_size,
    obs_mask_ratio=0.4,
    obs_labmbda=5,
    obs_mask_type="none",
    mask_type="forecastfuture",
    sampling=False,
    temperature=1,
):
    total_frames = (obs_frames + pred_frames) // tublet_size
    obs_frames = obs_frames // tublet_size
    pred_frames = total_frames - obs_frames

    if mask_type == "forecastfuture":
        return ForecastFutureMaskingGenerator(window_size, obs_frames, pred_frames)()
    if mask_type == "forecastpast":
        return ForecastPastMaskingGenerator(window_size, obs_frames, pred_frames)()
    if mask_type == "forecastfuturefrompasttube":
        return ForecastFuturefromPastTubeMaskingGenerator(
            window_size, obs_frames, pred_frames, obs_mask_ratio
        )()
    if mask_type == "forecastpastfromfuturetube":
        return ForecastPastfromFutureTubeMaskingGenerator(
            window_size, obs_frames, pred_frames, obs_mask_ratio
        )()
    if mask_type == "forecastfuturefrompastwithscheduling":
        return ForecastFuturefromPastMaskingGeneratorWithScheduling(
            window_size,
            obs_frames,
            pred_frames,
            obs_mask_ratio,
            lambda_=obs_labmbda,
            scheduling_type=obs_mask_type,
            patch_size=patch_size,
            tublet_size=tublet_size,
            sampling=sampling,
            temperature=temperature,
        )(process_data)
    if mask_type == "forecastpastfromfuturewithscheduling":
        return ForecastPastfromFutureMaskingGeneratorWithScheduling(
            window_size,
            obs_frames,
            pred_frames,
            obs_mask_ratio,
            lambda_=obs_labmbda,
            scheduling_type=obs_mask_type,
            patch_size=patch_size,
            tublet_size=tublet_size,
            sampling=sampling,
            temperature=temperature,
        )(process_data)
    if mask_type == "interpolatetube":
        return InterpolationTubeMaskingGenerator(
            window_size,
            obs_frames,
            pred_frames,
            obs_mask_ratio,
            patch_size,
            tublet_size,
            sampling=sampling,
            temperature=temperature,
        )(process_data)


def test_masked_position_generator(window_size, obs_frames, pred_frames, tublet_size):
    obs_frames = obs_frames // tublet_size
    pred_frames = pred_frames // tublet_size
    return ForecastFutureMaskingGenerator(window_size, obs_frames, pred_frames)()


def cum_exp_dist(lambda_, x):
    return 1 - np.exp(-lambda_ * x)


def linear(x):
    return x


def cubic(x):
    return x**3


def square(x):
    return x**2


def square_root(x):
    return np.sqrt(x)


def compute_weight(process_data, input_size=(5, 10, 10), patch_size=8, tublet_size=4):
    weights = []
    for i in range(input_size[0]):
        for j in range(input_size[1]):
            for k in range(input_size[2]):
                weights.append(
                    torch.sum(
                        process_data[
                            i * tublet_size : (i + 1) * tublet_size,
                            j * patch_size : (j + 1) * patch_size,
                            k * patch_size : (k + 1) * patch_size,
                        ]
                    ).item()
                )
    weights = torch.from_numpy(np.array(weights))
    return weights


def temperature_scaled_softmax(logits, temperature=1.0):
    logits = logits / temperature
    return torch.softmax(logits, dim=0)


class ForecastFutureMaskingGenerator:
    def __init__(self, input_size, obs_frames, pred_frames):
        self.frames, self.height, self.width = input_size
        self.obs_frames, self.pred_frames = obs_frames, pred_frames
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.total_masks = pred_frames * self.num_patches_per_frame

    def __repr__(self):
        repr_str = "Masks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        obs_mask_per_frame = np.hstack(
            [
                np.zeros(self.num_patches_per_frame),
            ]
        )
        pred_mask_per_frame = np.hstack(
            [
                np.ones(self.num_patches_per_frame),
            ]
        )
        obs_mask = np.tile(obs_mask_per_frame, (self.obs_frames, 1)).flatten()
        pred_mask = np.tile(pred_mask_per_frame, (self.pred_frames, 1)).flatten()
        mask = np.hstack([obs_mask, pred_mask])
        return mask


class ForecastPastMaskingGenerator:
    def __init__(self, input_size, obs_frames, pred_frames):
        self.frames, self.height, self.width = input_size
        self.obs_frames, self.pred_frames = obs_frames, pred_frames
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.total_masks = pred_frames * self.num_patches_per_frame

    def __repr__(self):
        repr_str = "Masks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        obs_mask_per_frame = np.hstack(
            [
                np.ones(self.num_patches_per_frame),
            ]
        )
        pred_mask_per_frame = np.hstack(
            [
                np.zeros(self.num_patches_per_frame),
            ]
        )
        obs_mask = np.tile(obs_mask_per_frame, (self.obs_frames, 1)).flatten()
        pred_mask = np.tile(pred_mask_per_frame, (self.pred_frames, 1)).flatten()
        mask = np.hstack([obs_mask, pred_mask])
        return mask


class ForecastFuturefromPastTubeMaskingGenerator:
    def __init__(self, input_size, obs_frames, pred_frames, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.obs_frames, self.pred_frames = obs_frames, pred_frames
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame_obs_frames = int(
            mask_ratio * self.num_patches_per_frame
        )
        self.total_masks = (
            obs_frames * self.num_masks_per_frame_obs_frames
            + pred_frames * self.num_patches_per_frame
        )

    def __repr__(self):
        repr_str = "Masks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        obs_mask_per_frame = np.hstack(
            [
                np.zeros(
                    self.num_patches_per_frame - self.num_masks_per_frame_obs_frames
                ),
                np.ones(self.num_masks_per_frame_obs_frames),
            ]
        )
        np.random.shuffle(obs_mask_per_frame)
        pred_mask_per_frame = np.hstack(
            [
                np.ones(self.num_patches_per_frame),
            ]
        )
        obs_mask = np.tile(obs_mask_per_frame, (self.obs_frames, 1)).flatten()
        pred_mask = np.tile(pred_mask_per_frame, (self.pred_frames, 1)).flatten()
        mask = np.hstack([obs_mask, pred_mask])
        return mask


class ForecastPastfromFutureTubeMaskingGenerator:
    def __init__(self, input_size, obs_frames, pred_frames, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.obs_frames, self.pred_frames = obs_frames, pred_frames
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame_pred_frames = int(
            mask_ratio * self.num_patches_per_frame
        )
        self.total_masks = (
            obs_frames * self.num_patches_per_frame
            + pred_frames * self.num_masks_per_frame_pred_frames
        )

    def __repr__(self):
        repr_str = "Masks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        obs_mask_per_frame = np.hstack(
            [
                np.ones(self.num_patches_per_frame),
            ]
        )
        pred_mask_per_frame = np.hstack(
            [
                np.zeros(
                    self.num_patches_per_frame - self.num_masks_per_frame_pred_frames
                ),
                np.ones(self.num_masks_per_frame_pred_frames),
            ]
        )
        np.random.shuffle(pred_mask_per_frame)

        obs_mask = np.tile(obs_mask_per_frame, (self.obs_frames, 1)).flatten()
        pred_mask = np.tile(pred_mask_per_frame, (self.pred_frames, 1)).flatten()
        mask = np.hstack([obs_mask, pred_mask])
        return mask


class InterpolationTubeMaskingGenerator:
    def __init__(
        self,
        input_size,
        obs_frames,
        pred_frames,
        mask_ratio,
        patch_size,
        tublet_size,
        sampling=False,
        temperature=1,
    ):
        self.input_size = input_size
        self.patch_size, self.tublet_size = patch_size, tublet_size
        self.frames, self.height, self.width = input_size
        self.obs_frames, self.pred_frames = obs_frames, pred_frames
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame
        self.sampling = sampling
        self.temperature = temperature

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self, process_data):
        if self.sampling:
            weights = compute_weight(
                process_data,
                self.input_size,
                patch_size=self.patch_size,
                tublet_size=self.tublet_size,
            )
            for i in range(1, len(weights) // self.num_patches_per_frame):
                weights[: self.num_patches_per_frame] += weights[
                    i
                    * self.num_patches_per_frame : (i + 1)
                    * self.num_patches_per_frame
                ]
            weight = weights[: self.num_patches_per_frame] / (
                len(weights) // self.num_patches_per_frame
            )
            weights_scaled = temperature_scaled_softmax(
                weight,
                temperature=self.temperature,
            ).numpy()
        if self.sampling and not np.isnan(weights_scaled).any():
            obs_mask_index = np.random.choice(
                len(weights_scaled),
                self.num_masks_per_frame,
                replace=False,
                p=weights_scaled,
            )
            pred_mask_index = np.random.choice(
                len(weights_scaled),
                self.num_masks_per_frame,
                replace=False,
                p=weights_scaled,
            )
            obs_mask_per_frame = np.zeros(self.num_patches_per_frame)
            pred_mask_per_frame = np.zeros(self.num_patches_per_frame)
            obs_mask_per_frame[obs_mask_index] = 1.0
            pred_mask_per_frame[pred_mask_index] = 1.0
        else:
            obs_mask_per_frame = np.hstack(
                [
                    np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
                    np.ones(self.num_masks_per_frame),
                ]
            )
            np.random.shuffle(obs_mask_per_frame)
            pred_mask_per_frame = np.hstack(
                [
                    np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
                    np.ones(self.num_masks_per_frame),
                ]
            )
            np.random.shuffle(pred_mask_per_frame)
        obs_mask = np.tile(obs_mask_per_frame, (self.obs_frames, 1)).flatten()
        pred_mask = np.tile(pred_mask_per_frame, (self.pred_frames, 1)).flatten()
        mask = np.hstack([obs_mask, pred_mask])
        return mask


class ForecastFuturefromPastMaskingGeneratorWithScheduling:
    def __init__(
        self,
        input_size,
        obs_frames,
        pred_frames,
        mask_ratio,
        lambda_,
        scheduling_type,
        patch_size,
        tublet_size,
        sampling=False,
        temperature=1,
    ):
        self.input_size = input_size
        self.patch_size, self.tublet_size = patch_size, tublet_size
        self.frames, self.height, self.width = input_size
        self.obs_frames, self.pred_frames = obs_frames, pred_frames
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.sampling = sampling
        self.temperature = temperature
        self.num_masks_per_frame_obs_frames = []
        if scheduling_type == "sine":
            schedule = CosineAnnealingWarmupRestarts(
                first_cycle_steps=(obs_frames + 1), max_lr=1, min_lr=0, warmup_steps=0
            )
        for i in range(obs_frames):
            if scheduling_type == "linear":
                mask_ratio = linear((i + 1) / (obs_frames + 1))
            elif scheduling_type == "cubic":
                mask_ratio = cubic((i + 1) / (obs_frames + 1))
            elif scheduling_type == "square":
                mask_ratio = square((i + 1) / (obs_frames + 1))
            elif scheduling_type == "square_root":
                mask_ratio = square_root((i + 1) / (obs_frames + 1))
            elif scheduling_type == "sine":
                mask_ratio = 1 - np.array(schedule.step(i + 1))
            elif scheduling_type == "exp":
                mask_ratio = cum_exp_dist(lambda_, (i + 1) / (obs_frames + 1))
            elif scheduling_type == "random":
                mask_ratio = mask_ratio
            self.num_masks_per_frame_obs_frames.append(
                int(mask_ratio * self.num_patches_per_frame)
            )

    def __call__(self, process_data):
        if self.sampling:
            weights = compute_weight(
                process_data,
                self.input_size,
                patch_size=self.patch_size,
                tublet_size=self.tublet_size,
            )
        obs_mask = []
        for i, num_masks_per_frame_obs_frame in enumerate(
            self.num_masks_per_frame_obs_frames
        ):
            if self.sampling and not np.isnan(weights).any():
                weights_scaled = temperature_scaled_softmax(
                    weights[
                        i
                        * self.num_patches_per_frame : (i + 1)
                        * self.num_patches_per_frame
                    ],
                    temperature=self.temperature,
                ).numpy()
                mask_index = np.random.choice(
                    len(weights_scaled),
                    num_masks_per_frame_obs_frame,
                    replace=False,
                    p=weights_scaled,
                )
                obs_mask_per_frame = np.zeros(self.num_patches_per_frame)
                obs_mask_per_frame[mask_index] = 1.0
            else:
                obs_mask_per_frame = np.hstack(
                    [
                        np.zeros(
                            self.num_patches_per_frame - num_masks_per_frame_obs_frame
                        ),
                        np.ones(num_masks_per_frame_obs_frame),
                    ]
                )
                np.random.shuffle(obs_mask_per_frame)
            obs_mask.append(obs_mask_per_frame)
        pred_mask_per_frame = np.hstack(
            [
                np.ones(self.num_patches_per_frame),
            ]
        )
        obs_mask = np.array(obs_mask).flatten()
        pred_mask = np.tile(pred_mask_per_frame, (self.pred_frames, 1)).flatten()
        mask = np.hstack([obs_mask, pred_mask])
        return mask


class ForecastPastfromFutureMaskingGeneratorWithScheduling:
    def __init__(
        self,
        input_size,
        obs_frames,
        pred_frames,
        mask_ratio,
        lambda_,
        scheduling_type,
        patch_size,
        tublet_size,
        sampling=False,
        temperature=1,
    ):
        self.input_size = input_size
        self.patch_size, self.tublet_size = patch_size, tublet_size
        self.frames, self.height, self.width = input_size
        self.obs_frames, self.pred_frames = obs_frames, pred_frames
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.sampling = sampling
        self.temperature = temperature
        self.num_masks_per_frame_pred_frames = []
        if scheduling_type == "sine":
            schedule = CosineAnnealingWarmupRestarts(
                first_cycle_steps=(pred_frames + 1), max_lr=1, min_lr=0, warmup_steps=0
            )
        for i in range(pred_frames - 1, -1, -1):
            if scheduling_type == "linear":
                mask_ratio = linear((i + 1) / (pred_frames + 1))
            elif scheduling_type == "cubic":
                mask_ratio = cubic((i + 1) / (pred_frames + 1))
            elif scheduling_type == "square":
                mask_ratio = square((i + 1) / (pred_frames + 1))
            elif scheduling_type == "square_root":
                mask_ratio = square_root((i + 1) / (pred_frames + 1))
            elif scheduling_type == "sine":
                mask_ratio = 1 - np.array(schedule.step(i + 1))
            elif scheduling_type == "exp":
                mask_ratio = cum_exp_dist(lambda_, (i + 1) / (pred_frames + 1))
            self.num_masks_per_frame_pred_frames.append(
                int(mask_ratio * self.num_patches_per_frame)
            )

    def __call__(self, process_data):
        if self.sampling:
            weights = compute_weight(
                process_data,
                self.input_size,
                patch_size=self.patch_size,
                tublet_size=self.tublet_size,
            )
        obs_mask_per_frame = np.hstack(
            [
                np.ones(self.num_patches_per_frame),
            ]
        )
        pred_mask = []
        for i, num_masks_per_frame_pred_frame in enumerate(
            self.num_masks_per_frame_pred_frames
        ):
            if self.sampling and not np.isnan(weights).any():
                weights_scaled = temperature_scaled_softmax(
                    weights[
                        i
                        * self.num_patches_per_frame : (i + 1)
                        * self.num_patches_per_frame
                    ],
                    temperature=self.temperature,
                ).numpy()
                mask_index = np.random.choice(
                    len(weights_scaled),
                    num_masks_per_frame_pred_frame,
                    replace=False,
                    p=weights_scaled,
                )
                pred_mask_per_frame = np.zeros(self.num_patches_per_frame)
                pred_mask_per_frame[mask_index] = 1.0
            else:
                pred_mask_per_frame = np.hstack(
                    [
                        np.zeros(
                            self.num_patches_per_frame - num_masks_per_frame_pred_frame
                        ),
                        np.ones(num_masks_per_frame_pred_frame),
                    ]
                )
                np.random.shuffle(pred_mask_per_frame)
            pred_mask.append(pred_mask_per_frame)
        obs_mask = np.tile(obs_mask_per_frame, (self.obs_frames, 1)).flatten()
        pred_mask = np.array(pred_mask).flatten()
        mask = np.hstack([obs_mask, pred_mask])
        return mask
