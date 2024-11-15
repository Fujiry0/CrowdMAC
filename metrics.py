import torch
import torch.nn as nn
from einops import rearrange


def generate_predicted_videos(
    outputs,
    videos_patch,
    bool_masked_pos,
    batch_size,
    input_size,
    patch_size,
    tublet_size,
    num_frames,
):
    predicted_patch = videos_patch.clone()
    predicted_patch[bool_masked_pos] = outputs.reshape(
        [-1, tublet_size * patch_size * patch_size]
    ).to(torch.float32)
    predicted_videos = rearrange(
        predicted_patch,
        "b (t h w) (p0 p1 p2 c) ->  b c (t p0) (h p1) (w p2)",
        b=batch_size,
        c=1,
        t=num_frames // tublet_size,
        h=input_size // patch_size,
        w=input_size // patch_size,
        p0=tublet_size,
        p1=patch_size,
        p2=patch_size,
    )

    return predicted_videos


def compute_eval_metrics(videos, predicted_videos, obs_frames=8):
    B, C, T, H, W = videos.shape
    gt_fut_videos = videos[:, :, obs_frames:].reshape(-1, H, W)
    predicted_fut_videos = predicted_videos[:, :, obs_frames:].reshape(-1, H, W)

    d_akl = kl_divergence(gt_fut_videos, predicted_fut_videos)
    d_arkl = kl_divergence(predicted_fut_videos, gt_fut_videos)
    d_ajs = js_divergence(gt_fut_videos, predicted_fut_videos)

    gt_fut_videos = torch.unsqueeze(videos[:, :, -1], dim=2).reshape(-1, H, W)
    predicted_fut_videos = torch.unsqueeze(predicted_videos[:, :, -1], dim=2).reshape(
        -1, H, W
    )

    d_fkl = kl_divergence(gt_fut_videos, predicted_fut_videos)
    d_frkl = kl_divergence(predicted_fut_videos, gt_fut_videos)
    d_fjs = js_divergence(gt_fut_videos, predicted_fut_videos)

    return d_akl, d_arkl, d_ajs, d_fkl, d_frkl, d_fjs


def kl_divergence(p, q, eps=1e-10):
    p += eps
    q += eps
    p = p / (torch.sum(p, dim=(1, 2), keepdim=True) + eps)
    q = q / (torch.sum(q, dim=(1, 2), keepdim=True) + eps)
    p = torch.clamp(p, min=eps, max=1 - eps)
    q = torch.clamp(q, min=eps, max=1 - eps)
    return nn.KLDivLoss(reduction="batchmean")((q + eps).log(), p)


def js_divergence(p, q, eps=1e-10):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m, eps) + kl_divergence(q, m, eps))


if __name__ == "__main__":
    gt_videos = torch.rand([4, 1, 20, 80, 80])
    pred_videos = torch.rand([4, 1, 20, 80, 80])

    d_akl, d_arkl, d_ajs, d_fkl, d_frkl, d_fjs = compute_eval_metrics(
        gt_videos, pred_videos, obs_frames=8
    )
    print(d_akl, d_arkl, d_ajs, d_fkl, d_frkl, d_fjs)
