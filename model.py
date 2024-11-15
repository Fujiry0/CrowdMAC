# Standard Library
from functools import partial

# Third Party Library
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

# First Party Library
from layers import Block, PatchEmbed, get_sinusoid_encoding_table


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class VisionTransformerEncoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        num_frames=16,
        tubelet_size=2,
        use_checkpoint=False,
        use_learnable_pos_emb=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis)
        else:
            for blk in self.blocks:
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class VisionTransformerDecoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        num_classes=768,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        num_patches=196,
        tubelet_size=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == in_chans * tubelet_size * patch_size**2
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x, return_token_num):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            x = self.head(
                self.norm(x[:, -return_token_num:])
            )  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_in_chans=3,
        decoder_num_classes=1536,  # decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        use_checkpoint=False,
        num_frames=16,
        tubelet_size=2,
        num_classes=0,  # avoid the error from create_fn in timm
        in_chans=0,  # avoid the error from create_fn in timm
    ):
        super().__init__()
        self.encoder = VisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb,
        )

        self.decoder = VisionTransformerDecoder(
            patch_size=patch_size,
            in_chans=decoder_in_chans,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
        )

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim
        )

        trunc_normal_(self.mask_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "mask_token"}

    def forward(self, x, mask):
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = (
            self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        )
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat(
            [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1
        )  # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]

        return x


def get_model(cfg):
    if cfg.model.size == "small":
        model = VisionTransformer(
            img_size=cfg.model.input_size,
            patch_size=cfg.model.patch_size,
            in_chans=1,
            tubelet_size=cfg.model.tublet_size,
            num_frames=cfg.dataset.num_frames,
            encoder_in_chans=1,
            encoder_embed_dim=384,
            encoder_depth=12,
            encoder_num_heads=6,
            encoder_num_classes=0,
            decoder_in_chans=1,
            decoder_num_classes=1 * cfg.model.tublet_size * cfg.model.patch_size**2,
            decoder_embed_dim=192,
            decoder_num_heads=3,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=cfg.model.drop_path,
            decoder_depth=cfg.model.decoder_depth,
            use_checkpoint=cfg.model.use_checkpoint,
        )
    elif cfg.model.size == "base":
        model = VisionTransformer(
            img_size=cfg.model.input_size,
            patch_size=cfg.model.patch_size,
            in_chans=1,
            tubelet_size=cfg.model.tublet_size,
            num_frames=cfg.dataset.num_frames,
            encoder_in_chans=1,
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_num_classes=0,
            decoder_in_chans=1,
            decoder_num_classes=1 * cfg.model.tublet_size * cfg.model.patch_size**2,
            decoder_embed_dim=384,
            decoder_num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=cfg.model.drop_path,
            decoder_depth=cfg.model.decoder_depth,
            use_checkpoint=cfg.model.use_checkpoint,
        )
    elif cfg.model.size == "large":
        model = VisionTransformer(
            img_size=cfg.model.input_size,
            patch_size=cfg.model.patch_size,
            in_chans=1,
            tubelet_size=cfg.model.tublet_size,
            num_frames=cfg.dataset.num_frames,
            encoder_in_chans=1,
            encoder_embed_dim=1024,
            encoder_depth=24,
            encoder_num_heads=16,
            encoder_num_classes=0,
            decoder_in_chans=1,
            decoder_num_classes=1 * cfg.model.tublet_size * cfg.model.patch_size**2,
            decoder_embed_dim=512,
            decoder_num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=cfg.model.drop_path,
            decoder_depth=cfg.model.decoder_depth,
            use_checkpoint=cfg.model.use_checkpoint,
        )
    elif cfg.model.size == "huge":
        model = VisionTransformer(
            img_size=cfg.model.input_size,
            patch_size=cfg.model.patch_size,
            in_chans=1,
            tubelet_size=cfg.model.tublet_size,
            num_frames=cfg.dataset.num_frames,
            encoder_in_chans=1,
            encoder_embed_dim=1280,
            encoder_depth=32,
            encoder_num_heads=16,
            encoder_num_classes=0,
            decoder_in_chans=1,
            decoder_num_classes=1 * cfg.model.tublet_size * cfg.model.patch_size**2,
            decoder_embed_dim=640,
            decoder_num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=cfg.model.drop_path,
            decoder_depth=cfg.model.decoder_depth,
            use_checkpoint=cfg.model.use_checkpoint,
        )
    elif cfg.model.size == "small_symmetry":
        model = VisionTransformer(
            img_size=cfg.model.input_size,
            patch_size=cfg.model.patch_size,
            in_chans=1,
            tubelet_size=cfg.model.tublet_size,
            num_frames=cfg.dataset.num_frames,
            encoder_in_chans=1,
            encoder_embed_dim=384,
            encoder_depth=12,
            encoder_num_heads=6,
            encoder_num_classes=0,
            decoder_in_chans=1,
            decoder_num_classes=1 * cfg.model.tublet_size * cfg.model.patch_size**2,
            decoder_embed_dim=384,
            decoder_num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=cfg.model.drop_path,
            decoder_depth=12,
            use_checkpoint=cfg.model.use_checkpoint,
        )

    return model


if __name__ == "__main__":
    from einops import rearrange
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {
                "input_size": 80,
                "num_frames": 20,
                "tublet_size": 4,
                "patch_size": 8,
                "model": "CrowdMAC",
                "drop_path": 0.0,
                "decoder_depth": 4,
                "use_checkpoint": False,
                "size": "small",
            },
            "augmentation": {
                "train": {
                    "RandomHorizontalFlip": {"apply": True, "p": 0.5},
                    "RandomVerticalFlip": {"apply": True, "p": 0.5},
                },
            },
            "dataset": {
                "data_dir": "/mnt/ssd1/fujii/traj_pred/data/stanford_campus_dataset",
                "num_frames": 20,
            },
            "forecast": {"obs_frames": 8, "pred_frames": 12},
        }
    )
    patch_size = 8
    tublet_size = 4
    videos = torch.rand([4, 1, 20, 80, 80])
    mask = torch.zeros([4, 500])
    mask[:, 200:] = 1.0
    videos_patch = rearrange(
        videos,
        "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)",
        p0=tublet_size,  # tubelet size
        p1=patch_size,
        p2=patch_size,
    )

    bool_masked_pos = mask.flatten(1).to(torch.bool)
    B, _, C = videos_patch.shape
    labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
    model = get_model(cfg)
    outputs = model(videos, bool_masked_pos)
    print(f"Outputs: {outputs.shape}")
