import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

logger = logging.getLogger(__name__)
class GlobalProjector(nn.Module):
    def __init__(
        self,
        in_dim=384,
        out_dim=384,
        hidden=384,
        num_downsamples=3,
        pool_hw=1,
        gn_groups=8,
        norm_type=None,
        conv_layers=None,
        **kwargs,
    ):
        super().__init__()
        self.pool_hw = int(pool_hw)

        self.mix = nn.Conv2d(in_dim, hidden, kernel_size=1, bias=False)
        self.gn0 = nn.GroupNorm(num_groups=max(1, min(gn_groups, hidden)), num_channels=hidden)

        blocks = []
        for cfg in conv_layers:
            blocks.append(
                nn.Conv2d(
                    cfg.in_dim,
                    cfg.out_dim,
                    kernel_size=cfg.kernel_size,
                    stride=cfg.stride,
                    padding=cfg.padding,
                    bias=False,
                )
            )
            blocks.append(
                nn.GroupNorm(
                    num_groups=max(1, min(gn_groups, cfg.out_dim)),
                    num_channels=cfg.out_dim,
                )
            )
            blocks.append(nn.GELU())
        last_dim = conv_layers[-1].out_dim
        self.down_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((self.pool_hw, self.pool_hw))
        self.head = nn.Conv2d(last_dim, out_dim, kernel_size=1, bias=False)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = F.gelu(self.gn0(self.mix(x)))  # [B, 384, 14, 14] -> [B, 384, 14, 14]
        x = self.down_blocks(x)  # [B, 384, 14, 14] -> [B, 384, 2, 2]
        x = self.pool(x)  # [B, 384, 2, 2] -> [B, 384, pool_hw, pool_hw]
        x = self.head(x)  # [B, 384, pool_hw, pool_hw] -> [B, out_dim, pool_hw, pool_hw]
        if self.pool_hw == 1:
            x = x.flatten(1) #[B, 384, 1, 1] -> [B, 384]
            return self.ln(x)
        x = x.flatten(2).transpose(1, 2).contiguous() #[B, 384, 2, 2] -> [B, 4, 384]
        return self.ln(x)


class ChannelProjector(nn.Module):
    def __init__(
        self,
        in_dim=384,
        out_dim=384,
        norm_type=None,
        conv_layers=None,
        **kwargs,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.layer_norm_layers = nn.ModuleList()

        for cfg in (conv_layers or []):
            out_ch = cfg.out_dim
            self.conv_layers.append(
                nn.Conv2d(in_channels=cfg.in_dim, out_channels=cfg.out_dim, kernel_size=cfg.kernel_size, stride=cfg.stride, padding=cfg.padding)
            )
            self.batch_norm_layers.append(nn.BatchNorm2d(out_ch))
            self.layer_norm_layers.append(nn.LayerNorm(out_ch))

        self.activation = nn.ReLU()

    def forward(self, x):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.norm_type == "batch":
                x = self.batch_norm_layers[i](x)
            elif self.norm_type == "layer":
                x = x.permute(0, 2, 3, 1)
                x = self.layer_norm_layers[i](x)
                x = x.permute(0, 3, 1, 2)
            if len(self.conv_layers) > 1 and i != len(self.conv_layers) - 1:
                x = self.activation(x)

        x = x.flatten(2).transpose(1, 2)
        return x


class DinoV2Encoder(nn.Module):
    def __init__(
        self,
        name,
        feature_key,
        projector="none",
        projector_config=None,
        agg_type="flatten",
        agg_out_dim=None,
        agg_mlp_hidden_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        self.projector_name = projector
        self.agg_type = agg_type
        self.agg_out_dim = agg_out_dim
        self.agg_mlp_hidden_dim = agg_mlp_hidden_dim
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
            if projector in ("channel", "global"):
                if projector_config is None:
                    raise ValueError(
                        f"projector_config is required when projector='{projector}'"
                    )
                self.projector = projector_config
                if projector == "global" and hasattr(self.projector, "head"):
                    self.emb_dim = self.projector.head.out_channels
                elif hasattr(self.projector, "conv_layers") and len(self.projector.conv_layers) > 0:
                    self.emb_dim = self.projector.conv_layers[-1].out_channels
                elif hasattr(self.projector, "head"):
                    self.emb_dim = self.projector.head.out_channels
            else:
                logger.warning(
                    "Unknown projector '%s' for patch tokens; proceeding without projector.",
                    projector,
                )
                pass
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size
        if self.agg_type == "mlp":
            self._agg_mlp_in_dim = 196 * int(self.emb_dim)
            self._agg_out_dim = int(self.agg_out_dim) if self.agg_out_dim is not None else int(self.emb_dim)
            self._agg_mlp_hidden_dim = int(self.agg_mlp_hidden_dim) if self.agg_mlp_hidden_dim is not None else 4 * self._agg_out_dim
            self.agg_mlp = nn.Sequential(
                nn.Linear(self._agg_mlp_in_dim, self._agg_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self._agg_mlp_hidden_dim, self._agg_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self._agg_mlp_hidden_dim, self._agg_out_dim),
            )
            self.agg_post_norm = nn.LayerNorm(self._agg_out_dim)

    def agg(self, x):
        if self.agg_type == "mean":
            return x.mean(dim=1)
        x = x.contiguous().view(x.shape[0], -1)
        if self.agg_type == "flatten":
            return x
        if self.agg_type == "mlp":
            x = self.agg_mlp(x)
            return self.agg_post_norm(x)
        logger.warning(
            "Unknown agg_type '%s'. Expected 'mean', 'flatten', or 'mlp'.",
            self.agg_type,
        )
        return x

    def forward(self, x, return_agg=False):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if hasattr(self, "projector"):
            b, n, c = emb.shape
            h = w = int(n ** 0.5)
            if h * w != n:
                raise ValueError(f"Expected square number of tokens, got N={n}")
            emb = emb.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, 196, 384] -> [B, 384, 14, 14]
            if self.projector_name == "channel":
                emb = self.projector(emb)
                self.latent_ndim = 2
            elif self.projector_name == "global":
                emb = self.projector(emb)
                self.latent_ndim = 2 if emb.dim() == 3 else 1
        if return_agg and emb.dim() == 3:
            emb = self.agg(emb)
            #self.latent_ndim = 1
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb