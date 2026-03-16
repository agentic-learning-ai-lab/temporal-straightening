import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from models.vit import Transformer

class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        dim=384, # target output dimension
        depth=4,
        heads=8, # heads for internal dim
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
        agg_type="flatten",
        agg_out_dim=None,
        agg_mlp_hidden_dim=None,
        internal_dim=256,
        **kwargs
    ):
        super().__init__()
        self.name = "scratch_vit"
        self.emb_dim = dim
        self.patch_size = patch_size
        self.agg_type = agg_type
        self.agg_out_dim = agg_out_dim
        self.agg_mlp_hidden_dim = agg_mlp_hidden_dim
        self.internal_dim = internal_dim
        
        # Verify img_size is divisible by patch_size
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
            
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_size * patch_size * in_chans),
            nn.Linear(patch_size * patch_size * in_chans, internal_dim),
            nn.LayerNorm(internal_dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, internal_dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # heads * dim_head = internal_dim
        dim_head = internal_dim // heads
        self.transformer = Transformer(internal_dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # Projection head to target dimension
        self.to_out = nn.Sequential(
            nn.Linear(internal_dim, internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, dim),
            nn.LayerNorm(dim)
        )
        
        self.latent_ndim = 2 # (b, n, d)

        if self.agg_type == "mlp":
            self._agg_mlp_in_dim = num_patches * int(self.emb_dim)
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
        # flatten tokens
        x = x.contiguous().view(x.shape[0], -1)
        if self.agg_type == "flatten":
            return x
        if self.agg_type == "mlp":
            x = self.agg_mlp(x)
            return self.agg_post_norm(x)
        return x

    def forward(self, x):
        # x shape: (b, c, h, w) or (b*t, c, h, w)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_out(x)
        return x
