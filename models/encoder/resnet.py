import torch
import torchvision
import torch.nn as nn


class resnet18(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        unit_norm: bool = False,
    ):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.pretrained = pretrained
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unit_norm = unit_norm

        self.latent_ndim = 1
        self.emb_dim = 512
        self.name = "resnet"

    def forward(self, x):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            # flatten all dimensions to batch, then reshape back at the end
            x = x.reshape(-1, *orig_shape[-3:])
        x = self.normalize(x)
        out = self.resnet(x)
        out = self.flatten(out)
        if self.unit_norm:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], -1)
        out = out.unsqueeze(1)
        return out


class resblock(nn.Module):
    # this implementation assumes square images
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=32):
        super(resblock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample

        padding = int((kernel_size - 1) / 2)

        if resample == "down":
            self.skip = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding),
            )
            self.conv1 = nn.Conv2d(
                input_dim, input_dim, kernel_size, padding=padding, bias=False
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding),
                nn.MaxPool2d(2, stride=2),
            )
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample is None:
            self.skip = nn.Conv2d(input_dim, output_dim, 1)
            self.conv1 = nn.Conv2d(
                input_dim, output_dim, kernel_size, padding=padding, bias=False
            )
            self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size, padding=padding)
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)

        self.leakyrelu1 = nn.LeakyReLU()
        self.leakyrelu2 = nn.LeakyReLU()

    def forward(self, x):
        if (self.input_dim == self.output_dim) and self.resample is None:
            idnty = x
        else:
            idnty = self.skip(x)

        residual = x
        residual = self.conv1(residual)
        residual = self.bn1(residual)
        residual = self.leakyrelu1(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.leakyrelu2(residual)

        return idnty + residual


class SmallResNet(nn.Module):
    def __init__(self, dim=512):
        super(SmallResNet, self).__init__()

        self.hw = 224
        self.name = "small_resnet"
        self.emb_dim = dim
        self.latent_ndim = 1

        # 3x224x224
        self.rb1 = resblock(3, 16, 3, resample="down", hw=self.hw)
        # 16x112x112
        self.rb2 = resblock(16, 32, 3, resample="down", hw=self.hw // 2)
        # 32x56x56
        self.rb3 = resblock(32, 64, 3, resample="down", hw=self.hw // 4)
        # 64x28x28
        self.rb4 = resblock(64, 128, 3, resample="down", hw=self.hw // 8)
        # 128x14x14
        self.rb5 = resblock(128, 256, 3, resample="down", hw=self.hw // 16)
        # 512x7x7
        
        self.flat = nn.Flatten()
        
        self.projection = nn.Sequential(
            nn.Linear(256*7*7, 512),
            nn.GELU(),
            nn.Linear(512, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            # flatten all dimensions to batch, then reshape back at the end
            x = x.reshape(-1, *orig_shape[-3:])
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        #x = self.maxpool(x)
        out = self.flat(x)
        
        out = self.projection(out)
        out = out.unsqueeze(1) # (B, 1, dim) dummy patch dim
        
        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], 1, -1) # (..., 1, dim)
        return out


class ResNetSpatial(nn.Module):
    def __init__(
        self,
        dim=512,
        agg_type="flatten",
        agg_out_dim=None,
        agg_mlp_hidden_dim=None,
    ):
        super(ResNetSpatial, self).__init__()

        self.hw = 224
        self.name = "resnet_spatial"
        self.emb_dim = dim
        self.latent_ndim = 2
        self.agg_type = agg_type
        self.agg_out_dim = agg_out_dim
        self.agg_mlp_hidden_dim = agg_mlp_hidden_dim

        # Slightly wider/deeper than SmallResNetSpatial while keeping 14x14 tokens.
        self.rb1_down = resblock(3, 32, 3, resample="down", hw=self.hw)  # 112x112
        self.rb1 = resblock(32, 32, 3, resample=None, hw=self.hw // 2)

        self.rb2_down = resblock(32, 64, 3, resample="down", hw=self.hw // 2)  # 56x56
        self.rb2 = resblock(64, 64, 3, resample=None, hw=self.hw // 4)

        self.rb3_down = resblock(64, 128, 3, resample="down", hw=self.hw // 4)  # 28x28
        self.rb3 = resblock(128, 128, 3, resample=None, hw=self.hw // 8)

        self.rb4_down = resblock(128, 256, 3, resample="down", hw=self.hw // 8)  # 14x14
        self.rb4 = resblock(256, 256, 3, resample=None, hw=self.hw // 16)

        self.rb5 = resblock(256, dim, 3, resample=None, hw=self.hw // 16)

        self.post_norm = nn.LayerNorm(dim)
        self.num_tokens = (self.hw // 16) ** 2

        if self.agg_type == "mlp":
            self._agg_mlp_in_dim = self.num_tokens * int(self.emb_dim)
            self._agg_out_dim = (
                int(self.agg_out_dim)
                if self.agg_out_dim is not None
                else int(self.emb_dim)
            )
            self._agg_mlp_hidden_dim = (
                int(self.agg_mlp_hidden_dim)
                if self.agg_mlp_hidden_dim is not None
                else 4 * self._agg_out_dim
            )
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
        return x

    def forward(self, x, return_agg=False):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            x = x.reshape(-1, *orig_shape[-3:])

        x = self.rb1_down(x)
        x = self.rb1(x)
        x = self.rb2_down(x)
        x = self.rb2(x)
        x = self.rb3_down(x)
        x = self.rb3(x)
        x = self.rb4_down(x)
        x = self.rb4(x)
        x = self.rb5(x)

        out = x.flatten(2).transpose(1, 2).contiguous()  # (B, 196, dim)
        out = self.post_norm(out)
        self.latent_ndim = 2

        if return_agg and out.dim() == 3:
            out = self.agg(out)
            self.latent_ndim = 1

        if self.latent_ndim == 1:
            out = out.unsqueeze(1)  # dummy patch dim

        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            if self.latent_ndim == 1:
                out = out.reshape(*orig_shape[:-3], 1, -1)
            else:
                out = out.reshape(*orig_shape[:-3], self.num_tokens, -1)
        return out
