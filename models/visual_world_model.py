import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision import transforms
from einops import rearrange, repeat

log = logging.getLogger(__name__)

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        straighten=False,
        stop_grad=True,
        vcreg=False,
        vcreg_std_coeff=0,
        vcreg_cov_coeff=0,
        vcreg_apply_to="enc",
        **kwargs,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used
        self.straighten = False
        self.straighten_scale = 0.0
        self.curvature_mode = None
        self.stop_grad = bool(stop_grad)
        self.vcreg = bool(vcreg)
        self.std_coeff = float(vcreg_std_coeff)
        self.cov_coeff = float(vcreg_cov_coeff)
        if vcreg_apply_to != "enc":
            raise ValueError(
                f"Only encoder VCReg is supported, got vcreg_apply_to='{vcreg_apply_to}'."
            )

        if isinstance(straighten, str):
            if straighten.startswith("aggcos"):
                suffix = straighten.replace("aggcos", "")
                self.straighten_scale = float(suffix) if suffix else 1.0
                self.curvature_mode = "aggcos"
            elif straighten.startswith("cos"):
                suffix = straighten.replace("cos", "")
                self.straighten_scale = float(suffix) if suffix else 1.0
                self.curvature_mode = "cos"

        self.straighten = self.curvature_mode is not None and self.straighten_scale > 0

        log.info("num_action_repeat: %s", self.num_action_repeat)
        log.info("num_proprio_repeat: %s", self.num_proprio_repeat)
        log.info("proprio encoder: %s", proprio_encoder)
        log.info("action encoder: %s", action_encoder)
        log.info("proprio_dim: %s, after repeat: %s", proprio_dim, self.proprio_dim)
        log.info("action_dim: %s, after repeat: %s", action_dim, self.action_dim)
        log.info("emb_dim: %s", self.emb_dim)
        if self.straighten:
            log.info(
                "Straightening enabled: mode=%s, scale=%s",
                self.curvature_mode,
                self.straighten_scale,
            )
        else:
            log.info("Straightening disabled")
        log.info("Stop-grad enabled: %s", self.stop_grad)
        log.info(
            "VCReg enabled: %s, apply_to=enc, std_coeff=%s, cov_coeff=%s",
            self.vcreg,
            self.std_coeff,
            self.cov_coeff,
        )

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        log.info("Model emb_dim: %s", self.emb_dim)

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z
    
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def visual_only(self, z):
        if self.concat_dim == 0:
            return z[:, :, :-2, :]
        drop = self.proprio_dim + self.action_dim
        return z[..., :-drop] if drop > 0 else z

    def visual_prop(self, z):
        if self.concat_dim == 0:
            return z[:, :, :-1, :]
        return z[..., :-self.action_dim]

    def vcreg_std_loss(self, z: torch.Tensor) -> torch.Tensor:
        x = z.reshape(-1, z.shape[-1])
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std_x))

    def vcreg_cov_loss(self, z: torch.Tensor) -> torch.Tensor:
        x = z.reshape(-1, z.shape[-1])
        _, d = x.shape
        x = x - x.mean(dim=0)
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum() / d
        return cov_loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _cos_curvature(self, v1, v2, eps=1e-6, step_thresh=1e-6):
        cos = F.cosine_similarity(v1, v2, dim=-1, eps=eps)
        loss = 1.0 - cos
        if step_thresh > 0:
            step1 = v1.norm(dim=-1)
            step2 = v2.norm(dim=-1)
            mask = (step1 > step_thresh) & (step2 > step_thresh)
            loss = loss[mask]
        return loss.mean()

    def total_curvature(self, features, mode="cos"):
        if features.shape[1] < 3:
            raise ValueError(f"Features must have at least 3 frames for curvature calculation, got {features.shape[1]}")

        if mode == "aggcos":
            if not hasattr(self.encoder, "agg"):
                raise ValueError("curvature mode 'aggcos' requires encoder.agg().")
            b, t, p, d = features.shape
            tokens = features.reshape(b * t, p, d)
            z = self.encoder.agg(tokens).reshape(b, t, -1)
            v1 = z[:, 1:-1] - z[:, :-2]
            v2 = z[:, 2:] - z[:, 1:-1]
        elif mode == "cos":
            v1 = features[:, 1:-1] - features[:, :-2]
            v2 = features[:, 2:] - features[:, 1:-1]
        else:
            raise ValueError(f"Unknown curvature mode '{mode}'. Use 'cos' or 'aggcos'.")

        return self._cos_curvature(v1, v2)

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        decoder_enabled = self.decoder is not None and self.train_decoder
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if decoder_enabled:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            z_tgt_for_loss = z_tgt.detach() if self.stop_grad else z_tgt
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt_for_loss[:, :, :-2, :])
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt_for_loss[:, :, -2, :])
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt_for_loss[:, :, :-1, :])
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt_for_loss[:, :, :, :-(self.proprio_dim + self.action_dim)]
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
                    z_tgt_for_loss[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim]
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt_for_loss[:, :, :, :-self.action_dim]
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss

            if self.vcreg:
                z_vic_in = self.visual_prop(z)
                z_std_loss = self.vcreg_std_loss(z_vic_in)
                z_cov_loss = self.vcreg_cov_loss(z_vic_in)
                z_reg_loss = z_std_loss * self.std_coeff + z_cov_loss * self.cov_coeff
                loss_components["z_vicreg_std_loss"] = z_std_loss
                loss_components["z_vicreg_cov_loss"] = z_cov_loss
                loss_components["z_vcreg_loss_scaled"] = z_reg_loss
                loss = loss + z_reg_loss

            if self.straighten and self.straighten_scale > 0:
                feats = self.visual_only(z)
                curvature_loss = self.total_curvature(feats, mode=self.curvature_mode)
                loss = loss + curvature_loss * self.straighten_scale
                loss_components["curvature_loss_used_for_training"] = curvature_loss
        else:
            visual_pred = None
            z_pred = None

        if decoder_enabled:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z


    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z