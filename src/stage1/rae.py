import os
import torch
import torch.nn as nn
from .decoders import GeneralDecoder
from .encoders import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional
from math import sqrt
from typing import Protocol
from pathlib import Path

class Stage1Protocal(Protocol):
    # must have patch size attribute
    patch_size: int
    hidden_size: int 
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

def load_image_processor(encoder_config_path: str):
    """Load a HuggingFace image processor, optionally preferring a local cache.

    Set RAE_HF_CACHE to point at a pre-downloaded snapshot directory if you
    want offline / cluster-local loading; otherwise this defers to the normal
    HuggingFace hub download.
    """
    cache_env = os.environ.get("RAE_HF_CACHE")
    if cache_env:
        cache_base = Path(cache_env)
        try:
            if cache_base.exists():
                snap = sorted(cache_base.iterdir(), key=lambda p: p.stat().st_mtime)[-1]
                return AutoImageProcessor.from_pretrained(str(snap), local_files_only=True)
        except Exception as e:
            print(f"[warn] local snapshot load failed: {e}")

    try:
        return AutoImageProcessor.from_pretrained(encoder_config_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load image processor from '{encoder_config_path}' (and local cache).") from e


class RAE(nn.Module):
    def __init__(self, 
        # ---- encoder configs ----
        encoder_cls: str = 'Dinov2withNorm',
        encoder_config_path: str = 'facebook/dinov2-base',
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        # ---- decoder configs ----
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization-----
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        # ---- new options added ----
        norm_scaled_noise: bool = False,
        post_stat_norm: bool = False,
        preln: bool = False,
    ):
        super().__init__()
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
        print(f"encoder_config_path: {encoder_config_path}")
        if 'siglip' in encoder_config_path:
            proc = load_image_processor(encoder_config_path)
        else:
            proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        # encoder_config = AutoConfig.from_pretrained(encoder_config_path)
        # see if the encoder has patch size attribute            
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent
        
        # decoder
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches)) 
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        # load pretrained decoder weights
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")
        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.latent_mean = stats.get('mean', None)
            self.latent_var = stats.get('var', None)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False

        # Add new parameters
        self.norm_scaled_noise = norm_scaled_noise
        self.post_stat_norm = post_stat_norm
        self.preln = preln
        self.latent_flag = False
        self.latent_type = 0

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise
    def norm_scaled_noising(self, x, token_norm, clip=(0.20, 5.0), eps=1e-6):
        # token_norm should be broadcastable to x (e.g., [B,T,1] for [B,T,C] or [B,1,H,W] for [B,C,H,W])

        # compute per-sample median over all non-batch dims of token_norm
        B = token_norm.shape[0]
        tn_flat = token_norm.reshape(B, -1)
        med = tn_flat.median(dim=1, keepdim=True).values.reshape(B, *([1] * (token_norm.dim() - 1)))

        scale = token_norm / (med + eps)
        if clip is not None:
            scale = scale.clamp(*clip)

        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (x.dim() - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x) * scale
        return x + noise

    @torch.no_grad()
    def encode(self, x: torch.Tensor, output_token_norm=False) -> torch.Tensor:
        # normalize input
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        # z = self.encoder(x)
        if not self.latent_flag:
            try:
                z, latents = self.encoder(x)
                self.latent_flag = True
                self.latent_type = 0
            except:
                z = self.encoder(x)
                self.latent_flag = True
                self.latent_type = 1
        else:
            if self.latent_type == 0:
                z, latents = self.encoder(x)
            else:
                z = self.encoder(x)

        # import ipdb; ipdb.set_trace()
        
        # By default, we compute norm on tokens before added noise and before stat.
        # This means: 1. if preln=False and post_stat_norm=False, this token norm is a constant 
        # (always output norm=27) 2. if preln=True and post_stat_norm=False, this token norm would be
        # norm computed on preln feature without stat. 
        if self.preln:
            z = latents
            print("replacing z with preln feature")
        token_norm = torch.linalg.norm(z, ord=2, dim=-1, keepdim=True) # b, h*w, 1 
        
        if self.training and self.noise_tau > 0:
            if not self.post_stat_norm:
                if self.norm_scaled_noise:
                    z = self.norm_scaled_noising(z, token_norm)
                else:
                    z = self.noising(z)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
            token_norm = token_norm.view(b, h*w).view(b, 1, h*w).view(b, 1, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        if self.post_stat_norm: # This branch introduces norm added after stat. This is exactly input token norm.
            token_norm = torch.linalg.norm(z, ord=2, dim=1, keepdim=True) # b, 1, h, w
            if self.training and self.noise_tau > 0:
                if self.norm_scaled_noise:
                    z = self.norm_scaled_noising(z, token_norm)
                else:
                    z = self.noising(z)
        # tn = token_norm.detach().float().flatten(1)  
        # tn = torch.linalg.norm(latents, ord=2, dim=-1, keepdim=True) # b, h*w, 1 
        # # print("Token norm statistics:")
        # # print(f"tn shape: {tn.shape}")
        # # print(tn)
        # p001, p2, p50, p98, p99, p999 = torch.quantile(tn, torch.tensor([0.001, 0.02, 0.50, 0.98, 0.99, 0.999], device=tn.device), dim=1)  # each sample
        # print(f"token_norm p2(mean over batch)={p2.mean().item():.6g}, p50(mean over batch)={p50.mean().item():.6g}, p98(mean over batch)={p98.mean().item():.6g}")
        # print(f"token_norm p001(mean over batch)={p001.mean().item():.6g}, p99(mean over batch)={p99.mean().item():.6g}, p999(mean over batch)={p999.mean().item():.6g}")

        if output_token_norm:
            return z, token_norm
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        output = self.decoder(z, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        return x_rec
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec