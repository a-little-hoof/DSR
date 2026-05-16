from transformers import DINOv3ViTModel
from torch import nn
import torch
from math import *
from . import register_encoder


@register_encoder()
class Dinov3withNorm(nn.Module):
    def __init__(
        self,
        dinov3_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = DINOv3ViTModel.from_pretrained(dinov3_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = DINOv3ViTModel.from_pretrained(dinov3_path, local_files_only=False)
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.norm.elementwise_affine = False
            self.encoder.norm.weight = None
            self.encoder.norm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        print('Check following parameters')
        print(self.encoder.config.num_register_tokens)
        print(self.encoder.config.hidden_size)

        
    def dinov3_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        assert x.last_hidden_state.shape[1] == 256+unused_token_num
        image_features = x.last_hidden_state[:, unused_token_num:]
        return image_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov3_forward(x)
