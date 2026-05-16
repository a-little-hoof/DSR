from transformers import Dinov2Model
from torch import nn
import torch
from math import *
from . import register_encoder


@register_encoder()
class Dinov2withNormNoReg(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
        layer_index: int = -1
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = Dinov2Model.from_pretrained(dinov2_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2Model.from_pretrained(dinov2_path, local_files_only=False)
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        self.layer_index = layer_index
        if layer_index != -1:
            print(f"We are using intermediate layer index: {layer_index}")
        
    def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, output_hidden_states=True)
        if self.layer_index == -1:
            selected_hidden_state = x.last_hidden_state
        else:
            # print(f'Among {len(x.hidden_states)} we are using {self.layer_index+1} feature')
            selected_hidden_state = x.hidden_states[self.layer_index]
        unused_token_num = 1  # 1 CLS 
        image_features = selected_hidden_state[:, unused_token_num:]
        return image_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov2_forward(x)
