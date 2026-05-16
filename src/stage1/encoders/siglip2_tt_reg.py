from torch import nn
import torch
from math import *
from pathlib import Path
import sys
import yaml
from typing import Any, Dict

try:
    from . import register_encoder
    from .siglip2_tt_reg import TestTimeRegSiglip2
except ImportError:
    _stage1_dir = Path(__file__).resolve().parent.parent
    if str(_stage1_dir) not in sys.path:
        sys.path.insert(0, str(_stage1_dir))
    from encoders import register_encoder
    from encoders.siglip2_tt_reg import TestTimeRegSiglip2


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@register_encoder()
class SigLIP2wNormTTREG(nn.Module):
    def __init__(self, model_name: str = "", num_tokens=256, config_path=None, image_size=256):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        if config_path is None:
            # Default to the in-repo SigLIP2 base config; the YAML configs in
            # configs/stage2/training/ImageNet256/*tt_reg* override this.
            config_path = Path(__file__).resolve().parent / "siglip2_tt_reg/configs/siglip2_base_with_max.yaml"
        config = _load_config(config_path)
        self.runner = TestTimeRegSiglip2(config=config, IMAGE_SIZE=image_size, device="cuda")
        self.unused_token_num = self.runner.num_registers
        self.runner.model.eval()
        self.runner.model.vision_model.post_layernorm.elementwise_affine = False
        self.runner.model.vision_model.post_layernorm.weight = None
        self.runner.model.vision_model.post_layernorm.bias = None
        vision_model = getattr(self.runner.model, "vision_model", None)
        if vision_model is not None and hasattr(vision_model, "config"):
            self.hidden_size = vision_model.config.hidden_size
        else:
            self.hidden_size = self.runner.model.config.hidden_size
        self.patch_size = self.runner.patch_size
    @torch.no_grad() # encoder is always frozen
    def forward(self, images):
        """
        images is of shape (B, C, H, W)
        where B is batch size, C is number of channels, H and W are height and
        """
        # # outputs = self.model(images, output_hidden_states=True, interpolate_pos_encoding = True)
        # # image_features = outputs.last_hidden_state
        # # return image_features
        # # import ipdb; ipdb.set_trace()
        # outputs = self.runner(images, use_registers=True)
        # post_ln = outputs.last_hidden_state
        # hidden_states = getattr(outputs, "hidden_states", None)
        # pre_ln = hidden_states[-1] if hidden_states else post_ln
        # hook_hidden_states = self.runner.hook_manager.get_layer_outputs() # 768, 259, 768
        # # print(f"shape of hook_hidden_states: {hook_hidden_states} layers")
        # import ipdb; ipdb.set_trace()
        # for hook_hidden_state in hook_hidden_states:
        #     print(f"hook_hidden_state shape: {hook_hidden_state.shape}")
        #     hook_hidden_state_wo_reg = torch.tensor(hook_hidden_state[:-1, :], device=post_ln.device)
        #     norms = hook_hidden_state_wo_reg.norm(dim=-1)
        #     print(f"max norm (no reg): {norms.max().item()}")

        # import ipdb; ipdb.set_trace()
        # pre_ln = hook_hidden_states[-1]
        # # to tensor
        # pre_ln = torch.tensor(pre_ln, device=post_ln.device)
        # # if hidden_states and not getattr(self, "_norm_term_checked", False):
        # #     normed = torch.nn.functional.layer_norm(
        # #         pre_ln,
        # #         pre_ln.shape[-1:],
        # #         eps=self.runner.model.vision_model.post_layernorm.eps,
        # #     )
        # #     self._norm_term_matches = torch.allclose(post_ln, normed, rtol=1e-4, atol=1e-5)
        # #     self._norm_term_checked = True
        # # print(pre_ln.shape,post_ln.shape)
        # unused_token_num=1
        # post_ln = post_ln[:, :-unused_token_num]
        # pre_ln = pre_ln[:, :-unused_token_num]
        # return post_ln, pre_ln

        outputs = self.runner(images, use_registers=True)
        post_ln = outputs.last_hidden_state
        hidden_states = getattr(outputs, "hidden_states", None)
        pre_ln = hidden_states[-1] if hidden_states else post_ln
        unused_token_num=self.unused_token_num
        post_ln = post_ln[:, :-unused_token_num]
        pre_ln = pre_ln[:, :-unused_token_num]
        return post_ln, pre_ln
    

if __name__ == '__main__':
    encoder = SigLIP2wNormTTREG()
    
    image = torch.zeros([1, 3, 256, 256]).cuda()
    print(encoder(image)[0].shape)
    print(encoder._norm_term_checked)
    print(encoder._norm_term_matches)
