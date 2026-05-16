from typing import Any, Dict, Optional, Sequence

from pathlib import Path
import sys
import json

import torch
import torch.nn as nn

_pkg_dir = Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from shared.algorithms import find_register_neurons
from shared.utils import filter_layers
from siglipv2.siglip_state import load_siglip_state

class TestTimeRegSiglip2(nn.Module):
    """Combined interface exposing test-time registers on Siglip2."""

    def __init__(self, config: Dict[str, Any], IMAGE_SIZE: int = 224, device: str = "cuda:0"):
        super().__init__()
        self.state = load_siglip_state(config)
        self.run_model = self.state["run_model"]
        self.model = self.state["model"]
        self.preprocess = self.state["preprocess"] # Preprocess function for input images
        self.hook_manager = self.state["hook_manager"]
        self.num_layers = self.state["num_layers"]
        self.num_heads = self.state["num_heads"]
        self.patch_size = self.state["patch_size"]
        self.config = self.state["config"]
        self.patch_height = IMAGE_SIZE // self.patch_size
        self.patch_width = IMAGE_SIZE // self.patch_size
        self.device = device

        # parameters for register neuron discovery
        self.register_norm_threshold = config.get("register_norm_threshold", 30.0)
        self.detect_outliers_layer = config.get("detect_outliers_layer", -1)
        self.highest_layer = config.get("highest_layer", None)
        self.top_k = config.get("top_k", None)
        self.num_registers = config.get("num_registers", 1)
        print("reg_num", self.num_registers)

        neurons_to_ablate = config.get("neurons_to_ablate", config.get("neurinos_to_ablate"))
        if isinstance(neurons_to_ablate, (str, Path)):
            print('loading from ', neurons_to_ablate)
            with open(neurons_to_ablate, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            neurons_to_ablate = {int(layer): indices for layer, indices in loaded.items()}

        self.neurons_to_ablate = neurons_to_ablate
        print(f"identified neurons to ablate: {self.neurons_to_ablate}")
    
    def find_register_neurons(
        self,
        image_path: str,
        processed_image_cnt: int = 500,
        apply_sparsity_filter: bool = False,
    ):
        """Find register neurons using the shared algorithm and current model state."""
        return find_register_neurons(
            model_state=self.state,
            image_path=image_path,
            register_norm_threshold=self.register_norm_threshold,
            detect_outliers_layer=self.detect_outliers_layer,
            processed_image_cnt=processed_image_cnt,
            apply_sparsity_filter=apply_sparsity_filter,

        )

    def filter_register_neurons(
        self,
        highest_layer: Optional[int] = None,
    ):
        """Filter register neurons to a manageable number."""
        return filter_layers(
            self.register_neurons,
            highest_layer=highest_layer,
        )
    
    def get_neurons_to_ablate(
        self,
    ):
        """Get neurons to ablate based on filtered register neurons."""
        neurons_to_ablate: Dict[int, Sequence[int]] = {}
        candidates = self.filtered_register_neurons
        if self.top_k is not None:
            candidates = candidates[: self.top_k]
        for neuron_info in candidates:
            layer, neuron_idx, *_ = neuron_info
            neurons_to_ablate.setdefault(layer, []).append(neuron_idx)
        return neurons_to_ablate

    def forward(
        self,
        images,
        use_registers: bool = False,
        normal_values: str = "zero",
        scale: float = 1.0,
    ):
        """Run the model with optional register intervention embedded in the forward pass."""
        # import ipdb; ipdb.set_trace()
        num_registers = None
        self.hook_manager.reinit()
        if use_registers:
            num_registers = self.num_registers
            self.hook_manager.intervene_register_neurons(
                neurons_to_ablate=self.neurons_to_ablate,
                num_registers=num_registers,
                normal_values=normal_values,
                scale=scale,
            )
        self.hook_manager.finalize()
        return self.run_model(self.model, images, num_registers=num_registers)
    
    def hook_init(self, normal_values: str = "zero", scale: float = 1.0):
        """Initialize hooks for test-time registration."""
        self.hook_manager.reinit()
        self.hook_manager.intervene_register_neurons(
            neurons_to_ablate=self.neurons_to_ablate,
            num_registers=self.num_registers,
            normal_values=normal_values,
            scale=scale,
        )
        self.hook_manager.finalize()
        return self.num_registers
