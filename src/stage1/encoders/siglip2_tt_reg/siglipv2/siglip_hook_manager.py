import numpy as np
from shared.hook_manager import HookManager
import torch

class Siglip2VisionHookManager(HookManager):
    def __init__(self, model):
        super().__init__(model)
        self.vit = self._resolve_vit(model)  # Siglip2VisionTransformer
        self.layers = self.vit.encoder.layers
        self._validate_hook_points()

    def _validate_hook_points(self):
        if len(self.layers) == 0:
            raise AttributeError("Siglip2 vision model does not expose encoder layers.")
        layer = self.layers[0]
        if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
            raise AttributeError("Siglip2 vision layers must expose self_attn and mlp modules.")
        attn = layer.self_attn
        mlp = layer.mlp
        if not hasattr(attn, "pre_softmax_identity") or not hasattr(attn, "post_softmax_identity"):
            raise AttributeError(
                "Siglip2 attention hooks missing; call _patch_siglip2_for_hooks before creating the hook manager."
            )
        if not hasattr(mlp, "activation_identity"):
            raise AttributeError(
                "Siglip2 MLP hooks missing; call _patch_siglip2_for_hooks before creating the hook manager."
            )

    def _resolve_vit(self, model):
        # model could be Siglip2VisionModel (has .vision_model) or Siglip2Model (has .vision_model too),
        # but in both cases: the ViT module is usually at model.vision_model
        if hasattr(model, "vision_model") and hasattr(model.vision_model, "encoder"):
            return model.vision_model  # Siglip2VisionTransformer
        if hasattr(model, "vision_model") and hasattr(model.vision_model, "vision_model"):
            # Siglip2VisionModel wrapper: model.vision_model is Siglip2VisionTransformer already in your pasted code,
            # but keep a fallback anyway
            return model.vision_model.vision_model
        raise AttributeError("Cannot resolve Siglip2VisionTransformer from the provided model.")

    def num_layers(self):
        return len(self.layers)

    # --- Attention hook points (after you patch attention to expose these identities) ---
    def attn_pre_softmax_component(self, layer):
        return self.layers[layer].self_attn.pre_softmax_identity

    def attn_post_softmax_component(self, layer):
        return self.layers[layer].self_attn.post_softmax_identity

    # --- Whole layer output ---
    def layer_output_component(self, layer):
        return self.layers[layer]

    # --- MLP activation hook point (after you patch MLP to expose an identity) ---
    def neuron_activation_component(self, layer):
        return self.layers[layer].mlp.activation_identity

    def get_neuron_activations(self):
        neuron_activations = super().get_neuron_activations()
        if neuron_activations is None or neuron_activations.ndim != 3:
            return neuron_activations
        cls_token = neuron_activations.mean(axis=1, keepdims=True)
        # return np.concatenate([cls_token, neuron_activations], axis=1)
        return torch.cat([cls_token, neuron_activations], axis=1)

    def get_attention_maps(self):
        # todo: this get_attention_maps returns None, fix it
        attention_maps = super().get_attention_maps()
        if attention_maps is None or attention_maps.ndim != 4:
            return attention_maps
        cls_row = attention_maps.mean(axis=2, keepdims=True)
        cls_col = attention_maps.mean(axis=3, keepdims=True)
        cls_cls = cls_row.mean(axis=3, keepdims=True)
        # top = np.concatenate([cls_cls, cls_row], axis=3)
        # bottom = np.concatenate([cls_col, attention_maps], axis=3)
        # return np.concatenate([top, bottom], axis=2)
        top = torch.cat([cls_cls, cls_row], axis=3)
        bottom = torch.cat([cls_col, attention_maps], axis=3)
        return torch.cat([top, bottom], axis=2)

    def get_layer_outputs(self):
        layer_outputs = super().get_layer_outputs()
        if layer_outputs is None or layer_outputs.ndim != 3:
            return layer_outputs
        cls_token = layer_outputs.mean(axis=1, keepdims=True)
        # return np.concatenate([cls_token, layer_outputs], axis=1)
        return torch.cat([cls_token, layer_outputs], axis=1)
