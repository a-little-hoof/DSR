import inspect
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from .siglip_hook_manager import Siglip2VisionHookManager
from pathlib import Path

def _resolve_vision_model(model):
  if hasattr(model, "vision_model"):
    return model.vision_model
  return model

def _maybe_local_cache(encoder_config_path: str):
    """Return a Path to a local HF snapshot if RAE_HF_CACHE is set, else None."""
    cache_env = os.environ.get("RAE_HF_CACHE")
    if cache_env and 'so' not in encoder_config_path:
        return Path(cache_env)
    return None


def load_image_processor(encoder_config_path: str):
    cache_base = _maybe_local_cache(encoder_config_path)
    if cache_base is not None:
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


def load_image_model(encoder_config_path: str):
    cache_base = _maybe_local_cache(encoder_config_path)
    if cache_base is not None:
        try:
            if cache_base.exists():
                snap = sorted(cache_base.iterdir(), key=lambda p: p.stat().st_mtime)[-1]
                return AutoModel.from_pretrained(str(snap), local_files_only=True)
        except Exception as e:
            print(f"[warn] local snapshot load failed: {e}")
    try:
        return AutoModel.from_pretrained(encoder_config_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load image processor from '{encoder_config_path}' (and local cache).") from e



def _get_layers(model):
  vision_model = _resolve_vision_model(model)
  if hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layers"):
    return vision_model.encoder.layers
  raise AttributeError("Siglip2 vision model does not expose encoder layers.")

def _patch_attention_for_hooks(attn):
  if not hasattr(attn, "pre_softmax_identity"):
    attn.pre_softmax_identity = nn.Identity()
  if not hasattr(attn, "post_softmax_identity"):
    attn.post_softmax_identity = nn.Identity()
  if getattr(attn, "_ttreg_patched", False):
    return
  original_forward = attn.forward

  try:
    param_names = tuple(inspect.signature(attn.forward).parameters.keys())
  except (TypeError, ValueError):
    param_names = ()
  name_to_index = {name: i for i, name in enumerate(param_names)}
  q_proj = getattr(attn, "q_proj", None) or getattr(attn, "query", None)
  k_proj = getattr(attn, "k_proj", None) or getattr(attn, "key", None)
  v_proj = getattr(attn, "v_proj", None) or getattr(attn, "value", None)
  out_proj = (
    getattr(attn, "out_proj", None)
    or getattr(attn, "proj", None)
    or getattr(attn, "o_proj", None)
  )
  num_heads = getattr(attn, "num_heads", None) or getattr(attn, "num_attention_heads", None)
  head_dim = getattr(attn, "head_dim", None)
  if head_dim is None and q_proj is not None and num_heads:
    head_dim = q_proj.out_features // num_heads
  scale = getattr(attn, "scale", None) or getattr(attn, "scaling", None)
  if scale is None and head_dim:
    scale = 1.0 / math.sqrt(head_dim)
  dropout = getattr(attn, "dropout", None)
  if dropout is None:
    dropout = getattr(attn, "attention_dropout", 0.0)

  def _get_arg(name, default=None, args=None, kwargs=None):
    idx = name_to_index.get(name)
    if idx is not None and args is not None and idx < len(args):
      return args[idx]
    if kwargs is not None and name in kwargs:
      return kwargs[name]
    return default

  def wrapped_forward(*args, **kwargs):
    hidden_states = _get_arg("hidden_states", args=args, kwargs=kwargs)
    if hidden_states is None:
      hidden_states = args[0]
    attention_mask = _get_arg("attention_mask", default=None, args=args, kwargs=kwargs)

    if (
      q_proj is None
      or k_proj is None
      or v_proj is None
      or out_proj is None
      or num_heads is None
      or head_dim is None
    ):
      return original_forward(*args, **kwargs)

    bsz, tgt_len, _ = hidden_states.size()
    query_states = q_proj(hidden_states)
    key_states = k_proj(hidden_states)
    value_states = v_proj(hidden_states)
    query_states = query_states.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, -1, num_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, -1, num_heads, head_dim).transpose(1, 2)

    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
    if scale is not None:
      attn_weights = attn_weights * scale
    if attention_mask is not None:
      attn_weights = attn_weights + attention_mask
    attn_weights = attn.pre_softmax_identity(attn_weights)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = attn.post_softmax_identity(attn_weights)

    dropout_p = 0.0 if not attn.training else dropout
    if isinstance(dropout, nn.Dropout):
      attn_probs = dropout(attn_weights)
    elif isinstance(dropout_p, float) and dropout_p > 0.0:
      attn_probs = F.dropout(attn_weights, p=dropout_p, training=attn.training)
    else:
      attn_probs = attn_weights

    attn_output = torch.matmul(attn_probs, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, -1)
    attn_output = out_proj(attn_output)
    return attn_output, attn_weights

  attn.forward = wrapped_forward
  attn._ttreg_patched = True

def _patch_mlp_for_hooks(mlp):
  if not hasattr(mlp, "activation_identity"):
    mlp.activation_identity = nn.Identity()
  if getattr(mlp, "_ttreg_patched", False):
    return
  for attr in ("activation_fn", "activation", "act", "gelu"):
    if hasattr(mlp, attr):
      act = getattr(mlp, attr)
      if isinstance(act, nn.Module):
        setattr(mlp, attr, nn.Sequential(act, mlp.activation_identity))
      else:
        def wrapped(x, act=act, mlp=mlp):
          return mlp.activation_identity(act(x))
        setattr(mlp, attr, wrapped)
      mlp._ttreg_patched = True
      return
  def _fallback_hook(module, inputs, output):
    module.activation_identity(output)
  mlp.register_forward_hook(_fallback_hook)
  mlp._ttreg_patched = True

def _patch_siglip2_for_hooks(model):
  for layer in _get_layers(model):
    if hasattr(layer, "self_attn"):
      _patch_attention_for_hooks(layer.self_attn)
    if hasattr(layer, "mlp"):
      _patch_mlp_for_hooks(layer.mlp)

def _patch_register_tokens_for_forward(vision_model):
  if getattr(vision_model, "_ttreg_registers_patched", False):
    return
  original_forward = vision_model.forward
  if not hasattr(vision_model, "num_register_tokens"):
    vision_model.num_register_tokens = 0

  def wrapped_forward(pixel_values, interpolate_pos_encoding: bool = False, **kwargs):
    num_registers = getattr(vision_model, "num_register_tokens", 0)
    if num_registers <= 0:
      return original_forward(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, **kwargs)

    hidden_states = vision_model.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
    bsz, _, dim = hidden_states.shape
    zeros = hidden_states.new_zeros((bsz, num_registers, dim))
    # cls_token = zeros[:, :1, :]
    register_tokens = zeros
    hidden_states = torch.cat((hidden_states, register_tokens), dim=1)

    encoder_outputs = vision_model.encoder(
      inputs_embeds=hidden_states,
      **kwargs,
    )
    last_hidden_state = vision_model.post_layernorm(encoder_outputs.last_hidden_state)
    pooler_output = vision_model.head(last_hidden_state) if vision_model.use_head else None

    return BaseModelOutputWithPooling(
      last_hidden_state=last_hidden_state,
      pooler_output=pooler_output,
      hidden_states=getattr(encoder_outputs, "hidden_states", None),
      attentions=getattr(encoder_outputs, "attentions", None),
    )

  vision_model.forward = wrapped_forward
  vision_model._ttreg_registers_patched = True

# todo: read test-time-registers/clip/clip_state.py and decide what to do when is not None; you might want to reimpelment the forward function to support adding some registers or any other functions.
def run_model(model, image, num_registers = None):
  vision_model = _resolve_vision_model(model)
  if num_registers is not None:
    prev_num_registers = getattr(vision_model, "num_register_tokens", 0)
    vision_model.num_register_tokens = num_registers
  with torch.no_grad():
    forward_kwargs = {
      "pixel_values": image,
      "output_hidden_states": True,
    }
    try:
      if "interpolate_pos_encoding" in inspect.signature(vision_model.forward).parameters:
        forward_kwargs["interpolate_pos_encoding"] = True
    except (TypeError, ValueError):
      pass
    outputs = vision_model(**forward_kwargs)
  if num_registers is not None:
    vision_model.num_register_tokens = prev_num_registers
  return outputs

def get_num_neurons_per_mlp(model):
  mlp = _get_layers(model)[0].mlp
  for attr in ("fc1", "dense_h_to_4h", "intermediate_dense", "c_fc", "gate_proj", "w1"):
    layer = getattr(mlp, attr, None)
    if layer is not None and hasattr(layer, "out_features"):
      return layer.out_features
  if hasattr(model, "config") and hasattr(model.config, "intermediate_size"):
    return model.config.intermediate_size
  raise AttributeError("Cannot determine MLP hidden size for Siglip2 model.")

def get_num_layers(model):
  return len(_get_layers(model))

def get_num_heads(model):
  attn = _get_layers(model)[0].self_attn
  for attr in ("num_heads", "num_attention_heads", "n_heads"):
    if hasattr(attn, attr):
      return getattr(attn, attr)
  if hasattr(model, "config") and hasattr(model.config, "num_attention_heads"):
    return model.config.num_attention_heads
  raise AttributeError("Cannot determine attention head count for Siglip2 model.")

def load_siglip_state(config):
  model_id = config.get("model_id", config.get("model_name"))
  device = config.get("device", "cpu")

  # image_processor = AutoImageProcessor.from_pretrained(model_id)
  image_processor = load_image_processor(model_id)

  # model = AutoModel.from_pretrained(model_id)
  model = load_image_model(model_id)
  print(model_id)
  # model = load_image_model.from_pretrained(model_id)
  model.to(device)
  model.eval()

  _patch_siglip2_for_hooks(model)
  _patch_register_tokens_for_forward(_resolve_vision_model(model))

  vision_model = _resolve_vision_model(model)
  num_heads = get_num_heads(model)
  num_layers = get_num_layers(model)
  print(num_layers)
  num_neurons_per_layer = get_num_neurons_per_mlp(model)

  def _ensure_three_channels(img):
    if isinstance(img, torch.Tensor):
      if img.ndim == 2:
        return img.unsqueeze(0).repeat(3, 1, 1)
      return img
    if isinstance(img, np.ndarray):
      if img.ndim == 2:
        img = np.expand_dims(img, axis=0)
        return np.repeat(img, 3, axis=0)
      return img
    if isinstance(img, Image.Image):
      return img.convert("RGB")
    return img

  def preprocess(image, interpolate=False):
    if isinstance(image, (list, tuple)):
      image = [_ensure_three_channels(img) for img in image]
    else:
      image = _ensure_three_channels(image)
    if interpolate:
      # 1) Use the HF image_processor to do the usual stuff (RGB/normalize/to tensor),
      #    but force its resize to 256x256 first.
      processed = image_processor(
          images=image,
          return_tensors="pt",
          size={"height": 256, "width": 256},
      )

      # 2) Then interpolate once more to the encoder’s expected size (e.g., 224x224).
      x = processed["pixel_values"]  # [B, 3, 256, 256]
      x = F.interpolate(
          x,
          size=(224, 224),  # (224, 224)
          mode="bicubic",
          align_corners=False,
      )
      processed["pixel_values"] = x
      # 3) modify image_processor size back to 224,224
    else:
      processed = image_processor(images=image, return_tensors="pt")
    return processed["pixel_values"][0]

  hook_manager = Siglip2VisionHookManager(model)
  patch_size = getattr(vision_model, "patch_size", None)
  if patch_size is None and hasattr(vision_model, "config"):
    patch_size = vision_model.config.patch_size

  return dict(
    config=config,
    model=model,
    preprocess=preprocess,
    num_heads=num_heads,
    num_layers=num_layers,
    num_neurons_per_layer=num_neurons_per_layer,
    patch_size=patch_size,
    run_model=run_model,
    hook_manager=hook_manager,
  )
