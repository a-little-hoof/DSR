from abc import ABC
import numpy as np
from .hook_fn import activate_on_registers, log_internal, apply_func_on_internal
from functools import partial
import torch
from enum import Enum

class HookMode(Enum):
  ANALYSIS = "analysis"
  INTERVENE = "intervene"


class HookManager(ABC):
  def __init__(self, model):
    self.model = model
    self.mode = HookMode.ANALYSIS

    # Hooks
    self.hooks = {
      "log_layer_outputs": [],
      "log_attention_outputs": [],
      "log_neuron_activations": [],
      "log_attention_maps": [],
      "intervene_register_neurons": [],
      "intervene_layer_outputs": [],
      "intervene_attn_pre_softmax": [],
    }

    self.register_neurons_intervention = None
    self.layer_output_intervention = None
    self.attn_pre_softmax_intervention = None

    # Logs of internal model data
    self.logs = {
      "attention_maps": [],
      "attention_outputs": [],
      "layer_outputs": [],
      "neuron_activations": [],
    }

  def attn_post_softmax_component(self, layer):
    raise NotImplementedError("This method should be overridden in a subclass")

  def layer_output_component(self, layer):
    raise NotImplementedError("This method should be overridden in a subclass")

  def neuron_activation_component(self, layer):
    raise NotImplementedError("This method should be overridden in a subclass")

  def num_layers(self):
    raise NotImplementedError("This method should be overridden in a subclass")

  """
  Optional: override this method to get / intervene upon attention maps before softmax
  """
  def attn_pre_softmax_component(self, layer):
    return None

  """
  Optional: override this method to get / intervene upon attention outputs
  """
  def attn_output_component(self, layer):
    return None

  def initialize_log_hooks(self):
    for layer in range(self.num_layers()):
      # Attention maps
      log_attention_hook = self.attn_post_softmax_component(layer).register_forward_hook(partial(log_internal, store = self.logs["attention_maps"]))
      self.hooks["log_attention_maps"].append(log_attention_hook)

      # Attention outputs
      if self.attn_output_component(layer) is not None:
        log_attention_output_hook = self.attn_output_component(layer).register_forward_hook(partial(log_internal, store = self.logs["attention_outputs"]))
        self.hooks["log_attention_outputs"].append(log_attention_output_hook)

      # Layer outputs
      log_layer_output_hook = self.layer_output_component(layer).register_forward_hook(partial(log_internal, store = self.logs["layer_outputs"]))
      self.hooks["log_layer_outputs"].append(log_layer_output_hook)

      # Neuron activations
      log_neuron_activations_hook = self.neuron_activation_component(layer).register_forward_hook(partial(log_internal, store = self.logs["neuron_activations"]))
      self.hooks["log_neuron_activations"].append(log_neuron_activations_hook)

  def reinit(self, mode = HookMode.ANALYSIS):
    self.mode = mode

    # Remove all hooks
    self.unregister_all_hooks()
    self.hooks["intervene_register_neurons"] = []
    # Clear all logs
    for log_name in self.logs:
      self.logs[log_name].clear()

    # Clear all intervention data
    self.register_neurons_intervention = None
    self.layer_output_intervention = None
    self.attn_pre_softmax_intervention = None

  def finalize(self):
    # Initialize intervention hooks
    if self.register_neurons_intervention is not None:
      register_neurons = self.register_neurons_intervention["neurons_to_ablate"]
      for layer in register_neurons:
        intervene_register_neurons_fn = partial(
            activate_on_registers,
            num_registers=self.register_neurons_intervention["num_registers"],
            neuron_indices=register_neurons[layer],
            scale=self.register_neurons_intervention["scale"],
            normal_values=self.register_neurons_intervention["normal_values"],
        )

        intervene_register_neurons_hook = self.neuron_activation_component(layer).register_forward_hook(intervene_register_neurons_fn)
        self.hooks["intervene_register_neurons"].append(intervene_register_neurons_hook)

    # if self.layer_output_intervention is not None:
    #   for i, layer in enumerate(self.layer_output_intervention["layers"]):
    #     intervene_layer_output_fn = partial(apply_func_on_internal, func = self.layer_output_intervention["func"][i])
    #     intervene_layer_output_hook = self.layer_output_component(layer).register_forward_hook(intervene_layer_output_fn)
    #     self.hooks["intervene_layer_outputs"].append(intervene_layer_output_hook)
    # if self.attn_pre_softmax_intervention is not None and self.attn_pre_softmax_component(0) is not None:
    #   for i, layer in enumerate(self.attn_pre_softmax_intervention["layers"]):
    #     intervene_attn_pre_softmax_fn = partial(apply_func_on_internal, func = self.attn_pre_softmax_intervention["func"][i])
    #     intervene_attn_pre_softmax_hook = self.attn_pre_softmax_component(layer).register_forward_hook(intervene_attn_pre_softmax_fn)
    #     self.hooks["intervene_attn_pre_softmax"].append(intervene_attn_pre_softmax_hook)

    # # Initialize log hooks
    # if self.mode == HookMode.ANALYSIS:
    #   self.initialize_log_hooks()

  def finalize_zero_out(self, layer_to_subtract = 0, zero_noise=False):
    zero_cache = {}
    in_progress = {"active": False}
    suppress_logs = {"active": False}

    def _log_internal_guarded(module, input, output, store):
      if suppress_logs["active"]:
        return
      log_internal(module, input, output, store)

    def _reduce_first_block_by_zero_input(module, input, output):
      if in_progress["active"]:
        return output
      if torch.is_tensor(output):
        target = output
        output_kind = "tensor"
      elif isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
          return output
        target = output[0]
        output_kind = "tuple"
      elif isinstance(output, list):
        if not output or not torch.is_tensor(output[0]):
          return output
        target = output[0]
        output_kind = "list"
      else:
        return output

      key = (target.device, target.dtype, target.shape)
      baseline = zero_cache.get(key)
      if baseline is None:
        if not input or not torch.is_tensor(input[0]):
          return output
        zeroed = list(input)
        
        if zero_noise:
          zeroed[0] = torch.empty_like(input[0]).uniform_(-1.0, 1.0)
        else:
          zeroed[0] = torch.zeros_like(input[0])
        in_progress["active"] = True
        suppress_logs["active"] = True
        try:
          with torch.no_grad():
            baseline_out = module.forward(*zeroed)
        finally:
          suppress_logs["active"] = False
          in_progress["active"] = False
        if torch.is_tensor(baseline_out):
          baseline = baseline_out.detach()
        elif isinstance(baseline_out, (tuple, list)) and baseline_out and torch.is_tensor(baseline_out[0]):
          baseline = baseline_out[0].detach()
        else:
          return output
        zero_cache[key] = baseline

      if output_kind == "tensor":
        return output - baseline
      if output_kind == "tuple":
        return (output[0] - baseline,) + output[1:]
      if output_kind == "list":
        output = list(output)
        output[0] = output[0] - baseline
        return output
      return output

    target_layer = int(layer_to_subtract)
    target_layer = max(0, min(target_layer, self.num_layers() - 1))
    layer_hook = self.layer_output_component(target_layer).register_forward_hook(_reduce_first_block_by_zero_input)
    self.hooks["intervene_layer_outputs"].append(layer_hook)
    # Initialize log hooks
    if self.mode == HookMode.ANALYSIS:
      for layer in range(self.num_layers()):
        log_attention_hook = self.attn_post_softmax_component(layer).register_forward_hook(
            partial(_log_internal_guarded, store = self.logs["attention_maps"])
        )
        self.hooks["log_attention_maps"].append(log_attention_hook)

        if self.attn_output_component(layer) is not None:
          log_attention_output_hook = self.attn_output_component(layer).register_forward_hook(
              partial(_log_internal_guarded, store = self.logs["attention_outputs"])
          )
          self.hooks["log_attention_outputs"].append(log_attention_output_hook)

        log_layer_output_hook = self.layer_output_component(layer).register_forward_hook(
            partial(_log_internal_guarded, store = self.logs["layer_outputs"])
        )
        self.hooks["log_layer_outputs"].append(log_layer_output_hook)

        log_neuron_activations_hook = self.neuron_activation_component(layer).register_forward_hook(
            partial(_log_internal_guarded, store = self.logs["neuron_activations"])
        )
        self.hooks["log_neuron_activations"].append(log_neuron_activations_hook)


  def unregister_hook(self, hook):
    hook.remove()

  def unregister_all_hooks(self):
    for hook_name in self.hooks:
      if isinstance(self.hooks[hook_name], list):
        for hook in self.hooks[hook_name]:
          self.unregister_hook(hook)
      elif isinstance(self.hooks[hook_name], dict):
        for key in self.hooks[hook_name]:
          self.unregister_hook(self.hooks[hook_name][key])
      else:
        raise ValueError(f"Invalid type for hook name: {type(self.hooks[hook_name])}")

  def get_attention_outputs(self):
    assert self.mode == HookMode.ANALYSIS, "Analysis mode must be enabled to get attention outputs"
    if not self.logs["attention_outputs"]:
      return None
    # return np.concatenate(self.logs["attention_outputs"], axis=0)
    return torch.cat(self.logs["attention_outputs"], axis=0)

  def get_attention_maps(self):
    assert self.mode == HookMode.ANALYSIS, "Analysis mode must be enabled to get attention maps"
    if not self.logs["attention_maps"]:
      return None
    # return np.concatenate(self.logs["attention_maps"], axis=0)
    return torch.cat(self.logs["attention_maps"], axis=0)

  def get_layer_outputs(self):
    assert self.mode == HookMode.ANALYSIS, "Analysis mode must be enabled to get layer outputs"
    if not self.logs["layer_outputs"]:
      return None
    # return np.concatenate(self.logs["layer_outputs"], axis=0)
    return torch.cat(self.logs["layer_outputs"], axis=0)

  def get_neuron_activations(self):
    assert self.mode == HookMode.ANALYSIS, "Analysis mode must be enabled to get neuron activations"
    if not self.logs["neuron_activations"]:
      return None
    # return np.concatenate(self.logs["neuron_activations"], axis=0)
    return torch.cat(self.logs["neuron_activations"], axis=0)

  def set_mode(self, mode):
    self.mode = mode

  def intervene_register_neurons(self, num_registers, neurons_to_ablate, scale = 1.0, normal_values = "zero"):
    self.register_neurons_intervention = {
      "neurons_to_ablate": neurons_to_ablate,
      "num_registers": num_registers,
      "scale": scale,
      "normal_values": normal_values,
    }

  def intervene_attn_pre_softmax(self, layers, funcs):
    assert self.attn_pre_softmax_component(0) is not None, "Attn pre softmax component is not defined"
    self.attn_pre_softmax_intervention = {
      "layers": layers,
      "func": funcs,
    }

  def intervene_layer_output(self, layers, funcs):
    self.layer_output_intervention = {
      "layers": layers,
      "func": funcs,
    }
