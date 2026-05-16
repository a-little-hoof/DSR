import torch
from .utils import sign_max

def _primary_tensor(output):
  if torch.is_tensor(output):
    return output
  if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
    return output[0]
  return None

def log_internal(module, input, output, store):
  tensor = _primary_tensor(output)
  if tensor is None:
    return
  store.append(tensor.detach())
  # store.append(tensor.detach().cpu().numpy())

def replace_internal(module, input, output, new_value):
  if torch.is_tensor(output):
    return new_value
  if isinstance(output, tuple):
    if not output:
      return output
    return (new_value,) + output[1:]
  if isinstance(output, list):
    if output:
      output[0] = new_value
    return output
  return output

def apply_func_on_internal(module, input, output, func):
  if torch.is_tensor(output):
    return func(output)
  if isinstance(output, tuple):
    if not output:
      return output
    return (func(output[0]),) + output[1:]
  if isinstance(output, list):
    if output:
      output[0] = func(output[0])
    return output
  return output

def activate_on_registers(module, input, output, num_registers, neuron_indices, scale = 1.0, normal_values = "zero"):
  # print("USING NEW IMPLEMTATION")
  target = _primary_tensor(output)
  if target is None:
    return output
  # For all register neurons, set the activations of the extra registers to their max activation across patches
  patches = target[:, :, neuron_indices] # b, n, c
  pos_max = patches.amax(dim=1) # b,  c
  neg_max = patches.amin(dim=1)
  pos_max = pos_max.max(dim=1, keepdim=True).values # b,  c
  neg_max = neg_max.min(dim=1, keepdim=True).values
  signed_max = torch.where(pos_max.abs() > neg_max.abs(), pos_max, neg_max)
  # print("BUT FOLLOWING PAPER MAX", signed_max)
  if isinstance(scale, list):
    assert len(scale) == num_registers
    scale_tensor = torch.as_tensor(scale, device=target.device, dtype=target.dtype)
    if target[:, -num_registers:, neuron_indices].dim() == 2:
      target[:, -num_registers:, neuron_indices] = signed_max[:, None] * scale_tensor[None, :]
    else:
      target[:, -num_registers:, neuron_indices] = signed_max[:, None, :] * scale_tensor[None, :, None]
  else:
    if target[:, -num_registers:, neuron_indices].dim() == 2:
      target[:, -num_registers:, neuron_indices] = signed_max[:, None] * scale
    else:
      target[:, -num_registers:, neuron_indices] = signed_max[:, None, :] * scale
  if normal_values == "zero":
    # Set all image patch activations to 0
    target[:, 1:-num_registers, neuron_indices] = 0
  elif normal_values == "mean":
    # Set all image patch activations to the mean activation
    patch_activations = target[:, 1:-num_registers, neuron_indices]
    mean_activation = torch.mean(patch_activations, dim=1)  # Average across patches
    if patch_activations.dim() == 2:
      target[:, 1:-num_registers, neuron_indices] = mean_activation[:, None]
    else:
      target[:, 1:-num_registers, neuron_indices] = mean_activation[:, None, :].expand_as(patch_activations)
  elif normal_values == "only_outliers":
    # Set only the outliers within the image patches to the mean activation
    patch_activations = target[:, 1:-num_registers, neuron_indices].clone()

    # Calculate threshold for outliers (1 std above mean activation)
    means = torch.mean(patch_activations, dim=1)
    stds = torch.std(patch_activations, dim=1)
    outlier_thresholds = means + stds

    # Replace only outliers with the mean activation
    if patch_activations.dim() == 2:
      mask = patch_activations > outlier_thresholds[:, None]
      patch_activations[mask] = means[:, None].expand_as(patch_activations)[mask]
    else:
      mask = patch_activations > outlier_thresholds[:, None, :]
      patch_activations[mask] = means[:, None, :].expand_as(patch_activations)[mask]
    target[:, 1:-num_registers, neuron_indices] = patch_activations
  elif normal_values == "same":
    # Keep all the image patch activations the same
    pass
  else:
    raise ValueError(f"Invalid normal_values: {normal_values}")
  return target


# def activate_on_registers(module, input, output, num_registers, neuron_indices, scale = 1.0, normal_values = "zero"):
#   # print("USING OLD IMPLEMTATION")
#   # For all register neurons, set the activations of the extra registers to their max activation across patches
#   output = _primary_tensor(output)
#   # print(sign_max(output[0, :, neuron_indices]))
#   if isinstance(scale, list):
#     assert len(scale) == num_registers
#     for i in range(num_registers):
#       output[0, -num_registers + i, neuron_indices] = scale[i] * sign_max(output[0, :, neuron_indices])
#   else:
#     output[0, -num_registers:, neuron_indices] = scale * sign_max(output[0, :, neuron_indices]).unsqueeze(0).expand(num_registers, -1)
#   if normal_values == "zero":
#     # Set all image patch activations to 0
#     output[0, 1:-num_registers, neuron_indices] = 0
#   elif normal_values == "mean":
#     # Set all image patch activations to the mean activation
#     patch_activations = output[0, 1:-num_registers, neuron_indices]
#     mean_activation = torch.mean(patch_activations, dim=0)  # Average across patches
#     output[0, 1:-num_registers, neuron_indices] = mean_activation.unsqueeze(0).expand(output.shape[1] - num_registers - 1, -1)
#   elif normal_values == "only_outliers":
#     # Set only the outliers within the image patches to the mean activation
#     patch_activations = output[0, 1:-num_registers, neuron_indices].clone()

#     # Calculate threshold for outliers (1 std above mean activation)
#     means = torch.mean(patch_activations, dim=0)
#     stds = torch.std(patch_activations, dim=0)
#     outlier_thresholds = means + stds

#     # Replace only outliers with the mean activation
#     mask = patch_activations > outlier_thresholds.unsqueeze(0)
#     patch_activations[mask] = means.unsqueeze(0).expand_as(patch_activations)[mask]
#     output[0, 1:-num_registers, neuron_indices] = patch_activations
#   elif normal_values == "same":
#     # Keep all the image patch activations the same
#     pass
#   else:
#     raise ValueError(f"Invalid normal_values: {normal_values}")

