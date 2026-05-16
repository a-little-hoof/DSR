import torch

def capture_block_last(model, layer_idx: int, device="cpu"):
    if layer_idx == 100:
        cache = {}
        def hook_fn(module, inputs, output):
            # output: [B, T, D] (或含 register tokens)
            cache["feat"] = output.detach()
            if device == "cpu":
                cache["feat"] = cache["feat"].float().cpu()
        handle = model.final_layer.register_forward_hook(hook_fn) 
        return cache, handle
    cache = {}
    
    def hook_fn(module, inputs, output):
        # output: [B, T, D] (或含 register tokens)
        cache["feat"] = output.detach()
        if device == "cpu":
            cache["feat"] = cache["feat"].float().cpu()

    handle = model.blocks[layer_idx].register_forward_hook(hook_fn)
    
    return cache, handle