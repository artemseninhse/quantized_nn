import torch
import torch.nn as nn

# Code adapted from: https://github.com/jafermarq/WinogradAwareNets/blob/master/src/quantization.py

def quantize_dynamic(x, qmin, qmax):
    output = x.clone()
    min_val = x.detach().min()
    max_val = x.detach().max()

    # compute qparams --> scale and zp
    max_val, min_val = float(max_val), float(min_val)
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)

    if max_val == min_val:
        scale = 1.0
        zp = 0
    else:
        max_range = max(-min_val, max_val) # largest mag(value)
        scale = max_range / ((qmax - qmin) / 2)
        scale = max(scale, 1e-8)
        zp = 0.0 # this true for symmetric quantization

    output.div_(scale).add_(zp)
    output.round_().clamp_(qmin, qmax)  # quantize
    output.add_(-zp).mul_(scale)  # dequantize

    return output

def quantize_static(x, scale, zp, qmin, qmax):
    
    output = x.clone()
    output.div_(scale).add_(zp)
    output.round_().clamp_(qmin, qmax)  # quantize
    output.add_(-zp).mul_(scale)  # dequantize

    return output
