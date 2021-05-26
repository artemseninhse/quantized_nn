import torch
import torch.nn as nn

# Code adapted from: https://github.com/jafermarq/WinogradAwareNets/blob/master/src/quantization.py

class MyRound(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
    
custom_round = MyRound.apply


class QuantRegularizer:
    
    def __init__(self,
                 scale,
                 zp,
                 qmin,
                 qmax,
                 quant_type
                 ):
        self.scale = scale
        self.zp = zp
        self.qmin = qmin
        self.qmax =qmax
        self.quant_type = quant_type

    def func_ste(self,
                 x):
        return quantize_static(x,
                               self.scale,
                               self.zp,
                               self.qmin,
                               self.qmax)

    def func_qsin(self,
                  x):
        below_ = (x <= self.qmin).int()
        above_ = (x >= self.qmax).int()
        bw_ = 1 - below_ - above_
        return torch.sin(pi_ * x) ** 2 * bw_ + \
                pi_ ** 2 * (x - self.qmin) ** 2 * below_ + \
                pi_ ** 2 * (x - self.qmax) ** 2 * above_

    def process_tensor(self,
                       x):
        quant_func = getattr(self, "_".join(["func", self.quant_type]))
        return torch.sum((x - quant_func(x)) ** 2)


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
    if isinstance(scale, torch.Tensor):
        scale = scale.item()
    output = x.clone()
    output.div_(scale).add_(zp)
    output = custom_round(output)
    output.clamp_(qmin, qmax)  # quantize
    output.add_(-zp).mul_(scale)  # dequantize

    return output

