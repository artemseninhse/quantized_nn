import torch
import torch.nn as nn


class SimpleTextCNN(nn.Module):
    
    def __init__(self,
                 num_classes,
                 max_len,
                 vocab_size,
                 ksize_min,
                 ksize_max,
                 in_channels,
                 out_channels,
                 pool_size,
                 custom_layers={}):
        super(SimpleTextCNN, self).__init__()
        assert ksize_min > 1 and \
                ksize_max > ksize_min and \
                ksize_max < max_len, \
                "kernel size must exceed 1 and be less than maximum length"
        assert isinstance(custom_layers, dict), "custom layers must be a dict"
        self.max_len = max_len
        self.ksize_min = ksize_min
        self.ksize_max = ksize_max
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_size = pool_size
        self.vocab_size = vocab_size
        self.in_linear = sum(((self.max_len + 1 - ksize) // self.pool_size 
                              for ksize in range(self.ksize_min, self.ksize_max+1)))
        self.out_linear = 1 if num_classes == 2 else num_classes
        self.funcs = {
            "avgpool": nn.AvgPool1d,
            "emb": nn.Embedding,
            "conv": nn.Conv1d,
            "lin": nn.Linear,
            "act": nn.Sigmoid
        }
        
        if custom_layers:
            for layer_name, layer in custom_layers.items():
                if self.funcs.get(layer_name):
                    self.funcs[layer_name] = layer
        
        self.layers = {
            "avgpool": self.funcs["avgpool"](self.pool_size),
            "emb": self.funcs["emb"](self.vocab_size+1,
                                self.in_channels),
            "convs": [self.funcs["conv"](self.in_channels,
                                self.out_channels,
                                ksize) for ksize in \
                                range(self.ksize_min, self.ksize_max+1)],
            "lin": self.funcs["lin"](self.in_linear, 
                             self.out_linear),
            "act": self.funcs["act"]()
        }
        
    def forward(self, x):
        # TO DO: calculations
        return x
        
    