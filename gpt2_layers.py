import numpy as np
import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from quantizer import (
    custom_round,
    
    quantize_dynamic,
    quantize_static
)


class GPT2Block(nn.Module):
    def __init__(self, config, quantization=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        
        self.attn = GPT2Attention(config, quantization=quantization)
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, quantization=quantization)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config, quantization=quantization)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
    

class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, quantization=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention
        self.quantization = quantization

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim, quantization=self.quantization)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim, quantization=self.quantization)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim, quantization=self.quantization)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim, quantization=self.quantization)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config, quantization=None):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim, quantization=quantization)
        self.c_proj = Conv1D(embed_dim, intermediate_size, quantization=quantization)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx, quantization=None, momentum=0.01):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))
        self.quant_type = "fp"
        self.quant_bits = 16
        self.static_batch = 135
        if quantization:
            self.quant_type = quantization["type"]
            self.quant_bits = quantization["n_bits"]
        
        # parameters for static quantization
        self.momentum = momentum
        self.qmin = -1.0 * (2**self.quant_bits)/2
        self.qmax = -self.qmin - 1
        
        self.scale_x = None
        self.zp_x = None
        self.scale_wts = None
        self.zp_wts = None
        
        self.static_stats = {
            "min_x": [],
            "max_x": [],
            "min_wts": [],
            "max_wts": []
        }
        
        self.input_tensor = 0
        
#         self.min_wts = self.weight.detach().min()
#         self.max_wts = self.weight.detach().max()
#         self.scale_wts, self.zp_wts = self._calc_quant_params(self.min_wts, self.max_wts)
#         if self.quant_type == "static":
#             self.quant_wts = quantize_static(self.weight, self.scale_wts, self.zp_wts, self.qmin, self.qmax)
    def _get_mean_stats(self):
        for stat, vals in self.static_stats.items():
            self.static_stats[stat] = torch.Tensor(vals).mean()
#             if self.quant_type == "training":
#                 self.static_stats[stat] = nn.Parameter(self.static_stats[stat], requires_grad=True)
    
    def _calc_quant_params(self, min_val, max_val):
#         if isinstance(min_val, list):
#             min_val = float(np.median(min_val))
#             max_val = float(np.median(max_val))
        min_val = min(0.0, min_val.item())
        max_val = max(0.0, max_val.item())
        if max_val == min_val:
            scale = 1.0
            zp = 0
        else:
            max_range = max(-min_val, max_val) # largest mag(value)
            scale = max_range / ((self.qmax - self.qmin) / 2)
            scale = max(scale, 1e-8)
            zp = 0.0
        return scale, zp
#         self.scale_x = scale
#         self.zp_x = zp
    
#     def _calc_quant_stats(self,
#                           x):
#         min_val = x.detach().min()
#         max_val = x.detach().max()

#         # compute qparams --> scale and zero_point
#         max_val, min_val = float(max_val), float(min_val)
#         min_val = min(0.0, min_val)
#         max_val = max(0.0, max_val)

#         if max_val == min_val:
#             scale = 1.0
#             zp = 0
#         else:
#             max_range = max(-min_val, max_val) # largest mag(value)
#             scale = max_range / ((self.qmax - self.qmin) / 2)
#             scale = max(scale, 1e-8)
#             zp = 0.0 # this true for symmetric quantization
#         return scale, zp

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        if self.quant_type in ["static", "ste", "qsin"]:
            if self.static_batch > 0:
                x_inp = x.clone()
                wts_inp = self.weight.clone()
#             if len(self.scales_x) < self.static_batch:
#                 scale, zp = self._calc_quant_stats(x)
                self.static_stats["min_x"].append(float(x_inp.detach().min()))
                self.static_stats["max_x"].append(float(x_inp.detach().max()))
                self.static_stats["min_wts"].append(float(wts_inp.detach().min()))
                self.static_stats["max_wts"].append(float(wts_inp.detach().max()))
                self.static_batch -= 1
                x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
                x = x.view(*size_out)
                return x
            elif not self.scale_x and not self.zp_x:
                self._get_mean_stats()
                self.scale_x, self.zp_x = self._calc_quant_params(self.static_stats["min_x"], 
                                                                  self.static_stats["max_x"])
                self.scale_wts, self.zp_wts = self._calc_quant_params(self.static_stats["min_wts"], 
                                                                  self.static_stats["max_wts"])
                if self.quant_type in ["ste", "qsin"]:
                    if not isinstance(self.scale_wts, torch.Tensor):
                        self.scale_wts = torch.Tensor([self.scale_wts])
                    if not isinstance(self.scale_x, torch.Tensor):
                        self.scale_x = torch.Tensor([self.scale_x])
                    self.input_tensor = x
                    self.scale_wts = nn.Parameter(self.scale_wts, requires_grad=True).cuda()
                    self.scale_x = nn.Parameter(self.scale_x, requires_grad=True).cuda()
                x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
                x = x.view(*size_out)
                return x
            else:
                quant_x = quantize_static(x, self.scale_x, self.zp_x, self.qmin, self.qmax)
                quant_wts = quantize_static(self.weight, self.scale_wts, self.zp_wts, self.qmin, self.qmax)
                x = torch.addmm(self.bias, quant_x.view(-1, quant_x.size(-1)), quant_wts)
                x = x.view(*size_out)
                return x
        elif self.quant_type == "dynamic":
                quant_wts = quantize_dynamic(self.weight, self.qmin, self.qmax)
                quant_x = quantize_dynamic(x, self.qmin, self.qmax)
                x = torch.addmm(self.bias, quant_x.view(-1, quant_x.size(-1)), quant_wts)
                x = x.view(*size_out)
                return x
        elif self.quant_type == "fp":
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            x = x.view(*size_out)
            return x
