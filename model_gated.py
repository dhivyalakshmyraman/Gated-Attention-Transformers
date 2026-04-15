"""
Pythia-70m wrapper injecting an SDPA elementwise sigmoid gate.
Implements Section 4.2: Gating Introduces Input-Dependent Sparsity.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from config import MODEL_NAME, DEVICE, NUM_HEADS, HEAD_DIM

class GatedDense(nn.Module):
    def __init__(self, original_dense, wrapper):
        super().__init__()
        self.original_dense = original_dense
        # Store wrapper as a non-registered reference to avoid circular nn.Module graph
        # (GatedAttentionWrapper -> attn.dense -> GatedDense -> wrapper -> GatedAttentionWrapper...)
        object.__setattr__(self, '_wrapper', wrapper)
        self.num_heads = wrapper.num_heads
        self.head_dim = wrapper.head_dim
        
    def forward(self, hidden_states):
        b, s, d = hidden_states.shape
        wrapper = self._wrapper
        
        # Reshape to [batch, seq, num_heads, head_dim] to apply gate per head
        attn_output = hidden_states.view(b, s, self.num_heads, self.head_dim)
        wrapper.last_pre_gate_output = attn_output.detach().cpu()
        
        # The true layer input was saved exactly before attention processing
        layer_input = wrapper.layer_input
        layer_input_reshaped = layer_input.view(b, s, self.num_heads, self.head_dim)
        
        # Compute gate scores
        scores = torch.einsum('bshd,hd->bsh', layer_input_reshaped, wrapper.gate_weight)
        gate = torch.sigmoid(scores)
        wrapper.last_gate_scores = gate.detach().cpu()
        
        # Multiply elementwise
        gated_attn_output = attn_output * gate.unsqueeze(-1)
        
        # Reshape back and apply dense projection
        gated_merged = gated_attn_output.view(b, s, d)
        return self.original_dense(gated_merged)

class GatedAttentionWrapper(nn.Module):
    def __init__(self, original_attn, layer_idx, num_heads, head_dim):
        super().__init__()
        self.attn = original_attn
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.gate_weight = nn.Parameter(torch.zeros(num_heads, head_dim))
        
        self.last_gate_scores = None
        self.last_pre_gate_output = None
        self.layer_input = None
        
        # Intercept dense layer
        original_dense = self.attn.dense
        self.attn.dense = GatedDense(original_dense, self)

    def forward(self, hidden_states, *args, **kwargs):
        self.layer_input = hidden_states
        return self.attn(hidden_states, *args, **kwargs)

class PythiaGated(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager").to(DEVICE)
        self.attn_weights = {}
        self._hooks = []
        self.gated_layers = []
        
        # Wrap all attention layers with our Gated wrapper
        for i, layer in enumerate(self.model.gpt_neox.layers):
            original_attn = layer.attention
            wrapper = GatedAttentionWrapper(original_attn, i, NUM_HEADS, HEAD_DIM)
            layer.attention = wrapper
            self.gated_layers.append(wrapper)

    def register_hooks(self):
        """Registers hooks to capture attention outputs for sink analysis."""
        self.clear_hooks()
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                attn = None
                if isinstance(output, tuple):
                    if len(output) >= 3 and isinstance(output[2], torch.Tensor):
                        attn = output[2]
                    elif len(output) >= 2 and isinstance(output[1], torch.Tensor):
                        attn = output[1]

                if attn is not None:
                    self.attn_weights[layer_idx] = attn.detach().cpu()
                return output
            return hook
            
        for i, layer in enumerate(self.model.gpt_neox.layers):
            # layer.attention is now our GatedAttentionWrapper
            # The actual GPTNeoXAttention is at layer.attention.attn
            h = layer.attention.attn.register_forward_hook(make_hook(i))
            self._hooks.append(h)

    def clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.attn_weights.clear()

    def forward(self, input_ids, output_attentions=False, output_hidden_states=False):
        return self.model(
            input_ids=input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
