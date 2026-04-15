"""
Unmodified Pythia-70m wrapper with attention map extraction hooks.
Implements Section 2: Attention sink identification in baseline models.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from config import MODEL_NAME, DEVICE

class PythiaBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the Pythia-70m model
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager").to(DEVICE)
        self.attn_weights = {} # layer_idx -> captured attention weights
        self._hooks = []
        
    def register_hooks(self):
        """Registers hooks to capture attention outputs for sink analysis."""
        self.clear_hooks()
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                attn = None
                if isinstance(output, tuple):
                    # GPTNeoXAttention output formats:
                    # - output_attentions=False: (attn_output, present)
                    # - output_attentions=True: (attn_output, present, attn_weights)
                    if len(output) >= 3 and isinstance(output[2], torch.Tensor):
                        attn = output[2]
                    elif len(output) >= 2 and isinstance(output[1], torch.Tensor):
                        attn = output[1]

                if attn is not None:
                    self.attn_weights[layer_idx] = attn.detach().cpu()
                return output
            return hook
            
        for i, layer in enumerate(self.model.gpt_neox.layers):
            h = layer.attention.register_forward_hook(make_hook(i))
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
