"""
Perplexity measurement and reasoning quality test cases.
Outputs the final summary table of baseline vs gated variants.
"""

import os
import torch
import math
from transformers import AutoTokenizer
from config import CACHE_DIR, MODEL_NAME, DEVICE

PROMPTS = [
    "The capital of France is",
    "If all cats are animals and Felix is a cat, then",
    "The next number in the sequence 2, 4, 8, 16 is",
    "To bake a cake you need flour, eggs, and",
    "She opened the door and saw that the room was",
]

def calculate_perplexity(model, val_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
            
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

def test_reasoning(baseline_model, gated_model):
    print("\n" + "="*80)
    print("TASK 8: Reasoning Quality Test Cases")
    print("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    baseline_model.eval()
    gated_model.eval()
    
    for idx, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        
        with torch.no_grad():
            base_out = baseline_model.model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gate_out = gated_model.model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
        base_text = tokenizer.decode(base_out[0], skip_special_tokens=True).replace('\n', ' ')
        gate_text = tokenizer.decode(gate_out[0], skip_special_tokens=True).replace('\n', ' ')
        
        print(f"Prompt {idx+1}: {prompt}")
        print(f"Baseline : {base_text}")
        print(f"Gated    : {gate_text}")
        print("-" * 80)
        
def save_summary_table(metrics_dict, save_name="results_summary.txt"):
    table = f"""
┌─────────────────────────────┬──────────────┬──────────────┐
│ Metric                      │ Baseline     │ Gated        │
├─────────────────────────────┼──────────────┼──────────────┤
│ Perplexity (val)            │ {metrics_dict['base_ppl']:>12.2f} │ {metrics_dict['gate_ppl']:>12.2f} │
│ Attention sink score        │ {metrics_dict['base_sink']:>12.3f} │ {metrics_dict['gate_sink']:>12.3f} │
│ Mean gate score             │          N/A │ {metrics_dict['gate_mean']:>12.3f} │
│ Gate sparsity (score<0.2)   │          N/A │ {metrics_dict['gate_sparse']*100:>11.1f}% │
│ Head redundancy rate        │ {metrics_dict['base_red']*100:>11.1f}% │ {metrics_dict['gate_red']*100:>11.1f}% │
│ Effective rank (approx)     │ {metrics_dict['base_rank']:>12.1f} │ {metrics_dict['gate_rank']:>12.1f} │
└─────────────────────────────┴──────────────┴──────────────┘
"""
    print(table)
    with open(os.path.join(CACHE_DIR, save_name), 'w', encoding="utf-8") as f:
        f.write(table.strip())
