"""
Orchestrates all 9 tasks in order.
"""

import os
import gc
import torch
import tracemalloc
import warnings
warnings.filterwarnings("ignore")

from data import get_dataloaders
from train import train_model
from model_baseline import PythiaBaseline
from model_gated import PythiaGated
from analysis import (
    get_attention_sink_score, plot_attention_sink, analyze_low_rank,
    get_approx_gated_rank, analyze_gating, analyze_head_utilization,
    visualize_dataset_example
)
from evaluate import calculate_perplexity, test_reasoning, save_summary_table
from config import CACHE_DIR, LR_BASELINE, LR_GATE, EPOCHS, WARMUP_STEPS, DEVICE

def header(n, desc):
    print("\n" + "=" * 60)
    print(f"TASK {n}: {desc}")
    print("=" * 60)

def main():
    tracemalloc.start()
    
    print("Loading data...")
    train_loader, val_loader = get_dataloaders()
    metrics = {}
    
    # TASK 1
    header(1, "Train baseline Pythia-70m on WikiText-103 subset")
    baseline_model = PythiaBaseline()
    baseline_checkpoint = os.path.join(CACHE_DIR, "baseline.pt")
    if not os.path.exists(baseline_checkpoint):
        train_model(baseline_model, train_loader, LR_BASELINE, EPOCHS, WARMUP_STEPS, "baseline.pt")
    else:
        #added weights only condition gpt
        baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location="cpu", weights_only=False))
        print("Loaded existing baseline checkpoint.")
        
    metrics['base_ppl'] = calculate_perplexity(baseline_model, val_loader)
    print(f"Baseline Validation PPL: {metrics['base_ppl']:.2f}")
    
    '''
    # TASK 1 (SKIPPED TRAINING)
    header(1, "Load baseline model (skip training)")
    baseline_model = PythiaBaseline()

    baseline_checkpoint = os.path.join(CACHE_DIR, "baseline.pt")
    baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location="cpu"))

    print("Loaded baseline checkpoint.")

    metrics['base_ppl'] = calculate_perplexity(baseline_model, val_loader)
    print(f"Baseline Validation PPL: {metrics['base_ppl']:.2f}")
    '''
    # TASK 2
    header(2, "Show attention sink on baseline")
    base_sink_scores, base_sink_mean = get_attention_sink_score(baseline_model, val_loader)
    print(f"Global mean baseline sink score: {base_sink_mean:.4f}")
    plot_attention_sink(base_sink_scores, save_name="attention_sink_baseline.png")
    metrics['base_sink'] = base_sink_mean

    # TASK 3
    header(3, "Demonstrate low-rank expressiveness limits WITHOUT gating")
    base_rank = analyze_low_rank(baseline_model, save_name="low_rank_analysis.png")
    metrics['base_rank'] = base_rank
    
    # TASK 6 (Baseline part)
    header(6, "Show why uniform attention across heads was inefficient (baseline)")
    base_red = analyze_head_utilization(baseline_model, val_loader, "head_analysis_baseline.png")
    print(f"Baseline Head Redundancy Rate: {base_red*100:.1f}%")
    metrics['base_red'] = base_red
    
    base_params = sum(p.numel() for p in baseline_model.parameters())

    # TASK 4
    header(4, "Inject SDPA sigmoid gate and eliminate attention sink")
    gated_model = PythiaGated()
    
    # Map weights due to wrapper structure differences
    base_state = baseline_model.state_dict()
    mapped_state = {}
    for k, v in base_state.items():
        if '.attention.' in k:
            new_k = k.replace('.attention.', '.attention.attn.')
            new_k = new_k.replace('.attn.dense.', '.attn.dense.original_dense.')
            mapped_state[new_k] = v
        else:
            mapped_state[k] = v
            
    gated_model.load_state_dict(mapped_state, strict=False)
    
    gate_checkpoint = os.path.join(CACHE_DIR, "gated.pt")
    if not os.path.exists(gate_checkpoint):
        def is_gate(name): return 'gate_weight' in name
        train_model(gated_model, train_loader, LR_GATE, EPOCHS, WARMUP_STEPS, "gated.pt", is_gate)
    else:
        gated_model.load_state_dict(torch.load(gate_checkpoint, map_location="cpu", weights_only=False))
        print("Loaded existing gated checkpoint.")
        
    metrics['gate_ppl'] = calculate_perplexity(gated_model, val_loader)
    print(f"Gated Validation PPL: {metrics['gate_ppl']:.2f}")
    
    gate_sink_scores, gate_sink_mean = get_attention_sink_score(gated_model, val_loader)
    print(f"Gated mean sink score: {gate_sink_mean:.4f}")
    plot_attention_sink(base_sink_scores, gate_sink_scores, "attention_sink_comparison.png")
    metrics['gate_sink'] = gate_sink_mean

    # TASK 5
    header(5, "Visualize how gating adds non-linearity to attention heads")
    gate_mean, gate_sparse = analyze_gating(gated_model, val_loader)
    print(f"Mean gate score: {gate_mean:.3f}, Fraction < 0.2: {gate_sparse*100:.1f}%")
    metrics['gate_mean'] = gate_mean
    metrics['gate_sparse'] = gate_sparse
    
    # Dataset real-case visualization
    visualize_dataset_example(baseline_model, gated_model)
    
    # TASK 6 (Gated part)
    header(6, "Show why uniform attention across heads was inefficient (gated)")
    gate_red = analyze_head_utilization(gated_model, val_loader, "head_analysis_gated.png")
    print(f"Gated Head Redundancy Rate: {gate_red*100:.1f}%")
    metrics['gate_red'] = gate_red

    # TASK 7
    header(7, "Confirm model size is unchanged by gating")
    gated_params = sum(p.numel() for p in gated_model.parameters())
    gate_only = sum(p.numel() for n, p in gated_model.named_parameters() if 'gate_weight' in n)
    pct_increase = (gate_only / base_params) * 100
    
    print(f"Baseline Params: {base_params:,}")
    print(f"Gated Params   : {gated_params:,}")
    print(f"Gate-only Params: {gate_only:,}")
    print(f"Percentage increase: {pct_increase:.4f}%")
    
    assert pct_increase < 1.0, f"Gate params increase {pct_increase}% >= 1%"
    print("Assertion PASS: Gate params < 1% of total params")
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    print(f"Peak memory usage: {peak_mem / 10**6:.2f} MB")
    
    # TASK 8
    test_reasoning(baseline_model, gated_model)
    
    # TASK 9
    header(9, "Reasoning quality improvement metrics summary")
    gate_rank = get_approx_gated_rank(gated_model)
    metrics['gate_rank'] = gate_rank
    save_summary_table(metrics)
    
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
