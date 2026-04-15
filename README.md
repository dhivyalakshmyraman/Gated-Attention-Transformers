**From Theory to Pythia-70M Implementation**

Implementing the NeurIPS Paper on Gated Attention to combat the Attention Sink problem.

https://neurips.cc/virtual/2025/loc/san-diego/poster/120216

PAPER AT: https://openreview.net/pdf?id=1b7whO4SfY

**🧠 Overview**

This project explores a modification to standard transformer attention called Gated Attention, focusing on:

Identifying weaknesses in standard softmax attention
Introducing a gating mechanism to sparsify attention outputs
Implementing and evaluating this idea on Pythia-70M

The goal is to reduce attention sinks and noisy activations while improving efficiency and interpretability.

**⚠️ Problem: Softmax Attention Vulnerability**

Standard attention is:

**Attention**

<img width="511" height="107" alt="gating" src="https://github.com/user-attachments/assets/12c78c5c-0d80-43ca-837a-2d09c1ab3df0" />

**Issue Identified**
Softmax forces all attention weights to sum to 1
This creates:
Attention sinks → tokens that absorb attention mass
Residual noise → irrelevant signals persist

**📌 What We Do?**

<img width="907" height="593" alt="whatwedo" src="https://github.com/user-attachments/assets/19d562f0-42cb-42a9-878a-1025c6e6d422" />


Even irrelevant tokens get non-zero attention
Leads to inefficient and noisy representations
💡**_ Solution: Gated Attention**_
**Core Idea**

Introduce a learned gate that filters attention outputs after SDPA (Scaled Dot-Product Attention).

**Mechanism**
Compute standard attention output
Apply a gate per head & dimension
Suppress low-importance signals

**⚙️ Architecture**

<img width="685" height="375" alt="architecture" src="https://github.com/user-attachments/assets/d02d2efd-fbbc-4489-a67f-d07c0d981541" />

**Gating Pipeline:**

SDPA output (batch, heads, seq, head_dim)
        ↓
hidden_states @ gate_weight^T
        ↓
sigmoid
        ↓
gate scores (0,1)
        ↓
element-wise multiply
        ↓
gated output (sparse)
        ↓
concat heads → W_o projection

**Key Properties:**

Gate values ∈ (0,1)
Most values → near 0 → sparsity
Acts like a filter on attention outputs


**🧪 Toy Math (Training Intuition):**

<img width="766" height="665" alt="toymath" src="https://github.com/user-attachments/assets/ba845eef-db28-4347-860a-8217cbdc33d6" />


**📊 Insight:**

Mean gate score ≈ 0.116
→ strong sparsity effect

**🔬 Implementation Details**

Model
Pythia-70M (pretrained from HuggingFace)
Dataset
WikiText-103
~200M tokens
Training Setup
Baseline:
AdamW
3 epochs
lr = 1e-4
Gated model:
Inject gate into SDPA
Train only gate parameters
Base model is frozen
lr = 1e-3

**🔄 Workflow**

Pretrained Pythia-70M
        ↓
Fine-tune baseline → baseline.pt
        ↓
Analyze attention (identify sinks)
        ↓
Inject gating mechanism
        ↓
Train gate only → gated.pt
        ↓
Compare results

**📊 Evaluation Metrics**

Perplexity (PPL)
Attention sink behavior
Sparsity (gate activations)
Rank / efficiency indicators

**🚀 Key Contributions**

Identifies attention sink problem in softmax attention
Introduces simple gating mechanism
Achieves:
Sparse activations
Noise reduction
Minimal training (only gate parameters)

🧩** Key Insight**


Instead of changing attention itself,
filter its output with a learned gate

**This preserves:**

Transformer structure
Pretrained weights

**While adding:**

Efficiency
Interpretability
