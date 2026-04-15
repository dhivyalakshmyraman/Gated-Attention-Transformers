"""
Configuration settings for the Gated Attention experiments.
Defines constants, paths, and hyperparameters for Pythia-70m and WikiText-103.
"""

import os
import torch

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_NAME = "EleutherAI/pythia-70m"
MAX_TRAIN_TOKENS = 100000
MAX_VAL_TOKENS = 14000

BLOCK_SIZE = 64
BATCH_SIZE = 2
ACCUMULATE_STEPS = 2
LR_BASELINE = 5e-5
LR_GATE = 5e-4
EPOCHS = 1
WARMUP_STEPS = 5

NUM_LAYERS = 6
NUM_HEADS = 8
HEAD_DIM = 64
MODEL_DIM = 512

DEVICE = torch.device("cpu")
