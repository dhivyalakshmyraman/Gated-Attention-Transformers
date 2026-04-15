"""
Training loop, optimizer schema, and checkpoint logic.
Shared across baseline and gated models.
"""

import os
import gc
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from config import CACHE_DIR, DEVICE, ACCUMULATE_STEPS

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def train_model(model, train_loader, lr, epochs, warmup_steps, save_name, trainable_param_predicate=None):
    """
    Trains the model with gradient accumulation.
    Only updates parameters where trainable_param_predicate is True.
    """
    model.to(DEVICE)
    model.train()

    if trainable_param_predicate is not None:
        for name, param in model.named_parameters():
            param.requires_grad = trainable_param_predicate(name)
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr)
    
    total_steps = (len(train_loader) // ACCUMULATE_STEPS) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    step = 0
    model.zero_grad()
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / ACCUMULATE_STEPS

            loss.backward()
            losses.append(loss.item() * ACCUMULATE_STEPS)

            if (i + 1) % ACCUMULATE_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                step += 1
                
                if step % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Step {step}/{total_steps} | Loss: {losses[-1]:.4f}")

    save_path = os.path.join(CACHE_DIR, save_name)
    torch.save(model.state_dict(), save_path, pickle_protocol=4)
    print(f"Saved checkpoint to {save_path}")

    plt.figure()
    plt.plot(losses)
    plt.title(f"Training Loss - {save_name}")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(CACHE_DIR, f"loss_{save_name.split('.')[0]}.png"))
    plt.close()
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
