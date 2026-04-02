import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralGMM import NeuralSwitchingGMM
from typing import List, Tuple, Optional, Dict


def train_model(
    model: NeuralSwitchingGMM,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 100,
    device: str = 'cuda',
):
    model.to(device)
    model.train()

    history = {
        'total': [],
        'nll': [],
        'sparsity': [],
        'binary': [],
        'distance': []
    }
    
    # Сохраняем лучшую модель в памяти
    best_loss = float('inf')
    best_state_dict = None
    
    for epoch in range(n_epochs):
        epoch_losses = {k: 0.0 for k in history.keys()}
        n_batches = 0
        
        for b in dataloader:
            batch, _ = b
            x = batch.to(device)
            
            optimizer.zero_grad()
            loss, outputs = model(x)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            for k in epoch_losses.keys():
                epoch_losses[k] += outputs['losses'][k].item()
            
            n_batches += 1
        
        for k in history.keys():
            history[k].append(epoch_losses[k] / n_batches)
        
        avg_total_loss = history['total'][-1]
        
        # Сохраняем копию весов лучшей модели в памяти
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            best_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        
        print(f"Epoch {epoch+1}/{n_epochs}: "
                f"Loss={avg_total_loss:.4f}, "
                f"NLL={history['nll'][-1]:.4f}, "
                f"Sparsity={history['sparsity'][-1]:.4f}, "
                f"Binary={history['binary'][-1]:.4f}",
                f"Distance={history['distance'][-1]:.4f}"
                )
    
    # Загружаем лучшие веса обратно в модель
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
    
    return model, history, best_loss

def diagnose_training(history: Dict[str, list]):
    """
    Проверка здоровости обучения.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Total Loss
    axes[0, 0].plot(history['total_loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    
    # NLL
    axes[0, 1].plot(history['nll'])
    axes[0, 1].set_title('Negative Log-Likelihood')
    axes[0, 1].set_xlabel('Epoch')
    
    # Sparsity
    axes[1, 0].plot(history['sparsity'])
    axes[1, 0].set_title('Mean Alpha (Sparsity)')
    axes[1, 0].set_xlabel('Epoch')
    
    # Entropy
    axes[1, 1].plot(history['entropy'])
    axes[1, 1].set_title('Entropy Penalty')
    axes[1, 1].set_xlabel('Epoch')
    
    plt.tight_layout()
    plt.show()