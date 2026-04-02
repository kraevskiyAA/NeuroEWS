import torch
import numpy as np
from typing import List, Tuple, Optional, Dict


def extract_gmm_parameters(
    model: torch.nn.Module,
    data: torch.Tensor,
    device: str = 'cpu',
) -> Dict[str, np.ndarray]:
    """
    Извлечение всех параметров GMM для каждого шага времени.
    
    Args:
        model: Обученная NeuralSwitchingGMM
        data: Временной ряд [T] или [T, 1]
        device: Устройство для вычислений
        
    Returns:
        Dictionary с параметрами:
        - 'mu': [T, K] — средние значения
        - 'sigma': [T, K] — стандартные отклонения
        - 'pi': [T, K] — веса компонент
        - 'alpha': [T] — вероятность разладки
        - 'log_prob': [T] — лог-правдоподобие
    """
    model.eval()
    
    # Подготовка данных
    if isinstance(data, np.ndarray):
        data = torch.tensor(data).float()
    
    if data.dim() == 1:
        data = data.unsqueeze(-1)  # [T, 1]
    
    # Добавляем batch dimension
    x = data.unsqueeze(0).to(device)  # [1, T, 1]
    
    # Forward pass с возвратом состояний
    with torch.no_grad():
        _, outputs = model(x, return_states=True)
    
    # Извлечение параметров
    params = outputs['params']
    
    results = {
        'mu': params['mu'].squeeze(0).cpu().numpy(),      # [T, K]
        'sigma': params['sigma'].squeeze(0).cpu().numpy(), # [T, K]
        'pi': params['pi'].squeeze(0).cpu().numpy(),       # [T, K]
        'alpha': outputs['alpha'].squeeze(0).squeeze(-1).cpu().numpy(),  # [T]
        'log_prob': outputs['log_prob'].squeeze(0).cpu().numpy(),  # [T]
    }
    
    # Дополнительная статистика
    results['dominant_component'] = np.argmax(results['pi'], axis=1)  # [T]
    results['weighted_mean'] = np.sum(results['pi'] * results['mu'], axis=1)  # [T]
    results['weighted_std'] = np.sqrt(
        np.sum(results['pi'] * (results['sigma']**2 + results['mu']**2), axis=1)  
        - results['weighted_mean']**2                                           
        )
    
    return results