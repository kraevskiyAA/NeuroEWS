import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np

class NeuralSwitchingGMM(nn.Module):
    """
    Neural Switching GMM для обнаружения разладок во временных рядах.
    
    Args:
        input_dim: Размерность входа (для 1D ряда = 1)
        hidden_dim: Размерность скрытого состояния RNN
        n_components: Количество компонент гауссовской смеси
        lambda_sparsity: Коэффициент регуляризации за частоту разладок
        lambda_binary: Коэффициент регуляризации за бинарность alpha
        lambda_distance: Штраф за близкие разладки (новый параметр)
        min_distance: Минимальное расстояние между разладками (новый параметр)
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        n_components: int = 3,
        lambda_sparsity: float = 0.01,
        lambda_binary: float = 0.001,
        lambda_distance: float = 1.0,      # НОВЫЙ: штраф за близкие пики
        min_distance: int = 50,             # НОВЫЙ: зона отчуждения
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_components = n_components
        self.lambda_sparsity = lambda_sparsity
        self.lambda_binary = lambda_binary
        self.lambda_distance = lambda_distance  # НОВЫЙ
        self.min_distance = min_distance         # НОВЫЙ
        
        # === RNN Encoder ===
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,
            dropout=0.1,
        )
        
        # === Detector Head (вероятность разладки) ===
        self.detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # === GMM Parameter Heads (в неограниченном пространстве) ===
        self.head_mu = nn.Linear(hidden_dim, n_components)
        self.head_logvar = nn.Linear(hidden_dim, n_components)
        self.head_logits = nn.Linear(hidden_dim, n_components)
        
        # === Инициализация начального состояния ===
        self.init_mu = nn.Parameter(torch.zeros(n_components))
        self.init_logvar = nn.Parameter(torch.zeros(n_components))
        self.init_logits = nn.Parameter(torch.zeros(n_components))
        
    def _blend_state(
        self,
        alpha: torch.Tensor,
        new_state: Dict[str, torch.Tensor],
        prev_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        blended = {}
        for key in ['mu', 'logvar', 'logits']:
            blended[key] = (
                alpha * new_state[key] +
                (1 - alpha) * prev_state[key]
            )
        return blended
    
    def _compute_gmm_params(
        self,
        blended_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        params = {}
        params['mu'] = blended_state['mu']
        params['sigma'] = torch.exp(0.5 * blended_state['logvar'])
        params['pi'] = F.softmax(blended_state['logits'], dim=-1)
        return params
    
    def _compute_gmm_likelihood(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B, T, _ = x.shape
        K = self.n_components
        
        x_expanded = x.unsqueeze(-1)
        mu = params['mu'].unsqueeze(-1)
        sigma = params['sigma'].unsqueeze(-1)
        pi = params['pi'].unsqueeze(-1)
        
        diff = (x_expanded - mu) / (sigma + 1e-8)
        log_gauss = (
            -0.5 * diff ** 2 
            - torch.log(sigma + 1e-8) 
            - 0.5 * torch.log(torch.tensor(2 * torch.pi, device=x.device))
        )
        
        log_weighted = torch.log(pi + 1e-8) + log_gauss
        log_prob = torch.logsumexp(log_weighted, dim=-2).squeeze(-1)
        
        return log_prob
    
    def _compute_binary_penalty(self, alpha: torch.Tensor) -> torch.Tensor:
        # Минимум при α=0 или α=1, максимум при α=0.5
        return torch.sqrt((alpha * (1 - alpha))).mean()  # или **2 для более резкого штрафа
        
    def _compute_distance_penalty(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Векторизованный штраф за близкие пики alpha.
        
        Если alpha[t] и alpha[t+k] оба высокие (k < min_distance) — штраф.
        Градиент течёт через оба множителя.
        """
        B, T = alpha.shape
        
        if self.min_distance <= 1:
            return torch.tensor(0.0, device=alpha.device)
        
        penalty = torch.tensor(0.0, device=alpha.device)
        
        # Векторизованный подсчёт: для каждого k в [1, min_distance)
        # считаем mean(alpha[:, :-k] * alpha[:, k:])
        for k in range(1, self.min_distance):
            penalty += (alpha[:, :-k] * alpha[:, k:]).mean()
        
        # Нормировка на количество пар
        penalty = penalty / (self.min_distance - 1)
        
        return penalty
    
    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, _ = x.shape
        device = x.device
        
        # === 1. RNN Encoder ===
        h, _ = self.rnn(x)
        
        # === 2. Detector Head ===
        alpha = self.detector(h)  # [B, T, 1]
        
        # === 3. GMM Parameter Heads ===
        new_state = {
            'mu': self.head_mu(h),
            'logvar': self.head_logvar(h),
            'logits': self.head_logits(h),
        }
        
        # === 4. Рекуррентное обновление состояния ===
        prev_state = {
            'mu': self.init_mu.unsqueeze(0).unsqueeze(0).expand(B, 1, self.n_components),
            'logvar': self.init_logvar.unsqueeze(0).unsqueeze(0).expand(B, 1, self.n_components),
            'logits': self.init_logits.unsqueeze(0).unsqueeze(0).expand(B, 1, self.n_components),
        }
        
        blended_states = {'mu': [], 'logvar': [], 'logits': []}
        
        for t in range(T):
            current_new = {k: v[:, t:t+1, :] for k, v in new_state.items()}
            current_blended = self._blend_state(
                alpha[:, t:t+1, :],
                current_new,
                prev_state
            )
            for k in blended_states.keys():
                blended_states[k].append(current_blended[k])
            prev_state = current_blended
        
        blended_state = {
            k: torch.cat(v, dim=1) for k, v in blended_states.items()
        }
        
        # === 5. Преобразование в валидные параметры ===
        params = self._compute_gmm_params(blended_state)
        
        # === 6. Вычисление правдоподобия ===
        log_prob = self._compute_gmm_likelihood(x, params)
        
        # === 7. Функция потерь ===
        nll_loss = -log_prob.mean()
        sparsity_loss = alpha.mean()
        binary_loss = self._compute_binary_penalty(alpha).mean()
        distance_loss = self._compute_distance_penalty(alpha.squeeze(-1))  # НОВЫЙ
        
        total_loss = (
            nll_loss +
            self.lambda_sparsity * sparsity_loss +
            self.lambda_binary * binary_loss +
            self.lambda_distance * distance_loss  # НОВЫЙ член
        )
        
        outputs = {
            'alpha': alpha,
            'params': params,
            'log_prob': log_prob,
            'losses': {
                'total': total_loss,
                'nll': nll_loss,
                'sparsity': sparsity_loss,
                'binary': binary_loss,
                'distance': distance_loss,  # НОВЫЙ
            }
        }
        
        if return_states:
            outputs['blended_state'] = blended_state
        
        return total_loss, outputs
    
    def detect_change_points(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        min_distance: int = None,
    ) -> torch.Tensor:
        self.eval()
        min_distance = min_distance or self.min_distance
        
        with torch.no_grad():
            _, outputs = self.forward(x, return_states=False)
        
        alpha = outputs['alpha'].squeeze(-1)  # [B, T]
        change_points = (alpha > threshold).float()
        
        # Пост-обработка: удаление слишком близких разладок
        for b in range(change_points.shape[0]):
            cp_indices = torch.where(change_points[b] > 0)[0]
            if len(cp_indices) > 1:
                filtered = [cp_indices[0]]
                for idx in cp_indices[1:]:
                    if idx - filtered[-1] >= min_distance:
                        filtered.append(idx)
                change_points[b] = 0
                change_points[b, torch.tensor(filtered, device=x.device)] = 1
        
        return change_points