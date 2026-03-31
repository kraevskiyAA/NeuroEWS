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
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        n_components: int = 3,
        lambda_sparsity: float = 0.01,
        lambda_binary: float = 0.001,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_components = n_components
        self.lambda_sparsity = lambda_sparsity
        self.lambda_binary = lambda_binary
        
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
        # Средние значения (R)
        self.head_mu = nn.Linear(hidden_dim, n_components)
        
        # Логарифм дисперсии (R) -> гарантирует положительность
        self.head_logvar = nn.Linear(hidden_dim, n_components)
        
        # Логиты весов (R) -> softmax гарантирует сумму = 1
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
        """Преобразование параметров с защитой от численной нестабильности"""
        params = {}
        
        # Средние: без изменений
        params['mu'] = blended_state['mu']  # [B, T, K]

        params['sigma'] = torch.exp(0.5 * blended_state['logvar'])  # [B, T, K]
        
        # Веса: softmax
        params['pi'] = F.softmax(blended_state['logits'], dim=-1)  # [B, T, K]
        
        return params
    
    def _compute_gmm_likelihood(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Вычисление логарифма правдоподобия GMM.
        
        Args:
            x: [B, T, 1] — наблюдаемые данные
            params: словарь с mu, sigma, pi
            
        Returns:
            log_prob: [B, T] — логарифм вероятности для каждой точки
        """
        B, T, _ = x.shape
        K = self.n_components
        
        x_expanded = x.unsqueeze(-1)  # [B, T, 1, 1]
        mu = params['mu'].unsqueeze(-1)  # [B, T, K, 1]
        sigma = params['sigma'].unsqueeze(-1)  # [B, T, K, 1]
        pi = params['pi'].unsqueeze(-1)  # [B, T, K, 1]
        
        # Гауссовская плотность для каждой компоненты
        # N(x | mu, sigma^2) = 1/(sqrt(2*pi)*sigma) * exp(-0.5 * ((x-mu)/sigma)^2)
        diff = (x_expanded - mu) / (sigma + 1e-8)
        log_gauss = (
            -0.5 * diff ** 2 
            - torch.log(sigma + 1e-8) 
            - 0.5 * torch.log(torch.tensor(2 * torch.pi, device=x.device))
        )  # [B, T, K, 1]
        
        # Взвешенная сумма (log-sum-exp trick для стабильности)
        # log(sum_k pi_k * N_k) = logsumexp(log(pi_k) + log(N_k))
        log_weighted = torch.log(pi + 1e-8) + log_gauss  # [B, T, K, 1]
        log_prob = torch.logsumexp(log_weighted, dim=-2).squeeze(-1)  # [B, T]
        
        return log_prob
    
    def _compute_entropy_penalty(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Вычисление бинарной энтропии для поощрения четких решений.
        
        H(alpha) = -[alpha * log(alpha) + (1-alpha) * log(1-alpha)]
        """
        entropy = -(
            alpha * torch.log(alpha) +
            (1 - alpha) * torch.log(1 - alpha)
        )
        return entropy
    
    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Прямой проход модели.
        
        Args:
            x: [B, T, 1] — временной ряд
            return_states: если True, возвращает все промежуточные состояния
            
        Returns:
            loss: скаляр — итоговая функция потерь
            outputs: словарь с alpha, params, likelihood
        """
        B, T, _ = x.shape
        device = x.device
        
        # === 1. RNN Encoder ===
        h, _ = self.rnn(x)  # [B, T, H]
        
        # === 2. Detector Head ===
        alpha = self.detector(h)  # [B, T, 1]
        
        # === 3. GMM Parameter Heads ===
        new_state = {
            'mu': self.head_mu(h),  # [B, T, K]
            'logvar': self.head_logvar(h),  # [B, T, K]
            'logits': self.head_logits(h),  # [B, T, K]
        }
        
        # === 4. Рекуррентное обновление состояния ===
        # Инициализация t=0
        prev_state = {
            'mu': self.init_mu.unsqueeze(0).unsqueeze(0).expand(B, 1, self.n_components),
            'logvar': self.init_logvar.unsqueeze(0).unsqueeze(0).expand(B, 1, self.n_components),
            'logits': self.init_logits.unsqueeze(0).unsqueeze(0).expand(B, 1, self.n_components),
        }
        
        # Последовательное обновление по времени
        blended_states = {'mu': [], 'logvar': [], 'logits': []}
        
        for t in range(T):
            # Текущие новые параметры
            current_new = {k: v[:, t:t+1, :] for k, v in new_state.items()}
            
            # Смешивание
            current_blended = self._blend_state(
                alpha[:, t:t+1, :],
                current_new,
                prev_state
            )
            
            # Сохранение
            for k in blended_states.keys():
                blended_states[k].append(current_blended[k])
            
            # Обновление предыдущего состояния
            prev_state = current_blended
        
        # Конкатенация по времени
        blended_state = {
            k: torch.cat(v, dim=1) for k, v in blended_states.items()
        }
        
        # === 5. Преобразование в валидные параметры ===
        params = self._compute_gmm_params(blended_state)
        
        # === 6. Вычисление правдоподобия ===
        log_prob = self._compute_gmm_likelihood(x, params)  # [B, T]
        
        # === 7. Функция потерь ===
        # NLL
        nll_loss = -log_prob.mean()
        
        # Sparsity penalty (штраф за частые разладки)
        sparsity_loss = alpha.mean()
        
        # Binary entropy penalty (штраф за нечеткие решения)
        entropy_loss = self._compute_entropy_penalty(alpha).mean()
        
        # Итоговый loss
        total_loss = (
            nll_loss +
            self.lambda_sparsity * sparsity_loss +
            self.lambda_binary * entropy_loss
        )
        
        outputs = {
            'alpha': alpha,  # [B, T, 1]
            'params': params,
            'log_prob': log_prob,  # [B, T]
            'losses': {
                'total': total_loss,
                'nll': nll_loss,
                'sparsity': sparsity_loss,
                'entropy': entropy_loss,
            }
        }
        
        if return_states:
            outputs['blended_state'] = blended_state
        
        return total_loss, outputs
    
    def detect_change_points(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        min_distance: int = 5,
    ) -> torch.Tensor:
        """
        Обнаружение точек разладки после обучения.
        
        Args:
            x: [B, T, 1] — временной ряд
            threshold: порог для alpha
            min_distance: минимальное расстояние между разладками
            
        Returns:
            change_points: [B, T] — бинарная маска точек разладки
        """
        self.eval()
        with torch.no_grad():
            _, outputs = self.forward(x, return_states=False)
        
        alpha = outputs['alpha'].squeeze(-1)  # [B, T]
        
        # Бинаризация по порогу
        change_points = (alpha > threshold).float()
        
        # Пост-обработка: удаление слишком близких разладок
        for b in range(change_points.shape[0]):
            cp_indices = torch.where(change_points[b] > 0)[0]
            if len(cp_indices) > 1:
                filtered = [cp_indices[0]]
                for idx in cp_indices[1:]:
                    if idx - filtered[-1] >= min_distance:
                        filtered.append(idx)
                
                # Сброс и установка отфильтрованных
                change_points[b] = 0
                change_points[b, torch.tensor(filtered)] = 1
        
        return change_points
    