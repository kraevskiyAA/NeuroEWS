import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt

def generate_ou_process(
    n_steps: int,
    dt: float = 0.01,
    theta: float = 1.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    x0: float | None = None,
    n_sequences: int = 1,
    random_state: int | None = None
) -> np.ndarray:
    """
    Генерирует одну или несколько последовательностей из процесса Орнштейна-Уленбека.
    
    Параметры
    ----------
    n_steps : int
        Количество временных шагов в каждой последовательности.
    dt : float, optional
        Временной шаг дискретизации (по умолчанию 0.01).
    theta : float, optional
        Параметр скорости возврата к среднему (по умолчанию 1.0).
    mu : float, optional
        Параметр среднего значения (по умолчанию 0.0).
    sigma : float, optional
        Параметр волатильности (по умолчанию 1.0).
    x0 : float, optional
        Начальное значение процесса. Если None, выбирается случайное 
        из стационарного распределения N(mu, sigma²/(2*theta)).
    n_sequences : int, optional
        Количество независимых последовательностей для генерации (по умолчанию 1).
    random_state : int, optional
        Seed для генератора случайных чисел для воспроизводимости.
    
    Возвращает
    ----------
    np.ndarray
        Массив формы (n_sequences, n_steps) с сгенерированными данными.
    
    Примеры
    --------
    >>> # Генерация одной последовательности
    >>> data = generate_ou_process(n_steps=1000, theta=0.5, mu=2.0, sigma=1.0)
    >>> data.shape
    (1000,)
    
    >>> # Генерация 100 последовательностей
    >>> batch = generate_ou_process(n_steps=500, n_sequences=100, random_state=42)
    >>> batch.shape
    (100, 500)
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()
    
    # Генерация приращений винеровского процесса: N(0, sqrt(dt))
    dW = rng.standard_normal((n_sequences, n_steps)) * np.sqrt(dt)
    
    # Инициализация массива результатов
    X = np.zeros((n_sequences, n_steps))
    
    # Начальное значение
    if x0 is None:
        # Начальное значение из стационарного распределения
        stationary_std = sigma / np.sqrt(2 * theta)
        X[:, 0] = rng.normal(mu, stationary_std, size=n_sequences)
    else:
        X[:, 0] = x0
    
    # Применяем схему Эйлера-Маруямы для каждого шага
    for t in range(1, n_steps):
        X[:, t] = (
            X[:, t - 1]  # X(t)
            + theta * (mu - X[:, t - 1]) * dt  # θ(μ - X(t))dt
            + sigma * dW[:, t]  # σ dW(t)
        )
    
    # Для одной последовательности возвращаем 1D массив
    if n_sequences == 1:
        return X[0]
    
    return X


class GMMChangePointGenerator:
    """
    Генератор одномерного временного ряда из гауссовской смеси с точками разладки.
    
    Параметры:
    -----------
    n_timesteps : int
        Общее количество временных меток
    change_points : List[int]
        Список индексов, где происходят разладки (моменты изменения распределения)
    n_components : int
        Количество компонент в GMM
    means : List[List[float]]
        Список списков средних для каждого сегмента (длина = len(change_points) + 1)
        Каждый внутренний список имеет длину n_components
    variances : List[List[float]]
        Список списков дисперсий для каждого сегмента (длина = len(change_points) + 1)
        Каждый внутренний список имеет длину n_components
    weights : Optional[List[List[float]]]
        Список списков весов для каждого сегмента (длина = len(change_points) + 1)
        Если None, веса равномерные (1/n_components)
    random_seed : Optional[int]
        Seed для воспроизводимости результатов
    """
    
    def __init__(
        self,
        n_timesteps: int,
        change_points: List[int],
        n_components: int,
        means: List[List[float]],
        variances: List[List[float]],
        weights: Optional[List[List[float]]] = None,
        random_seed: Optional[int] = None
    ):
        self.n_timesteps = n_timesteps
        self.change_points = change_points
        self.n_components = n_components
        
        # Проверка входных данных
        self._validate_inputs(means, variances, weights)
        
        self.means = means
        self.variances = variances
        self.weights = weights if weights is not None else self._create_uniform_weights()
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.segments = self._create_segments()
        
    def _validate_inputs(self, means, variances, weights):
        """Проверка корректности входных данных"""
        n_segments = len(self.change_points) + 1
        
        if len(means) != n_segments:
            raise ValueError(
                f"Количество списков средних ({len(means)}) не соответствует "
                f"количеству сегментов ({n_segments})"
            )
        
        if len(variances) != n_segments:
            raise ValueError(
                f"Количество списков дисперсий ({len(variances)}) не соответствует "
                f"количеству сегментов ({n_segments})"
            )
        
        for i, mean_list in enumerate(means):
            if len(mean_list) != self.n_components:
                raise ValueError(
                    f"Сегмент {i}: количество средних ({len(mean_list)}) не равно "
                    f"количеству компонент ({self.n_components})"
                )
        
        for i, var_list in enumerate(variances):
            if len(var_list) != self.n_components:
                raise ValueError(
                    f"Сегмент {i}: количество дисперсий ({len(var_list)}) не равно "
                    f"количеству компонент ({self.n_components})"
                )
            for j, var in enumerate(var_list):
                if var <= 0:
                    raise ValueError(
                        f"Сегмент {i}, компонента {j}: дисперсия должна быть положительной, "
                        f"получено {var}"
                    )
        
        if weights is not None:
            if len(weights) != n_segments:
                raise ValueError(
                    f"Количество списков весов ({len(weights)}) не соответствует "
                    f"количеству сегментов ({n_segments})"
                )
            for i, weight_list in enumerate(weights):
                if len(weight_list) != self.n_components:
                    raise ValueError(
                        f"Сегмент {i}: количество весов ({len(weight_list)}) не равно "
                        f"количеству компонент ({self.n_components})"
                    )
                if not np.isclose(sum(weight_list), 1.0):
                    raise ValueError(
                        f"Сегмент {i}: сумма весов ({sum(weight_list)}) не равна 1"
                    )
    
    def _create_uniform_weights(self) -> List[List[float]]:
        """Создание равномерных весов для всех сегментов"""
        n_segments = len(self.change_points) + 1
        uniform_weights = [1.0 / self.n_components] * self.n_components
        return [uniform_weights.copy() for _ in range(n_segments)]
    
    def _create_segments(self) -> List[Tuple[int, int]]:
        """Создание границ сегментов на основе точек разладки"""
        segments = []
        start = 0
        
        for cp in sorted(self.change_points):
            if cp <= start or cp >= self.n_timesteps:
                raise ValueError(
                    f"Точка разладки {cp} должна быть между {start+1} и {self.n_timesteps-1}"
                )
            segments.append((start, cp))
            start = cp
            
        segments.append((start, self.n_timesteps))
        return segments
    
    def _sample_from_gmm(
        self, 
        n_samples: int, 
        means: List[float], 
        variances: List[float], 
        weights: List[float]
    ) -> np.ndarray:
        """
        Семплирование из GMM с заданными параметрами
        
        Параметры:
        -----------
        n_samples : int
            Количество семплов
        means : List[float]
            Средние компонент
        variances : List[float]
            Дисперсии компонент
        weights : List[float]
            Веса компонент
            
        Возвращает:
        ------------
        np.ndarray : Семплы из GMM
        """
        # Выбор компоненты для каждого семпла
        components = np.random.choice(
            self.n_components, 
            size=n_samples, 
            p=weights
        )
        
        # Генерация семплов из выбранных компонент
        samples = np.zeros(n_samples)
        for k in range(self.n_components):
            mask = components == k
            n_k = np.sum(mask)
            if n_k > 0:
                samples[mask] = np.random.normal(
                    loc=means[k],
                    scale=np.sqrt(variances[k]),
                    size=n_k
                )
        
        return samples
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Генерация временного ряда и меток сегментов
        
        Возвращает:
        ------------
        data : np.ndarray
            Временной ряд (длина = n_timesteps)
        segment_labels : np.ndarray
            Метки сегментов для каждой временной метки (0, 1, 2, ...)
        change_points : List[int]
            Список точек разладки (использованные при генерации)
        """
        data = np.zeros(self.n_timesteps)
        segment_labels = np.zeros(self.n_timesteps, dtype=int)
        
        for seg_idx, (start, end) in enumerate(self.segments):
            n_samples = end - start
            
            # Генерация данных для сегмента
            data[start:end] = self._sample_from_gmm(
                n_samples=n_samples,
                means=self.means[seg_idx],
                variances=self.variances[seg_idx],
                weights=self.weights[seg_idx]
            )
            
            # Присвоение метки сегмента
            segment_labels[start:end] = seg_idx
        
        return data, segment_labels, self.change_points
    
    def generate_with_info(self) -> pd.DataFrame:
        """
        Генерация временного ряда с дополнительной информацией
        
        Возвращает:
        ------------
        pd.DataFrame : DataFrame с колонками:
            - time: индекс времени
            - value: значение временного ряда
            - segment: номер сегмента
            - is_change_point: флаг точки разладки
        """
        data, segment_labels, change_points = self.generate()
        
        df = pd.DataFrame({
            'time': np.arange(self.n_timesteps),
            'value': data,
            'segment': segment_labels,
            'is_change_point': False
        })
        
        # Отметка точек разладки
        df.loc[change_points, 'is_change_point'] = True
        
        return df
    
    def visualize(
        self, 
        figsize: Tuple[int, int] = (15, 8),
        show_segments: bool = True,
        show_change_points: bool = True,
        title: str = "Временной ряд из GMM с разладками"
    ):
        """
        Визуализация сгенерированного временного ряда
        
        Параметры:
        -----------
        figsize : Tuple[int, int]
            Размер графика
        show_segments : bool
            Показывать ли фоновую заливку сегментов
        show_change_points : bool
            Показывать ли вертикальные линии в точках разладки
        title : str
            Заголовок графика
        """
        df = self.generate_with_info()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Фоновая заливка сегментов
        if show_segments:
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.segments)))
            for seg_idx, (start, end) in enumerate(self.segments):
                ax.axvspan(start, end - 1, alpha=0.2, color=colors[seg_idx], 
                          label=f'Сегмент {seg_idx}' if seg_idx < 10 else "")
        
        # Основной ряд
        ax.plot(df['time'], df['value'], 'b-', linewidth=1, alpha=0.7, label='Временной ряд')
        
        # Точки разладки
        if show_change_points:
            cp_times = df[df['is_change_point']]['time'].values
            for cp in cp_times:
                ax.axvline(x=cp, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Время')
        ax.set_ylabel('Значение')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Добавление информации о параметрах на график
        info_text = f"Компонент GMM: {self.n_components}\n"
        info_text += f"Сегментов: {len(self.segments)}\n"
        info_text += f"Точки разладки: {self.change_points}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax