import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Union
import warnings

class TimeSeriesDataset(Dataset):
    """
    Dataset для одномерных временных рядов с созданием последовательностей.
    
    Args:
        data: 1D numpy array или torch tensor с временным рядом
        sequence_length: Длина каждой последовательности (T)
        stride: Шаг скольжения окна (1 = полное перекрытие)
        normalize: Если True, применяет z-score нормализацию
        normalize_params: (mean, std) для нормализации. Если None, вычисляются из data
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        sequence_length: int = 100,
        stride: int = 1,
        normalize: bool = True,
        normalize_params: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        
        # Конвертация в torch tensor
        if isinstance(data, np.ndarray):
            data = torch.tensor(data).float()
        
        # Проверка размерности
        if data.dim() == 1:
            data = data.unsqueeze(-1)  # [T, 1]
        elif data.dim() == 2 and data.shape[1] == 1:
            pass  # Уже [T, 1]
        else:
            raise ValueError(f"Ожидался 1D ряд, получена форма {data.shape}")
        
        self.original_data = data.clone()
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Нормализация
        if normalize:
            if normalize_params is not None:
                self.mean, self.std = normalize_params
            else:
                self.mean = data.mean()
                self.std = data.std() + 1e-8  # Защита от деления на ноль
            
            self.data = (data - self.mean) / self.std
        else:
            self.mean = 0.0
            self.std = 1.0
            self.data = data
        
        # Создание последовательностей
        self.sequences = self._create_sequences()
        
        print(f"Dataset создан: {len(self.sequences)} последовательностей длиной {sequence_length}")
        print(f"Статистика: mean={self.mean:.4f}, std={self.std:.4f}")
    
    def _create_sequences(self) -> torch.Tensor:
        """
        Создание перекрывающихся последовательностей из временного ряда.
        """
        T = len(self.data)
        
        if self.sequence_length > T:
            warnings.warn(
                f"sequence_length ({self.sequence_length}) > длины ряда ({T}). "
                f"Данные будут дополнены нулями."
            )
            padding = self.sequence_length - T
            data_padded = torch.cat([
                self.data,
                torch.zeros(padding, 1)
            ], dim=0)
            return data_padded.unsqueeze(0)  # [1, sequence_length, 1]
        
        # Количество последовательностей
        n_sequences = (T - self.sequence_length) // self.stride + 1
        
        if n_sequences <= 0:
            raise ValueError(
                f"Недостаточно данных для создания последовательностей. "
                f"Длина ряда: {T}, длина последовательности: {self.sequence_length}"
            )
        
        sequences = []
        for i in range(n_sequences):
            start = i * self.stride
            end = start + self.sequence_length
            sequences.append(self.data[start:end])
        
        return torch.stack(sequences)  # [N, sequence_length, 1]
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Возвращает последовательность формы [sequence_length, 1]
        """
        return self.sequences[idx]
    
    def get_normalization_params(self) -> Tuple[float, float]:
        """Возвращает параметры нормализации для применения на тестовых данных"""
        return self.mean, self.std
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Обратное преобразование нормализации.
        
        Args:
            data: Нормализованные данные
        Returns:
            Данные в исходном масштабе
        """
        return data * self.std + self.mean


class TimeSeriesDatasetWithChangePoints(TimeSeriesDataset):
    """
    Расширенный Dataset с метками точек разладок (для валидации на синтетических данных).
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        change_points: Optional[np.ndarray] = None,
        sequence_length: int = 100,
        stride: int = 1,
        normalize: bool = True,
    ):
        super().__init__(data, sequence_length, stride, normalize)
        
        self.change_points = change_points
        
        if change_points is not None:
            self.cp_sequences = self._create_cp_sequences()
        else:
            self.cp_sequences = None
    
    def _create_cp_sequences(self) -> torch.Tensor:
        """
        Создание последовательностей меток разладок.
        """
        if self.change_points is None:
            return None
        
        cp = torch.tensor(self.change_points).float()
        if cp.dim() == 1:
            cp = cp.unsqueeze(-1)
        
        T = len(cp)
        n_sequences = (T - self.sequence_length) // self.stride + 1
        
        sequences = []
        for i in range(n_sequences):
            start = i * self.stride
            end = start + self.sequence_length
            sequences.append(cp[start:end])
        
        return torch.stack(sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Возвращает (данные, метки разладок) если метки есть, иначе только данные.
        """
        if self.cp_sequences is not None:
            return self.sequences[idx], self.cp_sequences[idx]
        return self.sequences[idx]


def create_dataloaders(
    data: Union[np.ndarray, torch.Tensor],
    sequence_length: int = 100,
    stride: int = 1,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 0,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создание train/val/test DataLoader из временного ряда.
    
    Args:
        data: Исходный временной ряд
        sequence_length: Длина последовательности
        stride: Шаг скольжения
        batch_size: Размер батча
        train_ratio: Доля данных для обучения
        val_ratio: Доля данных для валидации
        shuffle: Перемешивать ли последовательности
        num_workers: Количество воркеров для загрузки
        normalize: Нормализовать ли данные
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Разделение временного ряда (не перемешивая!)
    T = len(data) if isinstance(data, np.ndarray) else data.shape[0]
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    
    if isinstance(data, np.ndarray):
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
    else:
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
    
    # Вычисление параметров нормализации только на train
    if normalize:
        if isinstance(train_data, np.ndarray):
            mean = train_data.mean()
            std = train_data.std() + 1e-8
        else:
            mean = train_data.mean().item()
            std = train_data.std().item() + 1e-8
        normalize_params = (mean, std)
    else:
        normalize_params = None
    
    # Создание datasets
    train_dataset = TimeSeriesDataset(
        train_data,
        sequence_length=sequence_length,
        stride=stride,
        normalize=normalize,
        normalize_params=normalize_params,
    )
    
    val_dataset = TimeSeriesDataset(
        val_data,
        sequence_length=sequence_length,
        stride=stride,
        normalize=normalize,
        normalize_params=normalize_params,  # Используем параметры train
    )
    
    test_dataset = TimeSeriesDataset(
        test_data,
        sequence_length=sequence_length,
        stride=stride,
        normalize=normalize,
        normalize_params=normalize_params,  # Используем параметры train
    )
    
    # Создание dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    
    print(f"\nDataLoaders созданы:")
    print(f"  Train: {len(train_loader)} батчей ({len(train_dataset)} последовательностей)")
    print(f"  Val:   {len(val_loader)} батчей ({len(val_dataset)} последовательностей)")
    print(f"  Test:  {len(test_loader)} батчей ({len(test_dataset)} последовательностей)")
    
    return train_loader, val_loader, test_loader