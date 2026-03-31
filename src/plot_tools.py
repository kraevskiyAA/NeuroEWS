import numpy as np
import matplotlib.pyplot as plt

def plot_3xN_comparison(
    array1: np.ndarray,  # [3, n]
    array2: np.ndarray,  # [3, n]
    labels: tuple = ('Array 1', 'Array 2'),
    component_names: tuple = ('Component 0', 'Component 1', 'Component 2'),
    x_axis: np.ndarray = None,  # Опциональная ось X (например, время)
    figsize: tuple = (12, 8),
):
    """
    Визуализация двух массивов 3×n в трёх сабплотах.
    """
    assert array1.shape == array2.shape == (3, array1.shape[1]), "Форма должна быть (3, n)"
    
    n = array1.shape[1]
    
    # Ось X по умолчанию: индексы
    if x_axis is None:
        x_axis = np.arange(n)
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    colors = ['blue', 'red']
    
    for i in range(3):
        ax = axes[i]
        
        # Построение обоих массивов
        ax.plot(x_axis, array1[i], label=labels[0], color=colors[0], linewidth=1.5, alpha=0.8)
        ax.plot(x_axis, array2[i], label=labels[1], color=colors[1], linewidth=1.5, alpha=0.8)
        
        # Оформление
        ax.set_ylabel(component_names[i])
        ax.set_title(f'{component_names[i]}: {labels[0]} vs {labels[1]}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    # Общая ось X
    axes[-1].set_xlabel('Index / Time Step')
    
    # Общий заголовок
    fig.suptitle('Comparison of Two 3×N Arrays', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    # Статистика
    print(f"📊 Статистика покомпонентного сравнения:")
    for i in range(3):
        diff = np.abs(array2[i] - array1[i])
        print(f"   Компонента {i}: mean|Δ|={diff.mean():.4f}, max|Δ|={diff.max():.4f}, corr={np.corrcoef(array1[i], array2[i])[0,1]:.4f}")