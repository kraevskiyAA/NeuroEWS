# Редукция диффузионной модели к задаче Ширяева для $\mathcal{N}(0,1)$

**Идея:** вместо высокоразмерных наблюдений $X_t \in \mathbb{R}^d$ с неизвестным $p_1$ перейти к скалярной статистике с **явно известными** обоими распределениями.

---

## Шаг 1 — Нормированная ошибка денойзинга

Для каждого $X_t$ вычисляем:

$$L_t = \left\|\hat\varepsilon_\theta\!\left(\mu(\tau)\, X_t + \sigma(\tau)\,\varepsilon,\;\tau,\;\mathbf{h}_{t-1}\right) - \varepsilon\right\|^2, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

По hold-out выборке $X^{(0)} \sim p_0$ оцениваем $\mu_0 = \mathbb{E}[L_t^{(0)}]$, $\sigma_0 = \mathrm{Std}[L_t^{(0)}]$ и нормируем:

$$\boxed{Z_t = \frac{L_t - \mu_0}{\sigma_0}}$$

---

## Шаг 2 — Распределения $Z_t$ известны точно

| Режим | Распределение |
|---|---|
| До разладки $\;(X_t \sim p_0)$ | $Z_t \approx \mathcal{N}(0,\,1)$ |
| После разладки $\;(X_t \sim p_1)$ | $Z_t \approx \mathcal{N}(\delta,\,1)$, где $\delta > 0$ |

**Обоснование $\delta > 0$.** По принципу минимума MSE оптимальный денойзер для $p_1$ отличается от обученного для $p_0$:

$$\mathbb{E}_{p_1}\!\left[\|\hat\varepsilon_\theta - \varepsilon\|^2\right] = \underbrace{\mathbb{E}_{p_1}\!\left[\|\hat\varepsilon_1 - \varepsilon\|^2\right]}_{\text{минимальная ошибка}} + \underbrace{\mathbb{E}_{p_1}\!\left[\|\hat\varepsilon_\theta - \hat\varepsilon_1\|^2\right]}_{= \delta\,\sigma_0 > 0 \text{ при } p_1 \neq p_0}$$

---

## Шаг 3 — Точное отношение правдоподобий

$$\boxed{\ell_t = \log\frac{p_1(Z_t)}{p_0(Z_t)} = \delta\, Z_t - \frac{\delta^2}{2}}$$

$$\Lambda_t = \exp\!\left(\delta\, Z_t - \frac{\delta^2}{2}\right)$$

---

## Шаг 4 — Статистика Ширяева–Робертса

$$\boxed{R_t = (1 + R_{t-1})\cdot\Lambda_t, \qquad R_0 = 0}$$

Апостериорная вероятность разладки (с дискретной интенсивностью $\lambda_d = 1 - e^{-\lambda\Delta t}$):

$$\pi_t = \frac{\left[\pi_{t-1} + (1-\pi_{t-1})\,\lambda_d\right]\Lambda_t}{\left[\pi_{t-1} + (1-\pi_{t-1})\,\lambda_d\right]\Lambda_t + (1-\pi_{t-1})(1-\lambda_d)}$$

$$\pi_t \approx \frac{\lambda_d\, R_t}{1 + \lambda_d\, R_t}$$

---

## Шаг 5 — Оптимальный порог из Section 22 (Peskir & Shiryaev)

Параметры задачи: $\mu = \delta$, $\sigma = 1$, $\gamma = \delta^2/2$, $\lambda$ (интенсивность разладки), $c$ (стоимость задержки).

**Вспомогательная функция:**

$$\alpha(\pi) = \log\!\frac{\pi}{1-\pi} - \frac{1}{\pi}$$

**Функция для smooth-fit уравнения:**

$$\psi(\pi) = -\frac{c}{\gamma}\,e^{-\frac{\lambda}{\gamma}\alpha(\pi)}\int_0^\pi \frac{e^{\frac{\lambda}{\gamma}\alpha(\rho)}}{\rho\,(1-\rho)^2}\,d\rho$$

**Оптимальный порог** $A_*$ — единственный корень:

$$\psi(A_*) = -1$$

*Существование и единственность:* $\psi(0^+) = 0$, $\psi(\pi) \to -\infty$ при $\pi \to 1$, $\psi$ монотонно убывает.

---

## Шаг 6 — Оптимальное правило остановки

$$\boxed{\tau^* = \inf\!\left\{t \geq 1 : \pi_t \geq A_*\right\}}$$

Эквивалентно: $\tau^* = \inf\!\left\{t : R_t \geq h^*\right\}$, где $h^* = A_* \,/\, [\lambda_d\,(1 - A_*)]$.

---

## Итоговая схема

```
ОФФЛАЙН (калибровка):
──────────────────────────────────────────────────────
Корпус {X_t} ~ p_0  →  обучить RNN_φ + ε_θ
Калибровочная выборка  →  μ_0, σ_0
Задать: δ_min, λ, c  →  γ = δ_min² / 2
Найти A* из ψ(A*) = -1  [бисекция на (0,1)]

ОНЛАЙН (детекция):
──────────────────────────────────────────────────────
Получен X_t
  ↓
h_{t-1} = RNN_φ(X_{t-l}, ..., X_{t-1})
  ↓
ε ~ N(0, I)
L_t = ‖ε_θ(μ(τ)X_t + σ(τ)ε, τ, h_{t-1}) − ε‖²
Z_t = (L_t − μ_0) / σ_0          ←  ~N(0,1) под p_0,  ~N(δ,1) под p_1
  ↓
Λ_t = exp(δ_min · Z_t − δ_min²/2)
R_t = (1 + R_{t-1}) · Λ_t
π_t = λ_d R_t / (1 + λ_d R_t)
  ↓
если π_t ≥ A*  →  ТРЕВОГА
```

---

## Ключевой вывод

Ценой сжатия $X_t \in \mathbb{R}^d \;\to\; Z_t \in \mathbb{R}$ мы получаем **точную** постановку задачи Ширяева с явно известными:

$$p_0 = \mathcal{N}(0,\,1), \qquad p_1 = \mathcal{N}(\delta,\,1)$$

Весь байесовский аппарат Section 22 (Peskir & Shiryaev, 2006) применяется без каких-либо приближений. Оптимальный порог $A_*$ минимизирует байесовский риск:

$$\rho(\tau) = c\,\mathbb{E}\!\left[(\tau - \theta)^+\right] + \mathbb{P}(\tau < \theta)$$
