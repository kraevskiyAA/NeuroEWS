# Score-Based Change-Point Detection via Diffusion Models

## 1. Классическая задача о разладке (Ширяев)

Наблюдаются $X_1, X_2, \ldots$ В момент $\theta$ (неизвестный) распределение меняется: $p_0 \to p_1$.

**Байесовская постановка.** Пусть $\theta$ имеет геометрическое априорное распределение $\mathsf{P}(\theta = k) = (1-\rho)^{k-1}\rho$. Апостериорная вероятность разладки:

$$\pi_t = \mathsf{P}(\theta \leq t \mid X_1, \ldots, X_t)$$

**Статистика Ширяева–Робертса:**

$$R_t = (1 + R_{t-1}) \cdot \Lambda_t(X_t), \quad R_0 = 0$$

где $\Lambda_t(x) = p_1(x)/p_0(x)$ — отношение правдоподобия.

Оптимальное правило остановки: $\tau^* = \inf\{t : R_t \geq h\}$.

**Ключевой объект:** $\log \Lambda_t(x) = \log p_1(x) - \log p_0(x)$.

---

## 2. Score-функция и тождество Стейна

**Определение.** Score-функция распределения $p$:

$$s(x) = \nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)} \in \mathbb{R}^d$$

**Тождество Стейна** — для любой достаточно гладкой $\varphi: \mathbb{R}^d \to \mathbb{R}^d$:

$$\mathsf{E}_p\bigl[s(x)^\top \varphi(x) + \nabla \cdot \varphi(x)\bigr] = 0$$

*Доказательство:* интегрирование по частям
$$\int \nabla_x \log p \cdot \varphi \cdot p\, dx = -\int \varphi \cdot \nabla p\, dx = \int p\, \nabla \cdot \varphi\, dx.$$

**Оператор Стейна** для $p_0$:

$$\mathcal{T}_{p_0}\varphi(x) = s_0(x)^\top \varphi(x) + \nabla \cdot \varphi(x)$$

Тогда:
- $x \sim p_0 \Rightarrow \mathsf{E}[\mathcal{T}_{p_0}\varphi(x)] = 0$
- $x \sim p_1 \neq p_0 \Rightarrow \mathsf{E}_{p_1}[\mathcal{T}_{p_0}\varphi] = \mathsf{E}_{p_1}\bigl[(s_0(x) - s_1(x))^\top \varphi(x)\bigr]$

Смещение определяется **разностью score-функций** $s_0 - s_1$ — это и есть сигнал разладки.

---

## 3. Диффузионные модели как машина для обучения score

**Прямой процесс (VP-SDE):**

$$dx = -\frac{\beta(t)}{2}\, x\, dt + \sqrt{\beta(t)}\, dW_t, \quad x_0 \sim p_0$$

Маргинальное распределение: $p_t(x_t) = \int p(x_t|x_0)\, p_0(x_0)\, dx_0$, где

$$p(x_t | x_0) = \mathcal{N}\!\left(x_t;\, \mu(t)\, x_0,\, \sigma^2(t)\, I\right), \quad \mu(t) = e^{-\frac{1}{2}\int_0^t \beta(s)\,ds},\quad \sigma^2(t) = 1 - \mu^2(t)$$

**Score маргинала** через условное математическое ожидание шума:

$$\nabla_{x_t} \log p_t(x_t) = -\frac{1}{\sigma(t)}\,\mathsf{E}\!\left[\varepsilon \mid x_t\right], \quad x_t = \mu(t)\,x_0 + \sigma(t)\,\varepsilon,\quad \varepsilon \sim \mathcal{N}(0,I)$$

**Обучение (Denoising Score Matching):**

$$\mathcal{L}(\theta) = \mathsf{E}_{t,\, x_0,\, \varepsilon}\!\left[\left\|s_\theta(x_t, t) + \frac{\varepsilon}{\sigma(t)}\right\|^2\right]$$

При $t \to 0$: $s_\theta(x, 0) \approx \nabla_x \log p_0(x)$.

**Обратный процесс (формула Андерсона):**

$$dx = \left[-\frac{\beta(t)}{2}\,x - \beta(t)\,\nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)}\, d\bar{W}_t$$

Score кодирует всю геометрию распределения и управляет обратной диффузией.

---

## 4. Score-статистика и её динамика

### Шаг 1. Обучение на доразладочных данных

По $x_1, \ldots, x_n \sim p_0$ обучаем:
$$s_\theta(x) \approx s_0(x) = \nabla_x \log p_0(x)$$

### Шаг 2. Stein-приращение

Для тестовой функции $\varphi: \mathbb{R}^d \to \mathbb{R}^d$ определяем:

$$\xi_t = \mathcal{T}_{s_\theta}\varphi(X_t) = s_\theta(X_t)^\top \varphi(X_t) + \nabla \cdot \varphi(X_t)$$

**Динамика процесса $Z_t = \sum_{i=1}^t \xi_i$:**

$$Z_t = Z_{t-1} + \xi_t, \quad Z_0 = 0$$

$$\mathsf{E}[\xi_t \mid \theta > t] = 0 \qquad \text{(тождество Стейна — мартингал до разладки)}$$

$$\mathsf{E}[\xi_t \mid \theta \leq t] = \mathsf{E}_{p_1}\!\left[(s_0(X) - s_1(X))^\top \varphi(X)\right] =: \delta \neq 0$$

### Шаг 3. Выбор $\varphi$

**Канонический выбор** $\varphi(x) = s_\theta(x)$:

$$\xi_t = \|s_\theta(X_t)\|^2 + \nabla \cdot s_\theta(X_t)$$

$$\delta = \mathsf{E}_{p_1}\!\left[\|s_0(X)\|^2 + \nabla \cdot s_0(X)\right] = \mathsf{E}_{p_1}\!\left[(s_0(X) - s_1(X))^\top s_0(X)\right]$$

**Оптимальный выбор** $\varphi$ (максимизирующий SNR):

$$\varphi^* = \arg\max_\varphi \frac{\left[\mathsf{E}_{p_1}(\mathcal{T}_{p_0}\varphi)\right]^2}{\mathsf{Var}_{p_0}(\mathcal{T}_{p_0}\varphi)}$$

При $\varphi \in \mathcal{H}_k$ (RKHS с ядром $k$) это даёт **Kernelized Stein Discrepancy (KSD)**:

$$\mathrm{KSD}^2(p_1, p_0) = \mathsf{E}_{x,x' \sim p_1}\!\left[u_{p_0}(x,x')\right]$$

где $u_{p_0}(x,x') = s_0(x)^\top \nabla_{x'} k(x,x') + s_0(x')^\top \nabla_x k(x,x') + \nabla_x \cdot \nabla_{x'} k(x,x') + s_0(x)^\top s_0(x')\, k(x,x')$.

---

## 5. Применение теории Ширяева

### CUSUM на $Z_t$

$$M_t = Z_t - \min_{0 \leq k \leq t} Z_k, \quad \tau^{\mathrm{cusum}} = \inf\{t : M_t > h\}$$

### Байесовская постановка

Если приращения $\xi_t$ приближённо гауссовы $\mathcal{N}(\delta \cdot \mathbf{1}_{t \geq \theta},\, \sigma_\xi^2)$, то:

$$\Lambda_t = \exp\!\left(\frac{\delta}{\sigma_\xi^2}\,\xi_t - \frac{\delta^2}{2\sigma_\xi^2}\right)$$

и статистика Ширяева–Робертса:

$$R_t = (1 + R_{t-1}) \cdot \Lambda_t, \quad \tau^* = \inf\{t : R_t \geq h\}$$

### Задача оптимальной остановки и свободная граница

Функция ценности:

$$V(\pi) = \sup_\tau\, \mathsf{E}_\pi\!\left[c\,(\tau - \theta)^+\,\mathbf{1}_{\tau \geq \theta} + (1 - \pi_\tau)\right]$$

По теореме Ширяева $V(\pi)$ удовлетворяет **задаче со свободной границей**:

$$\mathcal{L}\, V(\pi) = 0 \quad \text{в } \{\pi < \pi^*\} \quad \text{(область продолжения)}$$
$$V(\pi) = 1 - \pi \quad \text{в } \{\pi \geq \pi^*\} \quad \text{(область остановки)}$$

где $\mathcal{L}$ — инфинитезимальный генератор процесса $\pi_t$, который через рекуррентную формулу выражается через $\Lambda_t$, а значит через $\xi_t$ и $s_\theta$.

---

## 6. Полная схема

```
Данные до разладки  x_1, ..., x_n ~ p_0
            |
            v
  Обучение диффузионной модели:
  s_θ(x) ≈ ∇ log p_0(x)   (denoising score matching)
            |
            v
  Новые наблюдения  X_{n+1}, X_{n+2}, ...
            |
            v
  ξ_t = s_θ(Xₜ)ᵀ φ(Xₜ) + ∇·φ(Xₜ)    ← Stein residual
  E[ξ_t | t < θ] = 0,  E[ξ_t | t ≥ θ] = δ ≠ 0
            |
            v
  Z_t = Z_{t-1} + ξ_t                  ← мартингал до θ, дрейф после
            |
            v
  R_t = (1 + R_{t-1}) · Λ(ξ_t)        ← Ширяев–Робертс
            |
            v
  τ* = inf{t : R_t ≥ h}                ← оптимальная остановка
```

---

## 7. Практические замечания

**Дивергенция $\nabla \cdot s_\theta(x)$ в высоких размерностях** вычисляется через оценку Хатчинсона:

$$\nabla \cdot s_\theta(x) \approx v^\top \nabla_x(s_\theta(x)^\top v), \quad v \sim \mathcal{N}(0, I)$$

что требует одного прохода через Jacobian-vector product.

**Многошумовая версия:** вместо $s_\theta(x, 0)$ удобно использовать $s_\theta(x_{\tilde{t}}, \tilde{t})$ при фиксированном уровне шума $\tilde{t} > 0$ — это даёт более гладкую статистику и лучше работает в высоких размерностях.

**Связь с Fisher divergence:**

$$J(p_1 \| p_0) = \mathsf{E}_{p_1}\!\left[\|s_0(x) - s_1(x)\|^2\right]$$

При $\varphi = s_0 - s_1$ статистика $\xi_t$ напрямую оценивает Fisher divergence между $p_0$ и $p_1$.

---

## 8. Условная постановка: модель с памятью

### Мотивация

Для видео и биржевых стаканов безусловное распределение $p_0(x_t)$ нестационарно даже в норме: следующий кадр зависит от предыдущих. Если детектировать разладку в безусловном $p_0$, то нормальная динамика будет постоянно генерировать ложные тревоги.

**Правильный объект:** условное распределение следующего наблюдения при известной истории:

$$p_0(x_t \mid \mathbf{h}_{t-1}), \quad \mathbf{h}_{t-1} = \mathrm{enc}(x_{t-l}, \ldots, x_{t-1})$$

Разладка — это изменение именно этого условного закона при той же истории.

---

### Архитектура модели

**Кодировщик истории:**

$$\mathbf{h}_{t-1} = \mathrm{RNN}_\phi(x_{t-l}, \ldots, x_{t-1}) \in \mathbb{R}^m$$

Возможные варианты: GRU, LSTM, Transformer (causal), S4/Mamba для длинных контекстов.

**Условная диффузионная модель.** Прямой процесс по $x_0 = x_t$:

$$dx_\tau = -\frac{\beta(\tau)}{2}\, x_\tau\, d\tau + \sqrt{\beta(\tau)}\, dW_\tau, \quad \tau \in [0, T]$$

Денойзинговая сеть кондиционируется на историю $\mathbf{h}_{t-1}$ (например, через cross-attention или FiLM):

$$\hat{\varepsilon}_\theta(x_\tau, \tau, \mathbf{h}_{t-1}) \approx \mathsf{E}[\varepsilon \mid x_\tau, \mathbf{h}_{t-1}]$$

**Условная score-функция:**

$$s_\theta(x_\tau, \tau, \mathbf{h}_{t-1}) = -\frac{\hat{\varepsilon}_\theta(x_\tau, \tau, \mathbf{h}_{t-1})}{\sigma(\tau)} \approx \nabla_{x_\tau} \log p_\tau(x_\tau \mid \mathbf{h}_{t-1})$$

При $\tau \to 0$:

$$s_\theta(x, 0, \mathbf{h}_{t-1}) \approx \nabla_x \log p_0(x \mid \mathbf{h}_{t-1})$$

**Функция потерь:**

$$\mathcal{L}(\theta, \phi) = \mathsf{E}_{t,\, \tau,\, \varepsilon}\!\left[\left\|\hat{\varepsilon}_\theta\!\left(\mu(\tau)\,x_t + \sigma(\tau)\,\varepsilon,\; \tau,\; \mathrm{RNN}_\phi(x_{t-l:t-1})\right) - \varepsilon\right\|^2\right]$$

RNN-кодировщик и диффузионная голова обучаются **совместно** на историческом корпусе.

---

### Условное тождество Стейна

Для условного распределения $p(x \mid \mathbf{h})$ тождество Стейна принимает вид: для любой $\varphi: \mathbb{R}^d \times \mathbb{R}^m \to \mathbb{R}^d$:

$$\mathsf{E}_{x \sim p(\cdot \mid \mathbf{h})}\!\left[s(x \mid \mathbf{h})^\top \varphi(x, \mathbf{h}) + \nabla_x \cdot \varphi(x, \mathbf{h})\right] = 0 \quad \text{для п.в. } \mathbf{h}$$

**Условный оператор Стейна:**

$$\mathcal{T}_{p_0(\cdot \mid \mathbf{h})}\varphi(x, \mathbf{h}) = s_0(x \mid \mathbf{h})^\top \varphi(x, \mathbf{h}) + \nabla_x \cdot \varphi(x, \mathbf{h})$$

Тогда:

$$\mathsf{E}\!\left[\mathcal{T}_{p_0(\cdot \mid \mathbf{h}_{t-1})}\varphi(X_t, \mathbf{h}_{t-1}) \;\Big|\; \mathcal{F}_{t-1},\; \theta > t\right] = 0$$

$$\mathsf{E}\!\left[\mathcal{T}_{p_0(\cdot \mid \mathbf{h}_{t-1})}\varphi(X_t, \mathbf{h}_{t-1}) \;\Big|\; \mathcal{F}_{t-1},\; \theta \leq t\right] = \mathsf{E}_{p_1(\cdot \mid \mathbf{h}_{t-1})}\!\left[(s_0(X \mid \mathbf{h}_{t-1}) - s_1(X \mid \mathbf{h}_{t-1}))^\top \varphi\right] =: \delta(\mathbf{h}_{t-1})$$

---

### Динамика условной Stein-статистики

**Условное Stein-приращение:**

$$\xi_t = \mathcal{T}_{s_\theta(\cdot \mid \mathbf{h}_{t-1})}\varphi(X_t, \mathbf{h}_{t-1}) = s_\theta(X_t, 0, \mathbf{h}_{t-1})^\top \varphi(X_t, \mathbf{h}_{t-1}) + \nabla_x \cdot \varphi(X_t, \mathbf{h}_{t-1})$$

**Динамика:** $Z_t = Z_{t-1} + \xi_t$, $Z_0 = 0$.

| Режим | $\mathsf{E}[\xi_t \mid \mathcal{F}_{t-1}]$ | Процесс $Z_t$ |
|---|---|---|
| $t < \theta$ (норма) | $0$ | условный мартингал |
| $t \geq \theta$ (разладка) | $\delta(\mathbf{h}_{t-1}) \neq 0$ | дрейф по траектории |

**Ключевое свойство:** $\delta(\mathbf{h}_{t-1})$ зависит от истории, но в среднем по истории:

$$\bar{\delta} = \mathsf{E}_{\mathbf{h}_{t-1}}\!\left[\delta(\mathbf{h}_{t-1})\right] = \mathsf{E}_{p_1}\!\left[(s_0(X \mid \mathbf{h}) - s_1(X \mid \mathbf{h}))^\top \varphi(X, \mathbf{h})\right]$$

отражает **среднее по всем возможным историям** расхождение условных распределений.

**Канонический выбор** $\varphi(x, \mathbf{h}) = s_\theta(x, 0, \mathbf{h})$:

$$\xi_t = \left\|s_\theta(X_t, 0, \mathbf{h}_{t-1})\right\|^2 + \nabla_x \cdot s_\theta(X_t, 0, \mathbf{h}_{t-1})$$

Это скалярная величина — **условная энергия Стейна** — которая равна нулю в среднем под $p_0(\cdot \mid \mathbf{h}_{t-1})$ и отклоняется при сдвиге условного распределения.

---

### Интерпретация через инновации

Аналогия с фильтром Калмана: там инновация $x_t - \hat{x}_{t|t-1}$ — белый шум при правильной модели. Здесь:

$$\xi_t = \underbrace{\left\|s_\theta(X_t, 0, \mathbf{h}_{t-1})\right\|^2}_{\text{``score energy'' наблюдения}} + \underbrace{\nabla_x \cdot s_\theta(X_t, 0, \mathbf{h}_{t-1})}_{\text{``кривизна'' распределения в точке}}$$

— **полная мера несогласованности** $X_t$ с ожидаемым условным распределением $p_0(\cdot \mid \mathbf{h}_{t-1})$.

---

### Многомасштабная версия (multi-scale)

Вместо score при $\tau = 0$ (который дорого вычислять точно) удобно использовать несколько уровней шума $\tau_1 < \tau_2 < \ldots < \tau_K$ и агрегировать:

$$\xi_t = \sum_{k=1}^K w_k \cdot \xi_t^{(\tau_k)}, \quad \xi_t^{(\tau_k)} = \left\|\hat{\varepsilon}_\theta(x_t^{(\tau_k)}, \tau_k, \mathbf{h}_{t-1})\right\|^2$$

где $x_t^{(\tau_k)} = \mu(\tau_k)\,X_t + \sigma(\tau_k)\,\varepsilon_k$, $\varepsilon_k \sim \mathcal{N}(0, I)$ — зашумлённая версия $X_t$.

**Смысл:** при малых $\tau$ статистика чувствительна к тонким деталям (структура текстур, микро-движения цены), при больших $\tau$ — к грубым изменениям (смена сцены, резкий сдвиг уровня). Это аналог вейвлет-разложения, но адаптивное и данные-зависимое.

---

### Аппроксимация условного log-likelihood

Для Байесовской формулы нужно $\log p_0(X_t \mid \mathbf{h}_{t-1})$. Из условной диффузионной модели это вычисляется через **probability flow ODE** (точно, но дорого):

$$\log p_0(x \mid \mathbf{h}) = \log p_T(x_T \mid \mathbf{h}) + \int_0^T \left[\nabla_{x_\tau} \cdot f(x_\tau, \tau) - \tfrac{1}{2}g(\tau)^2\, \nabla_{x_\tau} \cdot s_\theta(x_\tau, \tau, \mathbf{h})\right] d\tau$$

**Практичная нижняя оценка (ELBO):**

$$\log p_0(x \mid \mathbf{h}) \geq -\mathcal{L}_{\mathrm{DSM}}(x, \mathbf{h}) + \mathrm{const}$$

где $\mathcal{L}_{\mathrm{DSM}}(x, \mathbf{h}) = \mathsf{E}_{\tau, \varepsilon}\!\left[\left\|\hat{\varepsilon}_\theta(x_\tau, \tau, \mathbf{h}) - \varepsilon\right\|^2\right]$ — средняя ошибка денойзинга.

Отношение правдоподобия для Ширяева–Робертса:

$$\log \Lambda_t \approx \mathcal{L}_{\mathrm{DSM}}^{(0)}(X_t, \mathbf{h}_{t-1}) - \mathcal{L}_{\mathrm{DSM}}^{(1)}(X_t, \mathbf{h}_{t-1})$$

где верхние индексы обозначают DSM-ошибку под моделью $p_0$ и $p_1$ соответственно.

---

## 9. Полная схема (условная версия)

```
  Корпус нормальных последовательностей
  (x_{t-l}, ..., x_{t-1}, x_t), t = 1,...,N
                  |
                  v
  Совместное обучение:
  RNN_φ: история → h_{t-1}              (кодировщик контекста)
  ε_θ(x_τ, τ, h_{t-1}): денойзер       (условная диффузия)
  Цель: E[||ε_θ(μ(τ)x_t + σ(τ)ε, τ, h_{t-1}) - ε||²] → min
                  |
                  v
  Стриминг новых данных X_t, t = n+1, n+2, ...
                  |
                  v
  h_{t-1} = RNN_φ(X_{t-l}, ..., X_{t-1})
                  |
                  v
  ξ_t = ||s_θ(X_t, 0, h_{t-1})||² + ∇·s_θ(X_t, 0, h_{t-1})
                          или
  ξ_t = Σ_k w_k ||ε_θ(X_t^(τ_k), τ_k, h_{t-1})||²    (multi-scale)
                  |
         E[ξ_t | F_{t-1}, t<θ] = 0
         E[ξ_t | F_{t-1}, t≥θ] = δ(h_{t-1}) ≠ 0
                  |
                  v
  Z_t = Z_{t-1} + ξ_t                  (условный мартингал → дрейф)
                  |
                  v
  R_t = (1 + R_{t-1}) · Λ(ξ_t)        (Ширяев–Робертс)
                  |
                  v
  τ* = inf{t : R_t ≥ h}                (оптимальная остановка)
```
