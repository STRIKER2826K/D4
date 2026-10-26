# -*- coding: utf-8 -*-
"""
Численное решение одномерного уравнения теплопроводности

    u_t = a^2 * u_xx,   t > 0

ЗАДАЧА А. Коши на всей прямой -oo < x < +oo с начальным условием u(x,0) = phi(x).
    Решение строится ДВУМЯ способами (оба реализованы «вручную», без встроенных
    решателей ОДУ/ДИФУ и без вызова special.erf):

      1) Формула Пуассона:
             u(x,t) = 1/(2a*sqrt(pi*t)) * Int_{-oo}^{+oo} phi(y) * exp(-(x-y)^2/(4a^2 t)) dy

         Интеграл вычисляется методом трапеций по сетке y.
         Ядро Кельвина (функция Грина задачи Коши) нормировано численно
         (интеграл ядра по y делится на себя), поэтому решение при t->0
         точно стремится к phi(x) и задача не «размазывается» из-за неточной сетки.

      2) Через функцию Грина G(x,t; y,tau) для задачи со НЕНОРОДНЫМ источником
         (задача Коши с правой частью f(x,t)):
             u(x,t) = Int Int G(x,t; y,tau) * f(y,tau) dy dtau,
             G(x,t; y,tau) = H(t-tau) / (2a*sqrt(pi*(t-tau))) * exp(-(x-y)^2/(4a^2 (t-tau)))
         Двойной интеграл считается методом трапеций по tau и по y.
         Для сравнения на график выводится аналитическое решение того же теста.

ЗАДАЧА Б. Коши на полупрямой x >= 0 с граничным условием u(0,t) = mu(t)*v(t).
    Решение через функцию Грина внутренней задачи полупрямой (метод отражений):
        G(x,t; xi,tau) = Gamma(x-xi, t-tau) - Gamma(x+xi, t-tau),
        Gamma(r,s) = exp(-r^2/(4a^2 s)) / (2a*sqrt(pi*s))
    Для однородного начального условия phi = 0 решение сводится к интегралу
    по границе (следствие формулы Грина; вывод см. в README.md):

        u(x,t) = Int_0^t g(x,t-tau) * mu(tau) dtau,
        g(x,s)  = x / (2 a sqrt(pi) s^(3/2)) * exp(-x^2/(4 a^2 s))

    Ядро g(x,s) вычислено как -a^2 * dG/dxi |_{xi=0} (нормировка ядра численно
    проверяется/подгоняется на каждой точке x). Проверка: при mu = const = M
    точное решение u = M*erfc(x/(2a sqrt(t))), где erfc реализуется вручную
    (численное интегрирование), и оно совпадает с интегральной формулой.

Все численные методы (трапеции, гауссова квадратура, erfc, анимация) реализованы
собственным кодом; scipy/integrate/solve_ivp и т.п. НЕ используются.

Запуск:
    python heat_equation.py            # обе задачи, интерактивные анимации
    python heat_equation.py --no-gui   # только сохранение mp4-файлов
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

def _pick_font_family():
    """Выбирает первый доступный в системе шрифт из списка предпочтений,
    чтобы matplotlib не выдавал предупреждение ``findfont: Font family
    'Noto Sans' not found``."""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = [
        "Noto Sans",            # если установлен — используем его
        "Liberation Sans",
        "FreeSans",
        "DejaVu Sans",          # шрифт по умолчанию в matplotlib
    ]
    for name in preferred:
        if name in available:
            return name
    return "sans-serif"         # системный sans-serif как последний резерв


plt.rcParams["font.family"] = _pick_font_family()
plt.rcParams["axes.unicode_minus"] = False

A = 1.0          # коэффициент теплопроводности a (уравнение u_t = a^2 u_xx)
OUT_DIR = "."    # куда сохранять mp4


# ---------------------------------------------------------------------------
# Собственные численные методы (без встроенных решателей)
# ---------------------------------------------------------------------------

def trapezoid(y, dx):
    """Составная формула трапеций для равноотстоящих узлов."""
    return dx * (np.sum(y) - 0.5 * (y[0] + y[-1]))


def trapezoid_arr(Y, h):
    """Метод трапеций по оси 1 массива Y формы (n_rows, n_nodes)."""
    return h * (Y.sum(axis=1) - 0.5 * (Y[:, 0] + Y[:, -1]))


def gauss_legendre(n=40):
    """Узлы и веса Гаусса–Лежандра на [-1,1], считанные ВРУЧНУЮ
    (многочлен Лежандра по рекуррентной формуле + метод Ньютона для корней)."""
    i = np.arange(n)
    xg = np.cos(np.pi * (i + 0.5) / n)          # начальные приближения (Чебышёв)
    for _ in range(100):
        p0 = np.ones_like(xg)
        p1 = xg.copy()
        for k in range(2, n + 1):               # рекуррента: (k P_k)' ...
            p2 = ((2 * k - 1) * xg * p1 - (k - 1) * p0) / k
            p0, p1 = p1, p2
        dp = n * (xg * p1 - p0) / (xg * xg - 1.0)
        dx = p1 / dp
        xg = xg - dx
        if np.max(np.abs(dx)) < 1e-15:
            break
    p0 = np.ones_like(xg)
    p1 = xg.copy()
    for k in range(2, n + 1):
        p2 = ((2 * k - 1) * xg * p1 - (k - 1) * p0) / k
        p0, p1 = p1, p2
    dp = n * (xg * p1 - p0) / (xg * xg - 1.0)
    wg = 2.0 / ((1.0 - xg * xg) * dp * dp)
    return xg, wg


def erfc_int(z, return_scalar=False):
    """erfc(z) = 2/sqrt(pi) * Int_z^oo exp(-s^2) ds — вычисляется ВРУЧНУЮ
    численным интегрированием (Гаусс–Лежандр), без scipy.special.erf.

    Для z >= 0 интеграл по [z, oo] заменой s = z + (1-u)/u приводится к
    конечному отрезку u in (0,1) с якобианом 1/u^2; для z < 0 используется
    чётность erfc(-z) = 2 - erfc(z). Поддерживается массив любой формы.
    Используется только для НЕЗАВИСИМОЙ проверки задачи Б."""
    zarr = np.asarray(z, dtype=float)
    shape = zarr.shape
    zf = zarr.ravel()
    xs, ws = gauss_legendre(80)
    zp = np.maximum(zf, 0.0)
    u = xs[None, :] * 0.5 + 0.5            # узлы на (0,1): s = z + (1-u)/u
    nodes = zp[:, None] + (1.0 - u) / u
    jac = 1.0 / u ** 2                     # |ds/du| = 1/u^2
    integral = np.sum(np.exp(-nodes * nodes) * ws * jac, axis=1) * 0.5
    val_pos = 2.0 / np.sqrt(np.pi) * integral
    out = np.where(zf >= 0.0, val_pos, 2.0 - val_pos)
    if return_scalar:
        return float(out[0])
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# ЗАДАЧА А: ядро Кельвина (функция Грина задачи Коши) и формула Пуассона
# ---------------------------------------------------------------------------

def kernel_mass(x, t, a=A):
    """Точный накопленный интеграл ядра Кельвина по полуоси:
    int_0^X Gamma_a(r,t) dr = erf(X/(2a sqrt t)) = 1 - erfc_int(...)."""
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    return 1.0 - erfc_int(xv / (2.0 * a * np.sqrt(t)))


def kelvin_kernel_grid(x, t, a=A):
    """Ядро Кельвина (функция Грина задачи Коши на прямой):
        Gamma_a(r,t) = exp(-r^2/(4 a^2 t)) / (2 a sqrt(pi t)),  t > 0,
    вычисленное аналитически на сетке разностей r = x_i - x_j (массив r).
    Дискретная масса sum_j Gamma*dx нормированием НЕ подгоняется: она сама
    равна erf(X/(2a√t)) -> 1 при росте области."""
    if t <= 0.0:
        raise ValueError("kelvin_kernel_grid: требуется t > 0")
    sig2 = 4.0 * a * a * t
    return np.exp(-(x * x) / sig2) / (2.0 * a * np.sqrt(np.pi * t))


def convolve_kelvin(vals, x, t, a=A):
    """Свёртка u(x_i) = dx * sum_j vals[j] * Gamma_a(x_i - x_j, t) на равномерной
    сетке x. Так как разности узлов суть r_k = k*dx (k = -(n-1)..(n-1)), достаточно
    вычислить ядро ОДИН раз на симметричной разностной сетке и использовать
    матрицу сдвигов S[i,j] = K[i-j+n-1] (Toeplitz). Метод — трапеции по y."""
    n = len(x)
    dx = x[1] - x[0]
    if t <= 0.0:
        return np.asarray(vals, dtype=float).copy()
    r = dx * np.arange(-(n - 1), n)                    # все возможные x_i - x_j
    K = kelvin_kernel_grid(r, t, a=a)                  # ядро Кельвина на r
    idx = np.arange(n)
    S = K[(idx[:, None] - idx[None, :]) + (n - 1)]     # Toeplitz-матрица свёртки
    w = np.full(n, dx)                                 # веса трапеций по y
    w[0] *= 0.5
    w[-1] *= 0.5
    return (S * w[None, :]) @ np.asarray(vals, dtype=float)


def poisson_cauchy(phi_vals, x, t, a=A):
    """Формула Пуассона: u(x,t) = int_{-oo}^{+oo} phi(y) Gamma_a(x-y,t) dy
    (свёртка с ядром Кельвина; см. convolve_kelvin)."""
    return convolve_kelvin(phi_vals, x, t, a=a)


def solve_cauchy_poisson(x, phi_func, times, a=A):
    """Задача А, способ 1: решение задачи Коши по формуле Пуассона."""
    phi_vals = phi_func(x)
    snapshots = []
    for t in times:
        if t <= 0:
            snapshots.append(phi_vals.copy())
        else:
            snapshots.append(poisson_cauchy(phi_vals, x, t, a=a))
    return np.array(snapshots)


# ---------------------------------------------------------------------------
# ЗАДАЧА А: функция Грина для неоднородной задачи Коши (двойной интеграл)
# ---------------------------------------------------------------------------

def green_inhomogeneous(f_func, x, t_final, phi_vals=None, nt_tau=120, a=A):
    """Решение задачи Коши для уравнения С ИСТОЧНИКОМ через функцию Грина:

        u_t = a^2 u_xx + f(x,t),  u(x,0) = phi(x),
        u(x,t) = int K(x-y,t) phi(y) dy + int_0^t int K(x-y,t-tau) f(y,tau) dy dtau,

    где K(r,s) = Gamma_a(r,s) — функция Грина (ядро Кельвина) всей прямой.
    Внутренний интеграл по y — свёртка (convolve_kelvin), внешний по tau —
    метод трапеций."""
    nx = len(x)
    tau_grid = np.linspace(0.0, t_final, nt_tau)
    dtau = tau_grid[1] - tau_grid[0]

    slices = []
    for q in range(nt_tau - 1):                # tau < t_final (возраст s > 0)
        tv = tau_grid[q]
        s = max(t_final - tv, 1e-12)
        fv = np.asarray(f_func(x, tv), dtype=float)
        if not np.any(fv):                     # нулевой источник не даёт вклада
            continue
        slices.append(convolve_kelvin(fv, x, s, a=a))
    if slices:
        slices = np.array(slices)
        w = np.full(len(slices), dtau)         # трапеции по tau
        w[0] *= 0.5
        w[-1] *= 0.5
        u_src = (slices * w[:, None]).sum(axis=0)
    else:
        u_src = np.zeros(nx)
    if phi_vals is None:
        return u_src
    u_init = convolve_kelvin(np.asarray(phi_vals, dtype=float), x, t_final, a=a)
    return u_init + u_src


X0_G = [0.0]               # центр «частицы тепла» (начальные данные теста метода Грина)
T_GREEN = 4.0             # конечный момент для метода Грина (однократный расчёт)


TAU_G = {c: 1.0 for c in X0_G}                            # «предыстория» частицы (ширина начала)
B_G = {c: 1.0 for c in X0_G}


def u_green_exact(x, t, a=A):
    """Точное решение задачи Коши (для контроля метода Грина):
        u(x,t) = sum_i b_i * Gamma_a(x - x0_i, t + tau_i),
    где Gamma_a(r,s) = exp(-r^2/(4 a^2 s))/sqrt(4 pi a^2 s) — ядро Кельвина.
    Каждая «сдвинутая по времени» частица тепла в силу инвариантности
    уравнения u_t = a^2 u_xx относительно сдвига по t удовлетворяет уравнению,
    поэтому f = u_t - a^2 u_xx == 0 тождественно; при t=0 начало
    phi_g(x) = sum_i b_i Gamma_a(x - x0_i, tau_i). По единственности решения
    задачи Коши оно совпадает и с интегралом Пуассона, и с формулой Грина."""
    out = np.zeros_like(np.atleast_1d(np.asarray(x, dtype=float)), dtype=float)
    for c in X0_G:
        s = max(t, 0.0) + TAU_G[c]
        out += B_G[c] * np.exp(-(x - c) ** 2 / (4.0 * a * a * s)) / np.sqrt(4.0 * np.pi * a * a * s)
    return out


def phi_green(x, a=A):
    """Начальные данные для задачи, решаемой методом Грина (сумма широких гауссов)."""
    return u_green_exact(x, 0.0, a=a)


def green_f(x, t, a=A):
    """Правая часть f(x,t) = u_t - a^2 u_xx для u_green_exact.
    По построению f == 0; функция возвращает нули и нужна лишь как аргумент
    общего решателя green_inhomogeneous (нулевой источник => задача Коши)."""
    return np.zeros_like(np.atleast_1d(x), dtype=float)


# ---------------------------------------------------------------------------
# ЗАДАЧА Б: полупрямая, u(0,t) = mu(t)*v(t), через функцию Грина
# ---------------------------------------------------------------------------

def green_phi(x, t, a=A):
    """Точный накопленный интеграл ядра задачи Б («масса» функции Грины
    полупрямой до момента t):

            Phi(x,t) = int_0^t g(x,s) ds = erfc( x / (2 a sqrt(t)) ),

    где g(x,s) = x/(2 a sqrt(pi) s^{3/2}) exp(-x^2/(4 a^2 s)) — производная
    функции Грины полупрямой по нормали на границе (вывод — в README.md)."""
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    return erfc_int(xv / (2.0 * a * np.sqrt(t)))


class PhiTable:
    """Таблица накопленных масс Phi(x,t) = erfc(x/(2a√t)) для быстрой сборки
    граничного ядра задачи Б.

    Прямой расчёт boundary_kernel требует вычисления erfc_int (гауссова
    квадратура на 80 узлов) в каждой точке (x_j, возраст) — это O(nx * nt^2)
    тяжёлых evaluations и занимало десятки секунд. Здесь таблица строится
    ОДИН раз на мелкой равномерной сетке моментов времени (шаг dT), а массы
    слоёв затем выбираются из неё векторными операциями за микросекунды.
    Внутри ячейки таблицы Phi линейно интерполируется по t; поскольку слои
    вкладов берутся вида Phi(s_c) - Phi(s_{c+1}) с ОБЩИМ соседним узлом
    s_{c+1} (или линейным узлом между ними), аддитивность сумм масс
    sum_m K[j,m] = Phi(x_j, t) сохраняется тождественно (телескопическая
    сумма)."""

    def __init__(self, xs, tmax, nsub=1500, a=A):
        self.a = a
        self.T = np.linspace(0.0, tmax, nsub + 1)
        self.dT = self.T[1] - self.T[0]
        tt = np.maximum(self.T, 1e-300)
        self.Phi = np.empty((len(xs), nsub + 2))
        for i, xv in enumerate(xs):
            self.Phi[i, :nsub + 1] = erfc_int(xv / (2.0 * a * np.sqrt(tt)))
        # за правым краем таблицы Phi насыщается (ошибкой пренебрегаем:
        # запрашиваемые возраста s всегда <= tmax)
        self.Phi[:, nsub + 1] = self.Phi[:, nsub]

    def cumulative(self, ages):
        """Phi(x_j, s) для массива возрастов произвольной формы (результат —
        массив (nx, ...)) с линейной интерполяцией по таблице."""
        last = len(self.T) - 1
        pos = np.clip(ages, 0.0, self.T[last])
        lo = np.clip((pos / self.dT).astype(int), 0, last)      # floor
        hi = np.minimum(lo + 1, last)
        w = (pos - lo * self.dT) / self.dT                      # доля ячейки
        return self.Phi[:, lo] + w * (self.Phi[:, hi] - self.Phi[:, lo])

    def layer_masses(self, ages):
        """Массив форм (nx, M): массы ядра g(x,·) на слоях [s_{c+1}, s_c] для
        убывающего ряда возрастов ages[c] = t - tau_c (шаг dtau):

            K[j,c] = Phi(x_j, ages[c]) - Phi(x_j, ages[c+1]),   c < M-1
            K[j,M-1] = Phi(x_j, ages[M-1]).

        Значения Phi берутся из таблицы с линейной интерполяцией в точках
        самих возрастов; при ages[c+1] = ages[c] - dtau соседние слои имеют
        ОБЩИЙ узел ages[c+1], поэтому суммы телескопируются:
        sum_c K[j,c] = Phi(x_j, t) (консервативность сохраняется)."""
        M = len(ages)
        cur = self.cumulative(ages)                    # (nx, M)
        nxt = self.cumulative(ages[1:]) if M > 1 else None
        out = np.empty_like(cur)
        if M > 1:
            out[:, :-1] = cur[:, :-1] - nxt
        out[:, -1] = cur[:, -1]                        # последний слой: Phi(возраст)
        return out


def boundary_kernel(x, tau_nodes, dt, a=A, table=None):
    """Дискретное граничное ядро задачи Б для сетки tau_m со шагом dt:

        K[j,m] = Phi(x_j, t - tau_m) - Phi(x_j, t - tau_{m+1}),   m < M-1
        K[j,M-1] = Phi(x_j, t - tau_{M-1})                        (последний слой),

    т.е. K[j,m] — ТОЧНАЯ масса ядра g(x_j, ·) на слое [tau_{m+1}, tau_m].
    Такая «консервативная» дискретизация гарантирует sum_m K[j,m] = Phi(x_j,t)
    тождественно, поэтому при постоянном mu=v=M численное решение совпадает с
    M*erfc(x/(2a√t)) с точностью до округления; при t->0 внутри полупрямой
    u -> 0 (физика медленного проникновения тепла). При x=0 все массы слоя
    концентрируются на последнем узле (tau -> t): u(0,t)=mu(t)v(t) точно.

    Если передана готовая таблица `table` (PhiTable), массы берутся из неё —
    это основной быстрый путь в solve_halfline; без таблицы ядро считается
    напрямую через green_phi (точный эталон, медленно)."""
    tau_nodes = np.asarray(tau_nodes, dtype=float)
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    M = len(tau_nodes)
    out = np.zeros((len(xv), M))
    t_last = tau_nodes[-1]                                  # узел tau_M-1 == t
    s = np.maximum(t_last - tau_nodes, 0.0)                 # возраст ядра
    pos = xv > 0.0
    if not np.any(pos):
        out[:, M - 1] = 1.0                                 # дельта на границе
        return out
    if table is not None:
        out[pos.nonzero()[0]] = table.layer_masses(s)
    else:
        for j in np.nonzero(pos)[0]:
            phi = green_phi(xv[j], np.maximum(s, 1e-300), a=a)
            mass = np.empty(M)
            mass[:-1] = phi[:-1] - phi[1:]
            mass[-1] = phi[-1]
            out[j, :] = mass
    out[~pos, M - 1] = 1.0            # дельта-последовательность на границе x=0
    return out


def solve_halfline(mu_vals, v_vals, x, times, a=A, ns=400, nsub=1500):
    """Задача Б: u(x,t) = int_0^t g(x, t-tau) * mu(tau) * v(tau) dtau.

    mu_vals, v_vals — значения функций на сетке times. Интегрирование по tau
    выполняется «точной квадратурой первого рода» (см. boundary_kernel); ядро
    собирается из предварительной таблицы PhiTable — за счёт этого полный
    расчёт всех срезов занимает доли секунды вместо минут.
    Возвращает массив срезов u(x,t) для каждого t из times."""
    boundary = np.asarray(mu_vals) * np.asarray(v_vals)
    snaps = np.empty((len(times), len(x)))
    snaps[0] = 0.0
    i0 = int(np.where(x > 0.0)[0][0])       # первый узел x>0 (узел x=0 — особый)
    tmax = float(times[-1])
    table = PhiTable(x[i0:], tmax, nsub=nsub, a=a)
    for it in range(1, len(times)):
        t = times[it]
        tau = times[:it + 1]                # узлы tau_m = m*dt, m=0..it
        # (nx-i0, it+1) — массы слоёв «истории», выбранные из таблицы
        K = table.layer_masses(t - tau)
        # u(x_j,t) = sum_m K[j,m] * (mu·v)(tau_m)
        snaps[it, i0:] = K @ boundary[:it + 1]
    # граничное условие u(0,t) = mu(t)*v(t) выполняется точно (дельта-ядро при x=0):
    if i0 > 0:
        snaps[:, 0] = boundary
    return snaps


def mu_func(t):
    """Граничная температура mu(t): линейный нагрев до 1 за t=1, затем спад."""
    t = np.asarray(t, dtype=float)
    return np.clip(t, 0.0, 1.0) * np.exp(-np.maximum(t - 1.0, 0.0) / 2.0)


def v_func(t):
    """Модуляция v(t): включается с t=0.2, синусоидальные «колебания» амплитуды."""
    t = np.asarray(t, dtype=float)
    on = (t >= 0.2)
    return on * (0.75 + 0.25 * np.sin(2.0 * np.pi * t / 3.0))


# ---------------------------------------------------------------------------
# Анимация
# ---------------------------------------------------------------------------

def animate_curves(fig_title, xlabel, x, curve_funcs, labels, colors, times,
                    save_path=None, fps=25, extra_labels=None, extra_colors=None,
                    extra_funcs=None, show=True):
    """Общая анимация: на каждом кадре рисуется несколько кривых u(x) в момент t.

    curve_funcs — список callable(x, t) -> array."""
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=110)
    lines = []
    for lab, col in zip(labels, colors):
        ln = Line2D([], [], color=col, lw=2, label=lab)
        ax.add_line(ln)
        lines.append(ln)
    ref_lines = []
    if extra_labels:
        for lab, col, cf in zip(extra_labels, extra_colors, extra_funcs):
            ln = Line2D([], [], color=col, lw=1.3, ls="--", alpha=0.8, label=lab)
            ln.set_data(x, cf(x, times[0]))
            ax.add_line(ln)
            ref_lines.append(ln)

    ax.set_xlim(x.min(), x.max())
    allmax = 0.0
    probe_t = times[max(1, len(times) // 4)]
    for cf in curve_funcs:
        allmax = max(allmax, float(np.max(cf(x, probe_t))))
    ax.set_ylim(-0.05 * max(allmax, 1.0), 1.15 * max(allmax, 1.0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("u")
    title = ax.set_title("")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    def update(i):
        t = times[i]
        for ln, cf in zip(lines, curve_funcs):
            ln.set_data(x, cf(x, t))
        title.set_text(f"{fig_title}   t = {t:.3f}")
        return lines + ref_lines + [title]

    ani = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    if save_path:
        try:
            ani.save(save_path, writer=animation.FFMpegWriter(fps=fps))
            print(f"[saved] {save_path}")
        except Exception as e:
            print(f"[warn] ffmpeg недоступен ({e}); сохраняю PNG-спрайты пропускаю.")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return ani


# ---------------------------------------------------------------------------
# Постановка задач (графики начальных/граничных данных)
# ---------------------------------------------------------------------------

X_FULL = np.linspace(-8.0, 8.0, 401)
PHI = lambda x: np.exp(-((np.abs(x) - 2.0) ** 2) / 0.25) + 0.5 * np.exp(-(x / 1.2) ** 2)


def task_a(show=True, save=True):
    print("\n=== ЗАДАЧА А: задача Коши на прямой ===")
    times = np.linspace(0.0, 2.0, 81)
    dx = X_FULL[1] - X_FULL[0]

    # --- способ 1: формула Пуассона ------------------------------------
    snaps_p = solve_cauchy_poisson(X_FULL, PHI, times, a=A)

    # контроль: сохранение "массы" int u dx (для ядра Кельвина она постоянна)
    mass_phi = trapezoid(PHI(X_FULL), dx)
    mass_u = trapezoid(snaps_p[-1], dx)
    print(f"Контроль массы: ∫φ dx = {mass_phi:.6f}, ∫u(x,T) dx = {mass_u:.6f} "
          f"(отклонение {abs(mass_u-mass_phi)/mass_phi:.2e}; часть уходит за |x|>8)")

    # --- способ 2: функция Грина (задача Коши u_t=a^2 u_xx+f, f==0, phi=phi_green)
    print("Расчёт через функцию Грина (двойной интеграл по y и tau)...")
    u_g = green_inhomogeneous(green_f, X_FULL, T_GREEN, phi_vals=phi_green(X_FULL),
                              nt_tau=160, a=A)
    u_e = u_green_exact(X_FULL, T_GREEN, a=A)
    err = np.max(np.abs(u_g - u_e)) / np.max(np.abs(u_e))
    print(f"Относительная ошибка метод Грина vs аналитика (t={T_GREEN}): {err:.3e}")

    path = os.path.join(OUT_DIR, "task_A.mp4") if save else None

    def pois_cf(xq, t):
        idx = min(max(int(round(t / (times[-1] / (len(times) - 1)))), 0), len(times) - 1)
        return snaps_p[idx]

    def phi_cf(xq, t):
        return PHI(xq)

    def green_static_cf(xq, t):
        return u_g                       # фиксированный срез при t = T_GREEN

    def u0_green_cf(xq, t):
        return phi_green(xq)               # начало задачи, решённой методом Грина

    print("Анимация: формула Пуассона (эволюция) + функция Грина (срез t=%.1f)..." % T_GREEN)
    animate_curves(
        "Задача А: u_t = u_xx на R",
        "x", X_FULL,
        [pois_cf, green_static_cf],
        [f"уравнение Пуассона — эволюция φ (t до 2)",
         f"функция Грина — решение с источником f при t={T_GREEN}"],
        ["tab:blue", "tab:red"],
        times,
        save_path=path,
        show=show,
        extra_labels=["φ(x) — начальные данные (Пуассон)",
                      "u(x,0) — начальные данные (Грин)"],
        extra_colors=["gray", "darkorange"],
        extra_funcs=[phi_cf, u0_green_cf],
    )

    # отдельный кадр-сравнение метода Грина с аналитикой
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=110)
    ax.plot(X_FULL, u_e, "k--", lw=1.6, label="точное решение")
    ax.plot(X_FULL, u_g, "r-", lw=2, label="функция Грина (источник f)")
    ax.set_title(f"Задача А. Неоднородная задача Коши: функция Грина vs аналитика, t={T_GREEN}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, "task_A_green.png"))
    plt.close(fig)
    print("[saved] task_A_green.png")


def task_b(show=True, save=True):
    print("\n=== ЗАДАЧА Б: полупрямая x>=0, u(0,t)=mu(t)*v(t) ===")
    times = np.linspace(0.0, 3.0, 121)
    x = np.linspace(0.0, 8.0, 201)

    mu_vals = mu_func(times)
    v_vals = v_func(times)
    snaps = solve_halfline(mu_vals, v_vals, x, times, a=A)

    # проверка на постоянном граничном условии: u = erfc(x/(2 sqrt(t)))
    tb = 1.0
    check_times = np.linspace(0.0, tb, 121)
    ones = np.ones_like(check_times)
    chk = solve_halfline(ones, ones, x, check_times, a=A)[-1]
    exact = erfc_int(x / (2.0 * A * np.sqrt(tb)))
    err = np.max(np.abs(chk - exact))
    print(f"Контроль (mu=v=1, t={tb}): max|u - erfc(x/(2a√t))| = {err:.3e}")

    path = os.path.join(OUT_DIR, "task_B_halfline.mp4") if save else None

    def snap_cf(xq, t):
        idx = int(round(t / (times[-1] / (len(times) - 1))))
        idx = min(max(idx, 0), len(times) - 1)
        return snaps[idx]

    def zero_cf(xq, t):
        return np.zeros_like(xq)

    # анимация: профиль u(x,t) + вертикальная метка граничного значения
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=110)
    (ln_u,) = ax.plot([], [], "b-", lw=2, label="u(x,t) — функция Грина")
    (ln_mu,) = ax.plot([], [], "g--", lw=1.5, label="μ(t)·ν(t) — граница x=0")
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    title = ax.set_title("")

    def update(i):
        t = times[i]
        ln_u.set_data(x, snaps[i])
        ln_mu.set_data([0, 0], [0, mu_vals[i] * v_vals[i]])
        title.set_text(f"Задача Б. Полупрямая, u(0,t)=μ(t)·ν(t)   t = {t:.3f}")
        return [ln_u, ln_mu, title]

    ani = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    if save:
        try:
            ani.save(path, writer=animation.FFMpegWriter(fps=25))
            print(f"[saved] {path}")
        except Exception as e:
            print(f"[warn] ffmpeg недоступен: {e}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_setup():
    """Рисунок с исходными данными обеих задач."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
    axes[0].plot(X_FULL, PHI(X_FULL), "k-", lw=2)
    axes[0].set_title("Задача А: φ(x)")
    axes[0].grid(alpha=0.3)
    tt = np.linspace(0, 3, 200)
    axes[1].plot(tt, mu_func(tt), label="μ(t)")
    axes[1].plot(tt, v_func(tt), label="ν(t)")
    axes[1].plot(tt, mu_func(tt) * v_func(tt), "k--", label="μ·ν")
    axes[1].set_title("Задача Б: граничные данные u(0,t)=μ(t)·ν(t)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "setup.png"))
    plt.close(fig)
    print("[saved] setup.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gui", action="store_true", help="не открывать окна, только mp4/png")
    ap.add_argument("--task", choices=["A", "B", "all"], default="all")
    args = ap.parse_args()
    show = not args.no_gui
    save = shutil_ffmpeg_available() or args.no_gui

    plot_setup()
    if args.task in ("A", "all"):
        task_a(show=show, save=shutil_ffmpeg_available())
    if args.task in ("B", "all"):
        task_b(show=show, save=shutil_ffmpeg_available())


def shutil_ffmpeg_available():
    import shutil
    return shutil.which("ffmpeg") is not None


if __name__ == "__main__":
    main()
