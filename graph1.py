import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.special import kv

mp.mp.dps = 50  # precisão para os polilogaritmos

# ---------------------------------
# Parâmetros
# ---------------------------------
betas = [0.2, 0.4, 0.5, 0.7, 0.9]
r_vals = np.linspace(0.1, 1.5, 180)   # r = a / \tilde{b}

n_max = 30
q_max = 30
q_arr = np.arange(1, q_max + 1, dtype=float)

# ---------------------------------
# Funções auxiliares
# ---------------------------------
def polylog_pair(s, beta):
    """
    Calcula Li_s(e^{2πiβ}) + Li_s(e^{-2πiβ}) = 2 Re[Li_s(e^{2πiβ})]
    """
    z = mp.e ** (2j * mp.pi * beta)
    return float(2 * mp.re(mp.polylog(s, z)))

def bessel_sum_x(x, r, q_arr):
    """
    Calcula:
        sum_q (x/q)^2 K_2(2π q r x)
    com x >= 0
    """
    if x < 1e-14:
        return np.sum(1.0 / (2.0 * np.pi**2 * q_arr**4 * r**2))

    arg = 2.0 * np.pi * q_arr * r * x
    return np.sum((x / q_arr)**2 * kv(2, arg))

def energy_density_btilde3(beta, r, n_max=30, q_arr=None):
    """
    Retorna:
        \tilde{b}^3 E / L^2
    em função de beta e r = a / \tilde{b}.
    """
    li5_part = (3.0 / 16.0) * (r / np.pi)**2 * polylog_pair(5, beta)
    li4_part = (r / (2.0 * np.pi**2)) * polylog_pair(4, beta)

    double_sum = 0.0
    for n in range(1, n_max + 1):
        double_sum += bessel_sum_x(n + beta, r, q_arr)
        double_sum += bessel_sum_x(n - beta, r, q_arr)

    last_sum = bessel_sum_x(beta, r, q_arr)

    y = -(1.0 / (8.0 * r)) * (li5_part + li4_part + double_sum + last_sum)
    return y

# ---------------------------------
# Gráfico
# ---------------------------------
plt.figure(figsize=(8, 5))

for beta in betas:
    y_vals = [energy_density_btilde3(beta, r, n_max=n_max, q_arr=q_arr) for r in r_vals]
    plt.plot(r_vals, y_vals, label=fr"$\beta = {beta}$")

plt.xlabel(r"$a/\tilde{b}$", fontsize=12)
plt.ylabel(r"$\tilde{b}^{3}E/L^{2}$", fontsize=12)
plt.title(r"$\tilde{b}^{3}E/L^{2}$ em função de $a/\tilde{b}$", fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend()

# Ajuste dos eixos como na imagem
plt.xlim(0.0, 1.5)
plt.xticks(np.arange(0.0, 1.41, 0.2))

plt.ylim(-1.0, 0.5)
plt.yticks(np.arange(-1.0, 0.41, 0.2))

plt.tight_layout()
plt.show()