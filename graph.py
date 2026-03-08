import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.special import kv

# Precisão para os polilogaritmos
mp.mp.dps = 50

# ----------------------------
# Parâmetros numéricos
# ----------------------------
ratios = [0.7, 0.9, 1.0, 1.2]   # r = a / \tilde{b}
beta_vals = np.linspace(0.0, 1.0, 250)

# Aumente esses valores se quiser mais precisão
n_max = 30
q_max = 30

# ----------------------------
# Funções auxiliares
# ----------------------------
def polylog_real_sum(s, beta):
    """
    Calcula:
        Li_s(e^{2π i β}) + Li_s(e^{-2π i β})
    que é igual a 2 Re[Li_s(e^{2π i β})].
    """
    z = mp.e ** (2j * mp.pi * beta)
    return float(2 * mp.re(mp.polylog(s, z)))

def scaled_bessel_term(x, q, r):
    """
    Calcula:
        (x/q)^2 * K_2(2π q r |x|)
    tratando corretamente o limite x -> 0,
    já que K_2(z) diverge mas a combinação é finita.
    """
    arg = 2.0 * np.pi * q * r * abs(x)

    if arg < 1e-10:
        # Limite de (x/q)^2 K2(2π q r |x|) quando x -> 0
        return 1.0 / (2.0 * np.pi**2 * q**4 * r**2)

    return (x / q)**2 * kv(2, arg)

def dimensionless_energy(beta, r, n_max=30, q_max=30):
    """
    Retorna:
        y = a^3 E / L^2
    em função de beta, com r = a / \tilde{b}.
    """

    # Parte com polilogaritmos: soma em δ = ±
    li5_part = (3.0 / 16.0) * (r / np.pi)**2 * polylog_real_sum(5, beta)
    li4_part = (r / (2.0 * np.pi**2)) * polylog_real_sum(4, beta)

    # Parte dupla: soma em δ = ±, n e q
    double_sum = 0.0
    for n in range(1, n_max + 1):
        for q in range(1, q_max + 1):
            for delta in (+1, -1):
                x = delta * n + beta
                double_sum += scaled_bessel_term(x, q, r)

    # Última soma
    last_sum = 0.0
    for q in range(1, q_max + 1):
        last_sum += scaled_bessel_term(beta, q, r)

    # Equação para a^3 E / L^2
    y = -(r**2 / 8.0) * (li5_part + li4_part + double_sum + last_sum)

    return y

# ----------------------------
# Cálculo e gráfico
# ----------------------------
plt.figure(figsize=(8, 5))

for r in ratios:
    y_vals = [dimensionless_energy(beta, r, n_max=n_max, q_max=q_max) for beta in beta_vals]
    plt.plot(beta_vals, y_vals, label=fr"$a/\tilde{{b}} = {r}$")

plt.xlabel(r"$\beta$", fontsize=12)
plt.ylabel(r"$a^3 E / L^2$", fontsize=12)
plt.title(r"Gráfico de $a^3E/L^2$ em função de $\beta$", fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()