import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.special import kv

# ============================================================
# Precisão numérica
# ============================================================
mp.mp.dps = 40

# ============================================================
# Parâmetros
# ============================================================
betas = [0.2, 0.4, 0.5, 0.7, 0.9]
r_vals = np.linspace(0.05, 3.5, 400)   # agora começa perto de 0

n_max = 50
q_max = 50
n_arr = np.arange(1, n_max + 1, dtype=float)
q_arr = np.arange(1, q_max + 1, dtype=float)

# ============================================================
# Função:
#     \tilde{b}^4 F_a / L^2
# ============================================================
def force_tildeb4_over_L2(beta, r):
    """
    Retorna a quantidade adimensional:
        \tilde{b}^4 F_a / L^2
    em função de beta e r = a / \tilde{b}.
    """

    # termo do polilogaritmo
    poly_sum = 0.0
    for delta in (+1.0, -1.0):
        z = mp.e ** (2j * mp.pi * beta * delta)
        poly_sum += float(mp.re(mp.polylog(5, z)))

    poly_term = -(3.0 / np.pi**2) * poly_sum

    # soma dupla
    double_sum_total = 0.0
    q_row = q_arr[None, :]

    for delta in (+1.0, -1.0):
        dn = delta * n_arr + beta
        dn_col = dn[:, None]
        abs_dn_col = np.abs(dn_col)

        arg = 2.0 * np.pi * r * q_row * abs_dn_col

        term1 = 3.0 * (dn_col / (r * q_row))**2 * kv(2, arg)
        term2 = 2.0 * np.pi * (dn_col**3 / (r * q_row)) * kv(1, arg)

        double_sum_total += np.sum(term1 + term2)

    # última soma
    arg0 = 2.0 * np.pi * r * q_arr * beta
    last_sum = np.sum(
        3.0 * (beta / (r * q_arr))**2 * kv(2, arg0)
        + 2.0 * np.pi * (beta**3 / (r * q_arr)) * kv(1, arg0)
    )

    result = -(1.0 / 128.0) * (poly_term + 16.0 * double_sum_total) \
             - (1.0 / 8.0) * last_sum

    return result

# ============================================================
# Gráfico
# ============================================================
plt.figure(figsize=(8.5, 6))
ax = plt.gca()

# fundo branco
ax.set_facecolor('white')

colors = ['blue', 'black', 'orange', 'red', 'green']

for beta, color in zip(betas, colors):
    y_vals = np.array([force_tildeb4_over_L2(beta, r) for r in r_vals])
    plt.plot(
        r_vals,
        y_vals,
        color=color,
        linewidth=1.6,
        linestyle='-',
        label=fr"${beta}$"
    )

plt.xlabel(r"$a/\tilde{b}$", fontsize=14)
plt.ylabel(r"$\tilde{b}^{4}F_{a}/L^{2}$", fontsize=14)

plt.xlim(0.0, 3.5)
plt.ylim(-0.03, 0.01)

plt.xticks(np.arange(0.0, 3.6, 0.5))
plt.yticks(np.arange(-0.03, 0.011, 0.01))

# grid cinza
plt.grid(True, which='major', color='gray', alpha=0.35, linewidth=0.8)
plt.minorticks_on()

leg = plt.legend(
    title=r"$\beta$",
    loc="lower right",
    frameon=True,
    facecolor="white",
    edgecolor="white",
    framealpha=1.0,
    fancybox=False,
    handlelength=3.0
)
leg.set_zorder(10)

plt.tight_layout()
plt.show()