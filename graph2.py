import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.special import kv

# ============================================================
# Precisão para o polilogaritmo
# ============================================================
mp.mp.dps = 40

# ============================================================
# Parâmetros numéricos
# ============================================================
ratios = [1.4, 1.6, 1.8, 2.0]          # valores de a / \tilde{b}
betas = np.linspace(1e-4, 1 - 1e-4, 250)  # evita beta=0 e beta=1 exatos
n_max = 30
q_max = 30

n_arr = np.arange(1, n_max + 1, dtype=float)
q_arr = np.arange(1, q_max + 1, dtype=float)

# ============================================================
# Função que calcula a quantidade adimensional:
#            a^4 F_a / L^2
# ============================================================
def force_dimless(beta, r, n_arr, q_arr):
    """
    beta : parâmetro da condição quasi-periódica
    r    : razão a / \tilde{b}

    Retorna:
        a^4 F_a / L^2
    """

    # ---------- termo do polilogaritmo ----------
    z = mp.e ** (2j * mp.pi * beta)
    poly_term = -6.0 / np.pi**2 * float(mp.re(mp.polylog(5, z)))
    # Isso corresponde a somar delta=+ e delta=- no termo Li_5

    # ---------- soma dupla em delta, n, q ----------
    double_sum = 0.0
    q_row = q_arr[None, :]

    for delta in (+1.0, -1.0):
        dn = delta * n_arr + beta          # delta*n + beta
        dn_col = dn[:, None]
        abs_dn_col = np.abs(dn_col)

        arg = 2.0 * np.pi * r * q_row * abs_dn_col

        term1 = 3.0 * (dn_col / (r * q_row))**2 * kv(2, arg)
        term2 = 2.0 * np.pi * (dn_col**3 / (r * q_row)) * kv(1, arg)

        double_sum += np.sum(term1 + term2)

    # ---------- última soma em q ----------
    arg0 = 2.0 * np.pi * r * q_arr * beta
    last_sum_terms = (
        3.0 * (beta / (r * q_arr))**2 * kv(2, arg0)
        + 2.0 * np.pi * (beta**3 / (r * q_arr)) * kv(1, arg0)
    )
    last_sum = np.sum(last_sum_terms)

    # ---------- expressão final ----------
    result = -(r**4 / 128.0) * (poly_term + 16.0 * double_sum) \
             - (r**4 / 8.0) * last_sum

    return result


# ============================================================
# Geração das curvas
# ============================================================
plt.figure(figsize=(8, 6))

for r in ratios:
    y_vals = np.array([force_dimless(beta, r, n_arr, q_arr) for beta in betas])
    plt.plot(betas, y_vals, linewidth=2, label=fr"$a/\tilde{{b}} = {r}$")

plt.xlabel(r"$\beta$", fontsize=14)
plt.ylabel(r"$a^4 F_a/L^2$", fontsize=14)
plt.title(r"Força de Casimir por unidade de área", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()