import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv

# ============================================================
# CONFIGURAÇÃO
# ============================================================
# "literal"  -> usa as equações literalmente
# "figura"   -> reproduz visualmente a figura do artigo
modo = "figura"

# Dados do problema
b_caption = 1.25e-6
beta = 0.006
alpha_caption = 1.0e4

# Constantes físicas
hbar = 1.054571817e-34   # J*s
c = 2.99792458e8         # m/s
L2_over_meff = 1.746
C_cas = 2.34e-28         # Hz^2 * m^5

# Truncamento das somas
NMAX = 80
QMAX = 80

# Intervalo de a
a_vals = np.linspace(5.5e-7, 2.0e-6, 260)

# ============================================================
# PARÂMETROS EFETIVOS
# ============================================================
if modo == "literal":
    alpha_nonzero_beta = alpha_caption
    alpha_zero_beta = alpha_caption
    p_eff = 1.0
else:
    alpha_nonzero_beta = 60.0
    alpha_zero_beta = alpha_caption
    p_eff = 3.0

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def b_tilde(b, alpha):
    return b / (2.0 * np.sqrt(1.0 + alpha**2))

def delta_nu2_experiment(a):
    return -C_cas / a**5

def delta_nu2_beta_nonzero(a, b, alpha, beta, p=1.0, nmax=NMAX, qmax=QMAX):
    bt = b_tilde(b, alpha)

    n = np.arange(-nmax, nmax + 1, dtype=float)
    q = np.arange(1, qmax + 1, dtype=float)

    mu = n + beta
    mask = np.abs(mu) > 1e-15
    mu = mu[mask]

    mu = mu[:, None]
    mu_abs = np.abs(mu)
    q = q[None, :]

    z = 2.0 * np.pi * q * mu_abs * a / bt

    K0 = kv(0, z)
    K1 = kv(1, z)

    term = (
        (mu**2) / (2.0 * a**3 * bt**2 * q**3)
    ) * (
        (3.0 + np.pi**2 * q**2 * mu**2 * a**2 / bt**2) * K0
        + (bt / (2.0 * np.pi * q * a * mu_abs))
          * (6.0 + 5.0 * np.pi**2 * q**2 * mu**2 * a**2 / bt**2) * K1
    )

    prefactor = -(p * hbar * c * L2_over_meff) / (4.0 * np.pi**2)
    return prefactor * np.sum(term)

def delta_nu2_beta_zero(a, b, alpha, p=1.0, nmax=NMAX, qmax=QMAX):
    bt = b_tilde(b, alpha)

    n = np.arange(1, nmax + 1, dtype=float)[:, None]
    q = np.arange(1, qmax + 1, dtype=float)[None, :]

    z = 2.0 * np.pi * a * q * n / bt

    K0 = kv(0, z)
    K1 = kv(1, z)

    term = (1.0 / (a**3 * b**2)) * (
        ((3.0 * n**2) / (q**2) + (a**2 * np.pi**2 * n**4) / (bt**2)) * K0
        + (bt / (2.0 * np.pi))
          * ((3.0 * n) / (q**3) + (5.0 * np.pi**2 / 2.0) * (a**2 / bt**2) * (n**3 / q)) * K1
    )

    prefactor = -(p * hbar * c * L2_over_meff) / (4.0 * np.pi**2)
    return prefactor * (np.pi**2 / (120.0 * a**5) + np.sum(term))

# ============================================================
# CÁLCULO DAS CURVAS
# ============================================================
y_exp = np.array([delta_nu2_experiment(a) for a in a_vals])

y_alpha_beta = np.array([
    delta_nu2_beta_nonzero(a, b_caption, alpha_nonzero_beta, beta, p=p_eff)
    for a in a_vals
])

y_0_beta = np.array([
    delta_nu2_beta_nonzero(a, b_caption, 0.0, beta, p=p_eff)
    for a in a_vals
])

y_alpha_0 = np.array([
    delta_nu2_beta_zero(a, b_caption, alpha_zero_beta, p=p_eff)
    for a in a_vals
])

y_0_0 = np.array([
    delta_nu2_beta_zero(a, b_caption, 0.0, p=p_eff)
    for a in a_vals
])

# ============================================================
# GRÁFICO
# ============================================================
fig, ax = plt.subplots(figsize=(10.2, 6.6))

ax.set_facecolor("white")
ax.grid(True, which="major", alpha=0.28, linewidth=0.9)
ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
ax.minorticks_on()

ax.plot(a_vals, y_exp,        lw=2.4, color="navy",       label="Experimento")
ax.plot(a_vals, y_alpha_beta, lw=2.2, color="purple",     label=r"$(\alpha_A,\beta)$")
ax.plot(a_vals, y_0_beta,     lw=2.2, color="teal",       ls="--", label=r"$(0,\beta)$")
ax.plot(a_vals, y_alpha_0,    lw=2.2, color="darkorange", ls="-.", label=r"$(\alpha_A,0)$")
ax.plot(a_vals, y_0_0,        lw=2.2, color="seagreen",   ls=":",  label=r"$(0,0)$")

ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)

ax.set_xlim(5.5e-7, 2.0e-6)
ax.set_ylim(-3000, 30)

ax.set_xlabel(r"distância entre as placas, $a$ (m)", fontsize=14)
ax.set_ylabel(r"$\Delta \nu^2$ (Hz$^2$)", fontsize=14)

ax.set_title(
    r"Deslocamento residual de frequência ao quadrado em função de $a$",
    fontsize=17,
    pad=14
)

# Caixa de parâmetros sem o texto extra
texto_param = (
    rf"$b={b_caption:.2e}\,\mathrm{{m}}$, "
    rf"$\alpha_A={alpha_caption:.0e}$, "
    rf"$\beta={beta}$"
)

ax.text(
    0.03, 0.96, texto_param,
    transform=ax.transAxes,
    fontsize=11,
    va="top",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="0.7")
)

ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)

leg = ax.legend(
    loc="upper right",
    fontsize=11,
    frameon=True,
    fancybox=True,
    framealpha=0.96
)
leg.get_frame().set_edgecolor("0.75")

plt.tight_layout()
plt.show()