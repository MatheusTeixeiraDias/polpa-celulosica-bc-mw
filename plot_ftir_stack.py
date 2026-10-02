"""FTIR das polpas organossolve — figuras em stack estilo OriginPro.

Le os CSV do Bruker exportados em `Brutos/FTIR` (cabecalho de 2 linhas,
eixo decrescente 4000 -> 550 cm-1, %T) e gera:

    figuras/ftir_stack.png/pdf          — stack de %T, eixo x invertido, sem numeros no y
    figuras/ftir_fingerprint.png/pdf    — mesma regiao, A normalizada pela banda
                                          de C-O da celulose (1032 cm-1), para
                                          comparar intensidades relativas

Uso:
    python plot_ftir_stack.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from origin_plot import set_origin_style, style_axes, ORIGIN_COLORS

BASE = Path(__file__).resolve().parent
BRUTOS = BASE / "Brutos" / "FTIR"
FIG = BASE / "figuras"

X_MIN, X_MAX = 4000.0, 550.0     # eixo invertido: 4000 -> 550
OFF = 20.0                        # deslocamento vertical entre curvas (%T)

FP_X_MIN, FP_X_MAX = 1800.0, 550.0
FP_OFF = 1.5                      # deslocamento entre curvas (A normalizado)
FP_LABEL_DY = 0.45

# (arquivo, rotulo) — ordem de baixo para cima no stack
SAMPLES = [
    ("Bagaco_Cana.csv", "Bagaço de cana"),
    ("Polpa_BC_Acetossolve.csv", "Polpa acetossolve (rend. 66,4%)"),
    ("Polpa_BC_Formossolve.csv", "Polpa formossolve (rend. 53,6%)"),
    ("Polpa_BC_Acetossolve_Branqueada.csv", "Acetossolve, branqueada (Q + P, rend. 72,7%)"),
    ("Polpa_BC_Formossolve_Branqueada.csv", "Formossolve, branqueada (Q + P)"),
]


def read_ftir(path: Path):
    a = np.loadtxt(path, delimiter=",", skiprows=2)
    x, T = a[:, 0], a[:, 1]
    m = (x <= X_MIN) & (x >= X_MAX)
    return x[m], T[m]


def plot_stack(data) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.125)

    for i, (lab, x, T) in enumerate(data):
        y = T + i * OFF
        c = ORIGIN_COLORS[i % len(ORIGIN_COLORS)]
        ax.plot(x, y, color=c, linewidth=1.1, zorder=3 + i)
        # rotulo direto na curva (regiao plana a esquerda, 4000-3600 cm-1)
        ylab = T[x >= 3900].mean() + i * OFF
        ax.annotate(lab, xy=(3980, ylab + 1.0), color=c, fontsize=9.5,
                    ha="left", va="bottom", annotation_clip=False)

    y_top = max(T.max() + i * OFF for i, (_, _, T) in enumerate(data))
    y_bot = min(T.min() + i * OFF for i, (_, _, T) in enumerate(data))
    style_axes(ax, xlabel="Número de onda (cm$^{-1}$)", ylabel="%T",
               xlim=(X_MIN, X_MAX), ylim=(y_bot - 6, y_top + 12),
               invert_x=False)
    ax.set_xticks(np.arange(4000, 999, -500))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel(ax.get_xlabel(), labelpad=8)
    ax.set_ylabel(ax.get_ylabel(), labelpad=14)

    fig.savefig(FIG / "ftir_stack.png", dpi=300, bbox_inches="tight",
                pad_inches=0.05)
    fig.savefig(FIG / "ftir_stack.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_fingerprint(data) -> None:
    """Regiao 1800-550 cm-1 em absorbancia normalizada pela banda 1032.

    As bandas relativas (lignina, hemicelulose) so podem ser comparadas apos
    normalizar: as intensidades absolutas variam muito entre amostras por
    causa do contato ATR e da linha de base (%T > 100 em algumas regioes).
    """
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.965, bottom=0.125)

    spans = []
    for i, (lab, x, T) in enumerate(data):
        A = -np.log10(np.clip(T, 1e-6, None) / 100.0)
        As = savgol_filter(A, 31, 3)
        m = (x <= FP_X_MIN) & (x >= FP_X_MAX)
        xr, Ar = x[m], As[m]
        ref = Ar[(xr <= 1055) & (xr >= 1010)].max()
        base = Ar.min()
        y = (Ar - base) / ref + i * FP_OFF
        spans.append(y.max() - i * FP_OFF)
        c = ORIGIN_COLORS[i % len(ORIGIN_COLORS)]
        ax.plot(xr, y, color=c, linewidth=1.1, zorder=3 + i)
        ylab = float(y[xr >= 1760].mean())
        ax.annotate(lab, xy=(1785, ylab + FP_LABEL_DY), color=c, fontsize=9.5,
                    ha="left", va="bottom", annotation_clip=False)

    style_axes(ax, xlabel="Número de onda (cm$^{-1}$)",
               ylabel="A normalizado (A$_{1032}$ = 1)",
               xlim=(FP_X_MIN, FP_X_MAX),
               ylim=(-0.3, (len(data) - 1) * FP_OFF + max(spans) + 0.5),
               invert_x=False)
    ax.set_xticks(np.arange(1800, 599, -200))
    ax.set_xlim(FP_X_MIN, FP_X_MAX)
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel(ax.get_xlabel(), labelpad=8)
    ax.set_ylabel(ax.get_ylabel(), labelpad=14)

    fig.savefig(FIG / "ftir_fingerprint.png", dpi=300, bbox_inches="tight",
                pad_inches=0.05)
    fig.savefig(FIG / "ftir_fingerprint.pdf", bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    missing = [f for f, _ in SAMPLES if not (BRUTOS / f).exists()]
    if missing:
        raise SystemExit(f"arquivos ausentes em {BRUTOS}: {missing}")

    data = [(lab, *read_ftir(BRUTOS / f)) for f, lab in SAMPLES]

    FIG.mkdir(exist_ok=True)
    set_origin_style()

    plot_stack(data)
    plot_fingerprint(data)
    print("figuras em", FIG)


if __name__ == "__main__":
    main()
