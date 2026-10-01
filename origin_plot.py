"""
Estilo OriginPro para matplotlib.
Reproduz o visual das figuras de referência:
- caixa fechada (4 spines pretas, ~1.0 pt)
- sem grid, fundo branco
- ticks internos (in), apenas bottom/left
- fonte serifada (Times New Roman), labels grandes
- legenda interna no canto superior esquerdo, sem moldura
- linhas finas (~1.5 pt) com paleta tipo Origin

Uso rápido:
    from origin_plot import plot_origin

    fig, ax = plot_origin(
        x, y_list, labels,
        xlabel="Temperature (°C)", ylabel="tan δ",
        xlim=(30, 250), ylim=(0.005, 0.042),
        savepath="figura.png",
    )

Ou passo a passo:
    from origin_plot import set_origin_style
    import matplotlib.pyplot as plt

    set_origin_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(x, y, label="MWL")
    ...
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Paleta inspirada no Origin (ordem das suas figuras:
# preto, vermelho, azul, verde, roxo, mostarda, ciano, marrom)
ORIGIN_COLORS = [
    "#2b2b2b",  # 0 preto (MWL)
    "#e41a1c",  # 1 vermelho
    "#1f66cc",  # 2 azul
    "#1a9e4b",  # 3 verde
    "#984ea3",  # 4 roxo
    "#cc9900",  # 5 mostarda / amarelo-queimado
    "#00bfc4",  # 6 ciano
    "#8c564b",  # 7 marrom
    "#377eb8",  # extras para >8 curvas
    "#4daf4a",
]

# Ciclo de marcadores do Origin (séries só com pontos, sem ligar):
# quadrado, círculo, triângulo-cima, triângulo-baixo, losango, ...
ORIGIN_MARKERS = ["s", "o", "^", "v", "D", "p", "*", "h", "X", "d"]


def set_origin_style(base_fontsize=12):
    """Aplica o estilo global tipo OriginPro via rcParams."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.1,
        "axes.labelcolor": "black",

        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIX"],
        "mathtext.fontset": "stix",
        "axes.labelsize": base_fontsize + 2,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize - 2,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 6.0,
        "ytick.major.size": 6.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.size": 3.5,
        "ytick.minor.size": 3.5,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        # ticks só embaixo/esquerda (como nas suas figuras)
        "xtick.top": False,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.right": False,

        "axes.grid": False,
        "grid.alpha": 0.0,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.4,
        "legend.numpoints": 1,
        "legend.scatterpoints": 1,
        "legend.markerscale": 1.0,
        "lines.linewidth": 1.6,
        "lines.markersize": 7.0,
        "lines.markeredgewidth": 0.7,
        "errorbar.capsize": 3.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def style_axes(ax, xlabel=None, ylabel=None,
               xlim=None, ylim=None,
               invert_x=False,
               xticks=None, yticks=None,
               labelpad_x=12, labelpad_y=10):
    """Ajustes finos de um Axes já criado."""
    from matplotlib.ticker import AutoMinorLocator
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.1)

    ax.tick_params(axis="both", which="major", direction="out",
                   length=6.0, width=1.0, colors="black",
                   top=False, right=False, labelsize=plt.rcParams["xtick.labelsize"])
    ax.tick_params(axis="both", which="minor", direction="out",
                   length=3.5, width=0.8, colors="black")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if xlabel:
        ax.set_xlabel(xlabel, labelpad=labelpad_x)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=labelpad_y)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if invert_x:
        ax.invert_xaxis()
    return ax


def add_origin_legend(ax, loc="upper left", bbox_to_anchor=(0.015, 0.985),
                      ncols=1, **kwargs):
    """Legenda interna sem moldura, como no Origin."""
    leg = ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor,
                    frameon=False, ncols=ncols, **kwargs)
    return leg


def plot_origin(x, y_list, labels, xlabel="", ylabel="",
                xlim=None, ylim=None, xticks=None, yticks=None,
                invert_x=False, figsize=(6.4, 5.2),
                colors=None, markers=None, markersize=7.0,
                linestyle="none", linewidth=1.6,
                yerr=None, capsize=3.0, elinewidth=1.0,
                legend_loc="upper left", savepath=None, dpi=300,
                show=False, title=None):
    """
    Plota uma ou várias séries no estilo OriginPro.

    x: array 1D
    y_list: lista de arrays (ou array 2D [n_curvas, n_pontos])
    labels: lista de strings (mesmo tamanho de y_list)
    linestyle: "none" (padrão: só marcadores, um por série) ou "-" etc.
    yerr: escalar, lista ou array — barras de incerteza por série.
    """
    import numpy as np
    set_origin_style()

    if colors is None:
        colors = ORIGIN_COLORS
    if markers is None:
        markers = ORIGIN_MARKERS

    y_arr = list(y_list)
    fig, ax = plt.subplots(figsize=figsize)

    for i, (y, lab) in enumerate(zip(y_arr, labels)):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        yy = np.asarray(y, dtype=float)
        ee = None
        if yerr is not None:
            ee = yerr[i] if isinstance(yerr, (list, tuple)) else yerr
        if ee is not None:
            ax.errorbar(x, yy, yerr=ee, label=lab, color=c,
                        fmt=m, ms=markersize, capsize=capsize,
                        elinewidth=elinewidth, markeredgewidth=0.7,
                        linestyle=linestyle, linewidth=linewidth)
        else:
            ax.plot(x, yy, label=lab, color=c,
                    marker=m, ms=markersize, markeredgewidth=0.7,
                    markeredgecolor=c, linestyle=linestyle,
                    linewidth=linewidth)

    style_axes(ax, xlabel=xlabel, ylabel=ylabel,
               xlim=xlim, ylim=ylim,
               invert_x=invert_x, xticks=xticks, yticks=yticks)
    add_origin_legend(ax, loc=legend_loc)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_from_dataframe(df, xcol, ycols, xlabel="", ylabel="", **kwargs):
    """Atalho para DataFrame do pandas: df[xcol] vs df[ycols]."""
    x = df[xcol].to_numpy()
    y_list = [df[c].to_numpy() for c in ycols]
    return plot_origin(x, y_list, labels=ycols,
                       xlabel=xlabel or xcol, ylabel=ylabel, **kwargs)
