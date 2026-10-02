"""Análise quantitativa dos FTIR das polpas (Brutos/FTIR).

Para cada amostra:
  - converte %T em absorbância A = -log10(T/100) e suaviza (Savitzky-Golay);
  - localiza a posição de cada banda diagnóstica;
  - mede a altura da banda contra duas regiões de referência vizinhas
    (A_banda - <A_ref1, A_ref2>), forma robusta para espectros de ATR com
    linha de base inclinada e intensidade baixa;
  - normaliza pela banda de C-O da celulose a ~1032 cm-1, para comparar
    amostras com contato/espessura diferentes.

Saídas em dados/:
  ftir_posicoes.csv — posição (cm-1) das bandas por amostra (n/d = não detectada)
  ftir_alturas.csv  — altura bruta, altura normalizada (1032) e flag de detecção
  ftir_indices.csv  — índices literários (lignina, hemicelulose, LOI, ...)
  ftir_resumo.txt   — as três tabelas acima, em UTF-8

Uso:
    python analise_ftir.py
"""
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

BASE = Path(__file__).resolve().parent
BRUTOS = BASE / "Brutos" / "FTIR"
DADOS = BASE / "dados"

# mesma ordem das curvas em plot_ftir_stack.py (de baixo para cima)
SAMPLES = {
    "Bagaço": "Bagaco_Cana.csv",
    "Acetossolve": "Polpa_BC_Acetossolve.csv",
    "Formossolve": "Polpa_BC_Formossolve.csv",
    "Acetossolve branq.": "Polpa_BC_Acetossolve_Branqueada.csv",
    "Formossolve branq.": "Polpa_BC_Formossolve_Branqueada.csv",
}

# nome, janela da banda (cm-1), ref1 (cm-1), ref2 (cm-1)
BANDS = [
    ("nuOH 3340", 3550, 3150, (3700, 3650), (3050, 3000)),
    ("CH 2920", 2960, 2870, (3020, 3000), (2840, 2820)),
    ("C=O éster 1740", 1780, 1695, (1805, 1790), (1690, 1670)),
    ("H2O 1640", 1665, 1620, (1700, 1690), (1600, 1590)),
    ("arom 1595", 1610, 1575, (1620, 1615), (1565, 1555)),
    ("arom 1510", 1530, 1495, (1550, 1540), (1480, 1470)),
    ("CH2 1430", 1450, 1415, (1470, 1465), (1405, 1400)),
    ("CH def 1372", 1390, 1355, (1405, 1400), (1345, 1335)),
    ("G + COAc 1245", 1265, 1225, (1285, 1275), (1215, 1205)),
    # 1160 (C-O-C) fica no flanco da envolvente 1200-1000: as "refs" locais
    # caem na própria inclinação e o metodo de janelas nao resolve a
    # omopaulda — ver figura ftir_fingerprint.png
    ("C-O cel 1032", 1055, 1010, (1095, 1085), (975, 965)),
    ("anomérico 897", 915, 875, (935, 930), (860, 850)),
]
REF_BAND = "C-O cel 1032"


def read_ftir(path):
    a = np.loadtxt(path, delimiter=",", skiprows=2)
    x, T = a[:, 0], a[:, 1]
    A = -np.log10(np.clip(T, 1e-6, None) / 100.0)
    As = savgol_filter(A, 31, 3)
    return x, T, A, As


def band_height(x, y, clo, chi, r1, r2):
    clo, chi = min(clo, chi), max(clo, chi)
    mc = (x >= clo) & (x <= chi)
    pos = float(x[mc][np.argmax(y[mc])])
    c = float(np.mean(y[mc]))
    refs = []
    for hi, lo in (r1, r2):
        hi, lo = max(hi, lo), min(hi, lo)
        m = (x >= lo) & (x <= hi)
        refs.append(float(np.mean(y[m])))
    return pos, c - sum(refs) / len(refs)


def main():
    DADOS.mkdir(exist_ok=True)
    data = {k: read_ftir(BRUTOS / v) for k, v in SAMPLES.items()}

    pos, alt, aln = {}, {}, {}
    for lab, clo, chi, r1, r2 in BANDS:
        pos[lab], alt[lab] = {}, {}
        for s, (x, T, A, As) in data.items():
            p, h = band_height(x, As, clo, chi, r1, r2)
            pos[lab][s] = p
            alt[lab][s] = h
    for lab, *_ in BANDS:
        aln[lab] = {s: alt[lab][s] / alt[REF_BAND][s] for s in data}

    cols = list(SAMPLES)
    # altura minima para considerar a banda detectada (~5x o ruido do
    # espectro suavizado, sigma ~ 1.5e-5 u.a.); abaixo disso a posicao do
    # maximo local nao significa nada
    DETECT = 1.0e-4

    def ppos(lab, s):
        return "%d" % round(pos[lab][s]) if alt[lab][s] >= DETECT else "n/d"

    with open(DADOS / "ftir_posicoes.csv", "w", encoding="utf-8") as f:
        f.write("banda," + ",".join(cols) + "\n")
        for lab, *_ in BANDS:
            f.write(lab + "," + ",".join(ppos(lab, s) for s in cols) + "\n")

    with open(DADOS / "ftir_alturas.csv", "w", encoding="utf-8") as f:
        f.write("amostra,banda,pos_cm-1,altura,altura_norm_1032,deteccao\n")
        for s in cols:
            for lab, *_ in BANDS:
                f.write("%s,%s,%s,%.6f,%.4f,%s\n"
                        % (s, lab, ppos(lab, s), alt[lab][s], aln[lab][s],
                           "sim" if alt[lab][s] >= DETECT else "nao"))

    # índices de literatura (usando as alturas normalizadas)
    idx = [
        ("índice de lignina (1510/1032)", "arom 1510"),
        ("índice aromático (1595/1032)", "arom 1595"),
        ("índice de hemicelulose/éster (1740/1032)", "C=O éster 1740"),
        ("índice de água adsorvida (1640/1032)", "H2O 1640"),
        ("índice de G+COAc (1245/1032)", "G + COAc 1245"),
        ("índice CH2 1430/1032", "CH2 1430"),
        ("índice anomer. (897/1032)", "anomérico 897"),
        ("LOI 1430/897 (ordem lateral)", None),
    ]

    def idxval(band, s):
        return (aln[band][s] if band
                else alt["CH2 1430"][s] / alt["anomérico 897"][s])

    with open(DADOS / "ftir_indices.csv", "w", encoding="utf-8") as f:
        f.write("indice," + ",".join(cols) + "\n")
        for name, band in idx:
            f.write(name + ","
                    + ",".join("%.4f" % idxval(band, s) for s in cols) + "\n")

    # ---- resumo (console + dados/ftir_resumo.txt, UTF-8) ----
    lines = []
    lines.append("== posição das bandas (cm-1; n/d = banda não detectada) ==")
    lines.append("banda".ljust(18) + "".join(s.ljust(22) for s in cols))
    for lab, *_ in BANDS:
        lines.append(lab.ljust(18)
                     + "".join(ppos(lab, s).ljust(22) for s in cols))

    lines.append("")
    lines.append("== altura normalizada pela banda C-O da celulose (1032) ==")
    lines.append("banda".ljust(18) + "".join(s.ljust(22) for s in cols))
    for lab, *_ in BANDS:
        lines.append(lab.ljust(18)
                     + "".join(("%.4f" % aln[lab][s]).ljust(22) for s in cols))

    lines.append("")
    lines.append("== índices ==")
    lines.append("indice".ljust(42) + "".join(s.ljust(22) for s in cols))
    for name, band in idx:
        lines.append(name.ljust(42)
                     + "".join(("%.4f" % idxval(band, s)).ljust(22)
                               for s in cols))

    report = "\n".join(lines)
    (DADOS / "ftir_resumo.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print()
    print("arquivos em", DADOS)


if __name__ == "__main__":
    main()
