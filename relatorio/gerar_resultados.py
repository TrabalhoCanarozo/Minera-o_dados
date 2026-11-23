"""Gera as tabelas (.tex), as figuras e os números usados no relatório.

Reproduz a análise do notebook Mineração_produtos.ipynb:
  - Rodada 1: Apriori com suporte mínimo de 0,1% e regras com lift >= 1.
  - Rodada 2: sem os itens dominantes, suporte mínimo de 0,05% e lift >= 1,1.
e acrescenta o teste exato de Fisher para cada regra apresentada.

Uso (a partir da raiz do repositório):
    pip install pandas mlxtend matplotlib scipy
    python relatorio/gerar_resultados.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.stats import false_discovery_control, fisher_exact

warnings.filterwarnings("ignore", category=DeprecationWarning)

AQUI = Path(__file__).resolve().parent
CSV = AQUI.parent / "Groceries_dataset.csv"
TABELAS = AQUI / "tabelas"
FIGURAS = AQUI / "figuras"
TABELAS.mkdir(exist_ok=True)
FIGURAS.mkdir(exist_ok=True)

SUPORTE_R1, LIFT_R1 = 0.001, 1.0
SUPORTE_R2, LIFT_R2 = 0.0005, 1.1
DOMINANTES = {"whole milk", "yogurt", "sausage", "frankfurter"}

# Tradução dos produtos que aparecem nas tabelas e figuras do relatório.
# Produtos fora da lista continuam com o nome original, em inglês.
TRADUCAO = {
    "artif. sweetener": "adoçante",
    "beverages": "bebidas",
    "bottled beer": "cerveja de garrafa",
    "bottled water": "água engarrafada",
    "citrus fruit": "frutas cítricas",
    "flour": "farinha",
    "frankfurter": "salsicha alemã",
    "hamburger meat": "carne moída",
    "ham": "presunto",
    "hard cheese": "queijo duro",
    "herbs": "ervas",
    "napkins": "guardanapos",
    "oil": "óleo",
    "other vegetables": "outros vegetais",
    "pasta": "macarrão",
    "pastry": "folhados/doces",
    "pickled vegetables": "conservas",
    "pip fruit": "maçã/pera",
    "pork": "carne de porco",
    "processed cheese": "queijo processado",
    "red/blush wine": "vinho tinto/rosé",
    "rolls/buns": "pãezinhos",
    "root vegetables": "vegetais de raiz",
    "salty snack": "salgadinho",
    "sausage": "salsicha",
    "shopping bags": "sacolas de compras",
    "soda": "refrigerante",
    "soft cheese": "queijo macio",
    "specialty chocolate": "chocolate especial",
    "sweet spreads": "pastas doces",
    "tropical fruit": "frutas tropicais",
    "whole milk": "leite integral",
    "yogurt": "iogurte",
}


def pt(item: str) -> str:
    return TRADUCAO.get(item, item)


def fmt(x: float, casas: int = 2) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt_p(p: float) -> str:
    if p < 0.0001:
        return "$<0{,}0001$"
    return fmt(p, 4 if p < 0.01 else 3)


def lado(itens) -> str:
    return ", ".join(sorted(pt(i) for i in itens))


def tex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def transacoes(df: pd.DataFrame) -> list[list[str]]:
    t = df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).tolist()
    return [x for x in t if len(x) >= 2]


def codificar(trans):
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(trans).transform(trans), columns=te.columns_)


def unicas(regras: pd.DataFrame) -> pd.DataFrame:
    """Remove pares espelhados (A->B e B->A): fica a direção de maior confiança."""
    regras = regras.copy()
    regras["chave"] = regras.apply(
        lambda r: frozenset([r["antecedents"], r["consequents"]]), axis=1
    )
    # Os dois sentidos têm o mesmo lift, a menos de erro de arredondamento:
    # arredondar evita que esse erro decida qual sentido fica.
    regras["lift_arred"] = regras["lift"].round(6)
    regras = regras.sort_values(["lift_arred", "confidence"], ascending=False)
    return regras.drop_duplicates("chave").drop(columns=["chave", "lift_arred"])


def fisher(enc: pd.DataFrame, ant, cons) -> float:
    a = enc[list(ant)].all(axis=1)
    b = enc[list(cons)].all(axis=1)
    tabela = [[(a & b).sum(), (a & ~b).sum()], [(~a & b).sum(), (~a & ~b).sum()]]
    return fisher_exact(tabela, alternative="greater")[1]


def tabela_regras(regras, enc, n, arquivo, alfa_bonf):
    """Escreve as linhas da tabela; p-valor em negrito se passa na correção de Bonferroni.

    Devolve quantas regras têm p < 0,05 e quantas passam em Bonferroni.
    """
    linhas, sig, sig_bonf = [], 0, 0
    for _, r in regras.iterrows():
        compras = int(round(r["support"] * n))
        p = fisher(enc, r["antecedents"], r["consequents"])
        sig += p < 0.05
        sig_bonf += p < alfa_bonf
        p_txt = f"\\textbf{{{fmt_p(p)}}}" if p < alfa_bonf else fmt_p(p)
        linhas.append(
            f"{tex_escape(lado(r['antecedents']))} $\\rightarrow$ {tex_escape(lado(r['consequents']))}"
            f" & {compras} & {fmt(r['support'] * 100, 2)} & {fmt(r['confidence'] * 100, 1)}"
            f" & {fmt(r['lift'])} & {p_txt} \\\\"
        )
    tabela = [
        r"\begin{tabularx}{\textwidth}{XrRRrr}",
        r"\toprule",
        r"Regra & Compras & Suporte\newline(\%) & Confiança\newline(\%) & Lift & $p$ \\",
        r"\midrule",
        *linhas,
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    (TABELAS / arquivo).write_text("\n".join(tabela) + "\n", encoding="utf-8")
    return int(sig), int(sig_bonf)


# Eixos dos gráficos com vírgula decimal
VIRGULA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def cores():
    return {"escuro": "#1F3B2D", "destaque": "#E07A2F", "neutro": "#9DB0A2"}


def main():
    c = cores()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    bruto = pd.read_csv(CSV)
    todas = bruto.groupby(["Member_number", "Date"])["itemDescription"].apply(list)
    n_linhas, n_produtos, n_clientes = len(bruto), bruto["itemDescription"].nunique(), bruto["Member_number"].nunique()

    # ---------------- Rodada 1 ----------------
    t1 = transacoes(bruto)
    enc1 = codificar(t1)
    n1 = len(t1)
    freq1 = apriori(enc1, min_support=SUPORTE_R1, use_colnames=True)
    regras1 = association_rules(freq1, metric="lift", min_threshold=LIFT_R1)
    unicas1 = unicas(regras1)
    top1 = unicas1.head(10)
    alfa1 = 0.05 / len(unicas1)  # Bonferroni sobre todas as regras únicas testáveis
    sig1, bonf1 = tabela_regras(top1, enc1, n1, "regras_rodada1.tex", alfa1)
    p1 = [fisher(enc1, r["antecedents"], r["consequents"]) for _, r in unicas1.iterrows()]
    bonf1_todas = sum(p < alfa1 for p in p1)
    q1 = false_discovery_control(p1)  # Benjamini-Hochberg

    # Gráfico: lift das 10 regras de topo
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rotulos = [f"{lado(a)} → {lado(b)}" for a, b in zip(top1["antecedents"], top1["consequents"])]
    cores_barras = [c["destaque"]] + [c["neutro"]] * (len(top1) - 1)
    ax.barh(rotulos[::-1], top1["lift"].tolist()[::-1], color=cores_barras[::-1])
    ax.axvline(1, color=c["escuro"], lw=0.8, ls="--")
    ax.set_xlabel("Lift")
    ax.xaxis.set_major_formatter(VIRGULA)
    for i, v in enumerate(top1["lift"].tolist()[::-1]):
        ax.text(v + 0.02, i, fmt(v), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "lift_rodada1.pdf")
    plt.close(fig)

    # Gráfico: suporte x confiança de todas as regras
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    sc = ax.scatter(regras1["support"] * 100, regras1["confidence"] * 100, c=regras1["lift"],
                    cmap="YlOrBr", s=14, edgecolors=c["escuro"], linewidths=0.2)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.15, 0.2, 0.3, 0.4, 0.6])
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Suporte (%) — escala logarítmica")
    ax.set_ylabel("Confiança (%)")
    barra = fig.colorbar(sc, ax=ax, label="Lift")
    barra.ax.yaxis.set_major_formatter(VIRGULA)
    fig.tight_layout()
    fig.savefig(FIGURAS / "dispersao_rodada1.pdf")
    plt.close(fig)

    # ---------------- Itens mais frequentes ----------------
    conjuntos = todas.apply(set)
    contagem = pd.Series([i for s in conjuntos for i in s]).value_counts()
    top_itens = (contagem.head(10) / len(conjuntos) * 100)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    cores_itens = [c["destaque"] if i in DOMINANTES else c["neutro"] for i in top_itens.index]
    ax.barh([pt(i) for i in top_itens.index][::-1], top_itens.values[::-1], color=cores_itens[::-1])
    ax.set_xlabel("% das compras que contêm o produto")
    for i, v in enumerate(top_itens.values[::-1]):
        ax.text(v + 0.2, i, fmt(v, 1) + "%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "itens_frequentes.pdf")
    plt.close(fig)

    # ---------------- Rodada 2 ----------------
    sem_dom = bruto[~bruto["itemDescription"].isin(DOMINANTES)]
    t2 = transacoes(sem_dom)
    enc2 = codificar(t2)
    n2 = len(t2)
    freq2 = apriori(enc2, min_support=SUPORTE_R2, use_colnames=True)
    regras2 = association_rules(freq2, metric="lift", min_threshold=LIFT_R2)
    unicas2 = unicas(regras2)
    top2 = unicas2.head(10)
    alfa2 = 0.05 / len(unicas2)
    sig2, bonf2 = tabela_regras(top2, enc2, n2, "regras_rodada2.tex", alfa2)
    p2 = [fisher(enc2, r["antecedents"], r["consequents"]) for _, r in unicas2.iterrows()]
    bonf2_todas = sum(p < alfa2 for p in p2)
    q2 = false_discovery_control(p2)

    # ---------------- Números para o texto ----------------
    melhor = top1.iloc[0]
    numeros = {
        "NumLinhas": fmt_int(n_linhas),
        "NumProdutos": fmt_int(n_produtos),
        "NumClientes": fmt_int(n_clientes),
        "NumCompras": fmt_int(len(todas)),
        "NumComprasRUm": fmt_int(n1),
        "MediaItens": fmt(todas.apply(len).mean(), 1),
        "MediaItensUnicos": fmt(conjuntos.apply(len).mean(), 1),
        "NumUmItemUnico": fmt_int(int((conjuntos.apply(len) < 2).sum())),
        "NumFrequentesRUm": fmt_int(len(freq1)),
        "NumRegrasRUm": fmt_int(len(regras1)),
        "NumRegrasUnicasRUm": fmt_int(len(unicas1)),
        "MaxConfRUm": fmt(regras1["confidence"].max() * 100, 1),
        "PctConfVinte": fmt((regras1["confidence"] < 0.2).mean() * 100, 0),
        "PctSupBaixo": fmt((regras1["support"] < 0.002).mean() * 100, 0),
        "CorrLiftConf": fmt(regras1[["lift", "confidence"]].corr().iloc[0, 1], 2),
        "MelhorLift": fmt(melhor["lift"]),
        "MelhorConf": fmt(melhor["confidence"] * 100, 1),
        "MelhorCompras": str(int(round(melhor["support"] * n1))),
        "MinComprasRUm": str(int(round(SUPORTE_R1 * n1))),
        "NumComprasRDois": fmt_int(n2),
        "NumFrequentesRDois": fmt_int(len(freq2)),
        "NumRegrasRDois": fmt_int(len(regras2)),
        "NumRegrasUnicasRDois": fmt_int(len(unicas2)),
        "MinComprasRDois": str(int(round(SUPORTE_R2 * n2))),
        "SigRUm": str(sig1),
        "SigRDois": str(sig2),
        "BonfRUm": str(bonf1),
        "BonfRDois": str(bonf2),
        "BonfTodasRUm": str(int(bonf1_todas)),
        "BonfTodasRDois": str(int(bonf2_todas)),
        "AlfaBonfRUm": fmt(alfa1, 5),
        "AlfaBonfRDois": fmt(alfa2, 5),
        "MenorPRUm": fmt(min(p1), 5),
        "MenorPRDois": fmt(min(p2), 4),
        "BHRUm": str(int((q1 < 0.05).sum())),
        "BHRDois": str(int((q2 < 0.05).sum())),
        "MenorQRUm": fmt(q1.min(), 3),
        "MenorQRDois": fmt(q2.min(), 2),
        "TopUmPct": fmt(top_itens.iloc[0], 1),
    }
    (TABELAS / "numeros.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in numeros.items()) + "\n",
        encoding="utf-8",
    )

    for k, v in numeros.items():
        print(f"{k:24s} {v}")
    faltando = {i for df in (top1, top2) for col in ("antecedents", "consequents")
                for s in df[col] for i in s if i not in TRADUCAO}
    faltando |= {i for i in top_itens.index if i not in TRADUCAO}
    print("\nSem tradução:", sorted(faltando) or "nenhum")


if __name__ == "__main__":
    main()
