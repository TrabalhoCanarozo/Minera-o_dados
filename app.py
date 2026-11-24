"""Simulador interativo de cesta de compras com regras de associação (Apriori).

Executar localmente:
    pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.stats import fisher_exact

from traducao_produtos import TRADUCAO

AQUI = Path(__file__).resolve().parent
CSV = AQUI / "Groceries_dataset.csv"
REPO = "https://github.com/TrabalhoCanarozo/Minera-o_dados"
DOMINANTES = ("frankfurter", "sausage", "whole milk", "yogurt")
CORES = {"escuro": "#1F3B2D", "destaque": "#E07A2F", "neutro": "#9DB0A2", "claro": "#F6F3EA"}

EXEMPLOS = {
    "☕ Café da manhã": ["yogurt", "whole milk"],
    "🍝 Macarrão": ["pasta"],
    "🥪 Lanche": ["ham", "rolls/buns"],
    "🍷 Vinho": ["red/blush wine"],
    "🧀 Queijos": ["soft cheese", "hard cheese"],
}


# ------------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------------
def pt(item: str) -> str:
    return TRADUCAO.get(item, item)


def lado(itens) -> str:
    return " + ".join(sorted(pt(i) for i in itens))


def num(x: float, casas: int = 0) -> str:
    """Número no formato brasileiro: 14.963 ou 2,18."""
    texto = f"{x:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data(show_spinner=False)
def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["itemDescription"] = df["itemDescription"].str.strip()
    return df


@st.cache_data(show_spinner=False)
def montar_transacoes(removidos: tuple[str, ...]) -> pd.DataFrame:
    """Uma linha por compra (cliente + data), uma coluna verdadeiro/falso por produto."""
    df = carregar_dados()
    if removidos:
        df = df[~df["itemDescription"].isin(removidos)]
    trans = df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).tolist()
    trans = [t for t in trans if len(t) >= 2]  # mesmo critério do notebook
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(trans).transform(trans), columns=te.columns_)


@st.cache_data(show_spinner=False)
def minerar(removidos: tuple[str, ...], suporte_min: float):
    """Roda o Apriori e gera todas as regras com lift >= 1."""
    enc = montar_transacoes(removidos)
    freq = apriori(enc, min_support=suporte_min, use_colnames=True)
    if len(freq) < 2:
        return freq, pd.DataFrame()
    regras = association_rules(freq, metric="lift", min_threshold=1.0)
    if regras.empty:
        return freq, regras
    n = len(enc)
    regras["compras"] = (regras["support"] * n).round().astype(int)
    regras["p"] = [p_fisher(r, n) for _, r in regras.iterrows()]
    regras["regra"] = [f"{lado(a)} → {lado(c)}" for a, c in zip(regras["antecedents"], regras["consequents"])]
    return freq, regras.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def p_fisher(r: pd.Series, n: int) -> float:
    """Teste exato de Fisher (unilateral) a partir das contagens da regra."""
    ab = round(r["support"] * n)
    a = round(r["antecedent support"] * n)
    b = round(r["consequent support"] * n)
    tabela = [[ab, a - ab], [b - ab, n - a - b + ab]]
    return fisher_exact(tabela, alternative="greater")[1]


def sem_espelhos(regras: pd.DataFrame) -> pd.DataFrame:
    """A→B e B→A têm o mesmo lift: fica só o sentido de maior confiança."""
    if regras.empty:
        return regras
    chave = [frozenset([a, c]) for a, c in zip(regras["antecedents"], regras["consequents"])]
    tmp = regras.assign(_chave=chave, _lift=regras["lift"].round(6))
    tmp = tmp.sort_values(["_lift", "confidence"], ascending=False).drop_duplicates("_chave")
    return tmp.drop(columns=["_chave", "_lift"]).reset_index(drop=True)


def colocar_exemplo(itens: list[str], disponiveis: list[str]) -> None:
    st.session_state["carrinho"] = [i for i in itens if i in disponiveis]


# ------------------------------------------------------------------
# Página
# ------------------------------------------------------------------
st.set_page_config(page_title="O que os clientes compram juntos?", page_icon="🛒", layout="wide")

st.title("🛒 O que os clientes compram juntos?")
st.caption(
    "Mineração de regras de associação com o algoritmo **Apriori** no *Groceries Dataset* · "
    "Inteligência Computacional — Prof. Canarozzo"
)

# ---------------- Barra lateral ----------------
with st.sidebar:
    st.header("⚙️ Parâmetros")
    remover = st.toggle(
        "Remover itens dominantes",
        help="Tira leite integral, iogurte, salsicha e salsicha alemã antes de minerar, "
        "como na segunda rodada do trabalho.",
    )
    suporte_pct = st.slider(
        "Suporte mínimo (%)", 0.05, 1.00, 0.10, 0.05, format="%.2f",
        help="Fração mínima das compras em que o conjunto precisa aparecer.",
    )
    confianca_pct = st.slider(
        "Confiança mínima (%)", 0, 40, 0, 1,
        help="De quem leva A, quantos também levam B.",
    )
    lift_min = st.slider(
        "Lift mínimo", 1.0, 4.0, 1.0, 0.1,
        help="Quantas vezes A aumenta a chance de B. Acima de 1 = associação positiva.",
    )
    st.divider()
    st.markdown(
        "**Sugestões para testar**\n"
        "- Suba o suporte e veja as regras sumirem.\n"
        "- Ligue *Remover itens dominantes* e procure pares de receita.\n"
        "- Monte um carrinho e compare a chance com a média."
    )

removidos = DOMINANTES if remover else ()
with st.spinner("Rodando o Apriori..."):
    enc = montar_transacoes(removidos)
    freq, regras = minerar(removidos, suporte_pct / 100)

n_trans = len(enc)
if not regras.empty:
    filtradas = regras[(regras["confidence"] >= confianca_pct / 100) & (regras["lift"] >= lift_min)]
else:
    filtradas = regras
unicas = sem_espelhos(filtradas)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Compras", num(n_trans), help="Compras (cliente + data) com dois ou mais itens.")
c2.metric("Conjuntos frequentes", num(len(freq)))
c3.metric("Regras", num(len(unicas)), help="Sem contar os pares espelhados (A→B e B→A).")
c4.metric("Suporte mín. (compras)", num(round(suporte_pct / 100 * n_trans)),
          help="Número mínimo de compras em que um conjunto precisa aparecer.")

aba_carrinho, aba_regras, aba_mapa, aba_dados, aba_teoria = st.tabs(
    ["🛒 Monte seu carrinho", "🔍 Regras", "🕸️ Mapa de produtos", "📊 Os dados", "📘 Como funciona"]
)

# ---------------- Monte seu carrinho ----------------
with aba_carrinho:
    disponiveis = sorted(enc.columns, key=pt)
    if "carrinho" in st.session_state:
        st.session_state["carrinho"] = [i for i in st.session_state["carrinho"] if i in disponiveis]

    st.subheader("Escolha produtos e veja o que o sistema recomenda")
    botoes = st.columns(len(EXEMPLOS))
    for coluna, (rotulo, itens) in zip(botoes, EXEMPLOS.items()):
        coluna.button(rotulo, on_click=colocar_exemplo, args=(itens, disponiveis), width="stretch")

    carrinho = st.multiselect(
        "Produtos no carrinho", options=disponiveis, format_func=pt, key="carrinho",
        placeholder="Digite ou escolha produtos...",
    )
    ordem = st.radio(
        "Ordenar recomendações por", ["Confiança (chance de levar)", "Lift (quanto o carrinho aumenta a chance)"],
        horizontal=True,
    )

    if not carrinho:
        st.info("Adicione produtos ao carrinho, ou clique num dos exemplos acima.")
    elif filtradas.empty:
        st.warning("Nenhuma regra com os parâmetros atuais. Tente baixar o suporte ou o lift mínimo.")
    else:
        no_carrinho = set(carrinho)
        aplicaveis = filtradas[
            filtradas["antecedents"].apply(lambda a: a <= no_carrinho)
            & filtradas["consequents"].apply(lambda c: not (c & no_carrinho))
        ]
        if aplicaveis.empty:
            st.warning(
                "Nenhuma regra parte desses produtos com os parâmetros atuais. "
                "Tente baixar o suporte mínimo na barra lateral."
            )
        else:
            chave = "lift" if ordem.startswith("Lift") else "confidence"
            melhores = (
                aplicaveis.sort_values([chave, "support"], ascending=False)
                .drop_duplicates("consequents")
                .head(6)
            )
            st.markdown(f"**{len(aplicaveis)} regra(s)** se aplicam ao carrinho. As melhores recomendações:")
            cards = st.columns(3)
            for i, (_, r) in enumerate(melhores.iterrows()):
                with cards[i % 3].container(border=True):
                    st.markdown(f"### {lado(r['consequents'])}")
                    st.markdown(f"porque você levou **{lado(r['antecedents'])}**")
                    # Barras na mesma escala: a diferença de tamanho é o lift
                    com, media = r["confidence"], r["consequent support"]
                    escala = max(com, media)
                    st.markdown(f"Com o seu carrinho: **{num(com * 100, 1)}%**")
                    st.progress(com / escala)
                    st.markdown(f"Média de todas as compras: **{num(media * 100, 1)}%**")
                    st.progress(media / escala)
                    st.caption(
                        f"Lift {num(r['lift'], 2)}× · baseado em {r['compras']} compras · "
                        f"p = {num(r['p'], 4)}"
                    )
            st.caption(
                "**Chance média** é a fração de todas as compras que têm o produto. "
                "O lift é a razão entre as duas chances."
            )

# ---------------- Regras ----------------
with aba_regras:
    if unicas.empty:
        st.warning("Nenhuma regra com os parâmetros atuais.")
    else:
        alfa = 0.05 / len(unicas)
        tabela = pd.DataFrame({
            "Regra": unicas["regra"],
            "Compras": unicas["compras"],
            "Suporte (%)": unicas["support"] * 100,
            "Confiança (%)": unicas["confidence"] * 100,
            "Lift": unicas["lift"],
            "p (Fisher)": unicas["p"],
            "Significativa*": unicas["p"] < alfa,
        })
        st.subheader(f"{num(len(unicas))} regras, da maior para a menor lift")
        st.dataframe(
            tabela, hide_index=True, width="stretch", height=380,
            column_config={
                "Suporte (%)": st.column_config.NumberColumn(format="%.2f"),
                "Confiança (%)": st.column_config.NumberColumn(format="%.1f"),
                "Lift": st.column_config.ProgressColumn(
                    format="%.2f", min_value=1.0, max_value=max(4.0, float(tabela["Lift"].max()))
                ),
                "p (Fisher)": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        n_sig = int(tabela["Significativa*"].sum())
        st.caption(
            f"*Teste exato de Fisher com correção de Bonferroni para {num(len(unicas))} regras "
            f"(p < {alfa:.6f}). Regras significativas: **{n_sig}**. "
            "Pares espelhados (A→B e B→A) aparecem uma vez só."
        )

        fig = px.scatter(
            unicas.assign(sup=unicas["support"] * 100, conf=unicas["confidence"] * 100),
            x="sup", y="conf", color="lift", hover_name="regra", log_x=True,
            hover_data={"sup": ":.2f", "conf": ":.1f", "lift": ":.2f", "compras": True},
            color_continuous_scale="YlOrBr",
            labels={"sup": "Suporte (%)", "conf": "Confiança (%)", "lift": "Lift", "compras": "Compras"},
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color=CORES["escuro"])))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), separators=",.",
                          title="Cada ponto é uma regra (passe o mouse para ver)")
        st.plotly_chart(fig, width="stretch")

# ---------------- Mapa de produtos ----------------
with aba_mapa:
    pares = unicas[(unicas["antecedents"].apply(len) == 1) & (unicas["consequents"].apply(len) == 1)] \
        if not unicas.empty else unicas
    if pares.empty:
        st.warning("Nenhum par de produtos com os parâmetros atuais.")
    else:
        max_lig = min(80, len(pares))
        n_lig = st.slider("Ligações mostradas (maiores lifts)", 1, max_lig, min(25, max_lig)) if max_lig > 1 else 1
        top = pares.head(n_lig)
        g = nx.Graph()
        for _, r in top.iterrows():
            a, b = next(iter(r["antecedents"])), next(iter(r["consequents"]))
            g.add_edge(a, b, lift=r["lift"], regra=r["regra"], conf=r["confidence"], weight=1.0)

        # Grupos desconectados se afastam demais no layout de molas: um nó central
        # invisível, ligado fracamente a cada grupo, mantém todos perto.
        h = g.copy()
        grupos = sorted(nx.connected_components(g), key=lambda c: (-len(c), sorted(c)))
        if len(grupos) > 1:
            for grupo in grupos:
                h.add_edge("__centro__", sorted(grupo)[0], weight=0.08)
        pos = nx.spring_layout(h, seed=7, k=2.8 / len(h) ** 0.5, iterations=400, weight="weight")
        pos.pop("__centro__", None)

        fig = go.Figure()
        lifts = [d["lift"] for *_, d in g.edges(data=True)]
        lmin, lmax = min(lifts), max(lifts)
        for a, b, d in g.edges(data=True):
            peso = 1 + 6 * ((d["lift"] - lmin) / (lmax - lmin) if lmax > lmin else 0.5)
            fig.add_trace(go.Scatter(
                x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
                line=dict(width=peso, color="rgba(157,176,162,0.55)"), hoverinfo="skip", showlegend=False,
            ))
        meio_x = [(pos[a][0] + pos[b][0]) / 2 for a, b in g.edges()]
        meio_y = [(pos[a][1] + pos[b][1]) / 2 for a, b in g.edges()]
        dicas = [f"{pt(a)} ↔ {pt(b)}<br>lift {num(d['lift'], 2)}" for a, b, d in g.edges(data=True)]
        fig.add_trace(go.Scatter(x=meio_x, y=meio_y, mode="markers", marker=dict(size=10, opacity=0),
                                 hovertext=dicas, hoverinfo="text", showlegend=False))

        nos = list(g.nodes())
        freq_no = [enc[i].mean() * 100 for i in nos]
        # Nome do lado oposto aos vizinhos, para não cair em cima das linhas e dos outros nomes
        lado_texto = []
        for i in nos:
            y_viz = sum(pos[v][1] for v in g.neighbors(i)) / g.degree(i)
            lado_texto.append("top center" if pos[i][1] >= y_viz else "bottom center")
        no_carrinho = set(st.session_state.get("carrinho", []))
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in nos], y=[pos[i][1] for i in nos], mode="markers+text",
            text=[pt(i) for i in nos], textposition=lado_texto,
            marker=dict(
                size=[12 + 4 * f for f in freq_no],
                color=[CORES["destaque"] if i in no_carrinho else CORES["neutro"] for i in nos],
                line=dict(width=1.5, color=CORES["escuro"]),
            ),
            hovertext=[f"{pt(i)}<br>em {num(f, 1)}% das compras" for i, f in zip(nos, freq_no)],
            hoverinfo="text", showlegend=False,
        ))
        fig.update_layout(
            height=620, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Cada círculo é um produto (maior = mais frequente). Cada linha é uma regra entre dois produtos "
            "(mais grossa = lift maior). Produtos do seu carrinho aparecem em laranja."
        )

# ---------------- Os dados ----------------
with aba_dados:
    bruto = carregar_dados()
    compras = bruto.groupby(["Member_number", "Date"])["itemDescription"].apply(set)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Registros no CSV", num(len(bruto)))
    d2.metric("Compras (cliente + data)", num(len(compras)))
    d3.metric("Clientes", num(bruto["Member_number"].nunique()))
    d4.metric("Produtos", num(bruto["itemDescription"].nunique()))

    esquerda, direita = st.columns(2)
    contagem = pd.Series([i for s in compras for i in s]).value_counts().head(15)
    top_itens = pd.DataFrame({
        "Produto": [pt(i) for i in contagem.index],
        "% das compras": contagem.values / len(compras) * 100,
        "Removido na 2ª rodada": ["sim" if i in DOMINANTES else "não" for i in contagem.index],
    })
    fig = px.bar(
        top_itens.iloc[::-1], x="% das compras", y="Produto", orientation="h",
        color="Removido na 2ª rodada", color_discrete_map={"sim": CORES["destaque"], "não": CORES["neutro"]},
        title="Os 15 produtos mais frequentes",
    )
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10), separators=",.")
    esquerda.plotly_chart(fig, width="stretch")

    tamanhos = compras.apply(len).value_counts().sort_index()
    fig = px.bar(x=tamanhos.index, y=tamanhos.values, labels={"x": "Produtos distintos na compra", "y": "Compras"},
                 title="Tamanho das compras", color_discrete_sequence=[CORES["escuro"]])
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10), separators=",.")
    direita.plotly_chart(fig, width="stretch")
    st.caption(
        f"Em média, cada compra tem {num(compras.apply(len).mean(), 1)} produtos distintos. "
        "Compras pequenas deixam poucas combinações possíveis, por isso o suporte das regras é baixo."
    )

# ---------------- Como funciona ----------------
with aba_teoria:
    st.subheader("As três métricas de uma regra A → B")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("**Suporte**: em que fração das compras o conjunto aparece.")
        st.latex(r"\text{sup}(A \rightarrow B) = P(A \cap B)")
    with m2:
        st.markdown("**Confiança**: de quem leva A, quantos também levam B.")
        st.latex(r"\text{conf}(A \rightarrow B) = \frac{P(A \cap B)}{P(A)}")
    with m3:
        st.markdown("**Lift**: quantas vezes A aumenta a chance de B.")
        st.latex(r"\text{lift}(A \rightarrow B) = \frac{P(A \cap B)}{P(A)\,P(B)}")

    st.subheader("O algoritmo Apriori")
    st.markdown(
        "1. Agrupa os registros por **cliente + data**, formando as compras.\n"
        "2. Transforma cada compra numa linha de verdadeiro/falso, uma coluna por produto.\n"
        "3. Encontra os conjuntos de produtos com suporte acima do mínimo. Se um conjunto é raro, "
        "qualquer conjunto maior que o contenha também é, e por isso o algoritmo descarta cedo "
        "a maior parte das combinações.\n"
        "4. Gera as regras a partir desses conjuntos e calcula confiança e lift."
    )
    st.subheader("Cuidado ao interpretar")
    st.markdown(
        "Com suporte baixo, uma regra pode se apoiar em poucas compras e surgir por acaso. "
        "Por isso o app mostra o **teste exato de Fisher** com **correção de Bonferroni**: "
        "no trabalho, nenhuma regra se manteve significativa depois da correção. "
        "As regras indicam tendências, não certezas."
    )

st.divider()
st.caption(f"Ailana · André · Pedro · Victor — código e relatório em [{REPO.removeprefix('https://')}]({REPO})")
