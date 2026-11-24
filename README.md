# Mineração de Regras de Associação — Groceries Dataset

**O que os clientes compram juntos?** Este projeto aplica o algoritmo **Apriori** a um histórico de compras de supermercado para descobrir quais produtos costumam ir juntos para o carrinho.

Trabalho da disciplina **Inteligência Computacional** — Prof. Canarozzo.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TrabalhoCanarozo/Minera-o_dados/blob/main/Minera%C3%A7%C3%A3o_produtos.ipynb)

📄 **Relatório completo (PDF):** [`relatorio/relatorio.pdf`](relatorio/relatorio.pdf)

🛒 **Simulador interativo:** [cesta-de-compras.streamlit.app](https://cesta-de-compras.streamlit.app), app em Streamlit para explorar as regras ao vivo ([detalhes](#simulador-interativo))

---

## Sumário

- [Dataset](#dataset)
- [Metodologia](#metodologia)
- [Estrutura do notebook](#estrutura-do-notebook)
- [Principais resultados](#principais-resultados)
- [Limitações](#limitações)
- [Relatório em LaTeX](#relatório-em-latex)
- [Simulador interativo](#simulador-interativo)
- [Como executar](#como-executar)
- [Autores](#autores)

---

## Dataset

O arquivo [`Groceries_dataset.csv`](Groceries_dataset.csv) registra um produto comprado por linha.

| Informação | Valor |
|---|---|
| Itens registrados (linhas) | 38.765 |
| Compras (cliente + data) | 14.963 |
| Clientes | 3.898 |
| Produtos diferentes | 167 |
| Itens por compra (média) | 2,6 |
| Período | 2014 e 2015 |

Colunas: `Member_number` (cliente), `Date` (data da compra) e `itemDescription` (produto).

Cada combinação de **cliente + data** foi tratada como uma compra (uma "cesta").

## Metodologia

1. **Leitura** do CSV com `pandas`.
2. **Agrupamento** das linhas por cliente e data, formando as transações.
3. **Codificação one-hot** com `TransactionEncoder`: uma coluna verdadeiro/falso por produto.
4. **Apriori** (`mlxtend`) para encontrar os conjuntos frequentes.
5. **Geração das regras** com `association_rules`, ordenadas por lift e confiança.

### Métricas

Para uma regra **A → B**:

| Métrica | Pergunta que responde | Fórmula |
|---|---|---|
| **Suporte** | Em que fração das compras o conjunto aparece? | P(A ∩ B) |
| **Confiança** | De quem leva A, quantos também levam B? | P(A ∩ B) / P(A) |
| **Lift** | Quantas vezes A aumenta a chance de B? | Confiança / P(B) |

Lift **> 1** indica associação positiva; **= 1**, independência; **< 1**, que um produto afasta o outro.

## Estrutura do notebook

O notebook [`Mineração_produtos.ipynb`](Minera%C3%A7%C3%A3o_produtos.ipynb) é dividido em três partes:

| Parte | Conteúdo |
|---|---|
| **1** | Carregamento dos dados, montagem das transações, Apriori (suporte mínimo de 0,1%) e regras com lift ≥ 1 |
| **2** | Tradução dos nomes dos produtos para português e nova geração das regras |
| **3** | Análise sem os itens dominantes (leite integral, iogurte e salsicha): primeiro filtrando as regras, depois removendo os itens do dataset e rodando o Apriori com suporte mínimo de 0,05% |
| **Gráfico** | Barras com o lift das 10 principais associações |

## Principais resultados

### Primeira rodada — suporte mínimo de 0,1%

O Apriori encontrou **750 conjuntos frequentes**. As regras de maior lift (pares espelhados A → B e B → A contados uma vez):

| Regra | Suporte | Confiança | Lift |
|---|---:|---:|---:|
| iogurte + leite integral → salsicha | 0,15% | 13,2% | **2,18** |
| salsicha + leite integral → iogurte | 0,15% | 16,4% | 1,91 |
| chocolate especial ↔ frutas cítricas | 0,14% | 8,8% / 2,6% | 1,65 |
| iogurte + salsicha → leite integral | 0,15% | 25,6% | 1,62 |
| frutas tropicais ↔ farinha | 0,11% | 1,6% / 11,0% | 1,62 |

A regra mais forte, **iogurte + leite integral → salsicha**, aparece em 22 compras: quem leva iogurte e leite tem **2,18 vezes** mais chance de levar salsicha.

### Itens dominantes

Alguns produtos aparecem em uma fração grande das compras e dominam o topo das regras:

| Produto | % das compras |
|---|---:|
| leite integral | 15,8% |
| outros vegetais | 12,2% |
| pãezinhos | 11,0% |
| refrigerante | 9,7% |
| iogurte | 8,6% |

### Segunda rodada — sem os dominantes, suporte mínimo de 0,05%

Removendo leite integral, iogurte e salsicha do dataset, sobram 12.270 compras com dois ou mais itens, e aparecem pares típicos de receitas:

| Par | Lift | Compras |
|---|---:|---:|
| presunto ↔ conservas | 3,32 | 8 |
| ervas ↔ óleo | 2,76 | 7 |
| macarrão ↔ carne moída | 2,67 | 7 |
| queijo macio ↔ carne moída | 2,56 | 9 |

## Limitações

- **Suporte baixo:** a melhor regra aparece em 22 compras, e as da segunda rodada em 7 a 9. As regras indicam **tendências**, não certezas.
- **Confiança baixa:** na primeira rodada, a confiança máxima é de 26%.
- **Bases diferentes:** remover itens do dataset descarta compras, então os suportes das duas rodadas não são comparáveis.
- **Regras espelhadas:** A → B e B → A têm o mesmo lift e aparecem duplicadas na saída.

- **Significância estatística:** pelo teste exato de Fisher, a maioria das regras do topo tem p < 0,05 isoladamente, mas **nenhuma se mantém significativa** após a correção para comparações múltiplas (Bonferroni ou Benjamini-Hochberg). Detalhes no relatório.

## Relatório em LaTeX

A pasta [`relatorio/`](relatorio) traz um relatório no formato de artigo, com fundamentação teórica, metodologia, resultados, teste de significância e discussão.

| Arquivo | Conteúdo |
|---|---|
| [`relatorio.pdf`](relatorio/relatorio.pdf) | Relatório compilado |
| [`relatorio.tex`](relatorio/relatorio.tex) | Código-fonte LaTeX |
| [`gerar_resultados.py`](relatorio/gerar_resultados.py) | Script que refaz a análise e gera tabelas, figuras e números do relatório |
| `tabelas/`, `figuras/` | Saídas do script, usadas pelo `.tex` |

Para gerar tudo de novo:

```bash
pip install pandas mlxtend matplotlib scipy
python relatorio/gerar_resultados.py
cd relatorio
pdflatex relatorio.tex
pdflatex relatorio.tex
```

## Simulador interativo

O arquivo [`app.py`](app.py) é um app em **Streamlit** que roda o mesmo Apriori do notebook e permite explorar os resultados ao vivo:

| Aba | O que faz |
|---|---|
| 🛒 **Monte seu carrinho** | Escolha produtos e veja as recomendações, comparando a chance de levar cada item com a média de todas as compras |
| 🔍 **Regras** | Tabela com suporte, confiança, lift e teste exato de Fisher (com correção de Bonferroni), mais um gráfico de dispersão |
| 🕸️ **Mapa de produtos** | Grafo dos produtos ligados pelas regras de maior lift |
| 📊 **Os dados** | Produtos mais frequentes e tamanho das compras |
| 📘 **Como funciona** | As métricas e o algoritmo explicados |

Na barra lateral dá para mudar o suporte, a confiança e o lift mínimos e remover os itens dominantes, como na segunda rodada do notebook.

```bash
pip install -r requirements.txt
streamlit run app.py
```

O app abre no navegador em `http://localhost:8501`. A tradução dos 167 produtos fica em [`traducao_produtos.py`](traducao_produtos.py).

## Como executar

**No Google Colab:** clique no botão "Abrir no Colab" acima e envie o arquivo `Groceries_dataset.csv` para a sessão antes de rodar as células.

**Localmente:**

```bash
git clone https://github.com/TrabalhoCanarozo/Minera-o_dados.git
cd Minera-o_dados
pip install pandas mlxtend jupyter
jupyter notebook Mineração_produtos.ipynb
```

O CSV precisa estar na mesma pasta do notebook. As células devem ser executadas em ordem, porque a Parte 3 usa variáveis criadas na Parte 2.

## Autores

- Ailana
- André
- Pedro
- Victor
