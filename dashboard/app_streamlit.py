"""
app_streamlit.py

Dashboard interativo que mostra os clientes classificados por risco de
churn e o impacto financeiro estimado — a "camada de visualização" deste
projeto (no lugar do Power BI, já que este roda 100% local e gratuito,
inclusive em Mac sem licença).

Como rodar:
    streamlit run dashboard/app_streamlit.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import carregar_previsoes, classificar_faixa_risco  # noqa: E402

st.set_page_config(page_title="Painel de Churn — Olist", layout="wide")

st.title("🔮 Painel de Risco de Churn — Clientes Olist")
st.caption(
    "Clientes de primeira compra classificados por probabilidade de não "
    "retornar dentro de 180 dias, com base no modelo XGBoost treinado."
)

try:
    df = carregar_previsoes()
except FileNotFoundError as erro:
    st.error(str(erro))
    st.stop()

df["faixa_risco"] = df["probabilidade_churn"].apply(classificar_faixa_risco)

# ── Filtros na barra lateral ────────────────────────────────────────
st.sidebar.header("Filtros")
faixas_selecionadas = st.sidebar.multiselect(
    "Faixa de risco",
    options=["Alto risco", "Risco médio", "Baixo risco"],
    default=["Alto risco", "Risco médio", "Baixo risco"],
)
estados_disponiveis = sorted(df["customer_state"].dropna().unique())
estados_selecionados = st.sidebar.multiselect(
    "Estado do cliente", options=estados_disponiveis, default=estados_disponiveis
)

df_filtrado = df[
    df["faixa_risco"].isin(faixas_selecionadas) & df["customer_state"].isin(estados_selecionados)
]

# ── KPIs principais ──────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Clientes no filtro", f"{len(df_filtrado):,}")
col2.metric(
    "Taxa de churn real (conjunto de teste)", f"{df_filtrado['churn_real'].mean():.1%}"
)
col3.metric(
    "Probabilidade média de churn", f"{df_filtrado['probabilidade_churn'].mean():.1%}"
)
col4.metric(
    "Receita em risco (alto risco)",
    f"R$ {df_filtrado[df_filtrado['faixa_risco'] == 'Alto risco']['valor_total_itens'].sum():,.0f}",
)

st.divider()

# ── Gráficos ─────────────────────────────────────────────────────────
col_esq, col_dir = st.columns(2)

with col_esq:
    st.subheader("Distribuição de clientes por faixa de risco")
    contagem_faixas = df_filtrado["faixa_risco"].value_counts().reset_index()
    contagem_faixas.columns = ["faixa_risco", "quantidade"]
    fig_faixas = px.bar(
        contagem_faixas,
        x="faixa_risco",
        y="quantidade",
        color="faixa_risco",
        color_discrete_map={
            "Alto risco": "#d62728",
            "Risco médio": "#ff7f0e",
            "Baixo risco": "#2ca02c",
        },
    )
    st.plotly_chart(fig_faixas, use_container_width=True)

with col_dir:
    st.subheader("Receita em risco por estado (top 10)")
    receita_por_estado = (
        df_filtrado[df_filtrado["faixa_risco"] == "Alto risco"]
        .groupby("customer_state")["valor_total_itens"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig_estado = px.bar(
        receita_por_estado, x="customer_state", y="valor_total_itens", color_discrete_sequence=["#d62728"]
    )
    st.plotly_chart(fig_estado, use_container_width=True)

st.divider()

# ── Tabela de clientes de alto risco ─────────────────────────────────
st.subheader("Clientes de alto risco (priorizados para campanha de retenção)")
tabela_alto_risco = (
    df_filtrado[df_filtrado["faixa_risco"] == "Alto risco"]
    .sort_values("probabilidade_churn", ascending=False)
    [
        [
            "customer_unique_id",
            "customer_state",
            "valor_total_itens",
            "categoria_principal",
            "prazo_entrega_dias",
            "probabilidade_churn",
        ]
    ]
)
st.dataframe(tabela_alto_risco, use_container_width=True, hide_index=True)
