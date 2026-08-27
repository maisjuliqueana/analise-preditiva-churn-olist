"""
evaluate_business_impact.py

Traduz a saída do modelo em uma linguagem que faz sentido pra negócio:
quanto de receita está "em risco" nos clientes classificados como alto
risco de churn, e quanto uma campanha de retenção poderia recuperar.

Premissas assumidas (documentadas, ajustáveis):
    - Valor de uma "segunda compra" recuperada = ticket médio da 1ª compra
      dos próprios clientes de alto risco (proxy conservador).
    - Taxa de sucesso de uma campanha de retenção (ex: cupom, e-mail)
      sobre os clientes de alto risco = 15% (parâmetro ajustável).

Saída: outputs/relatorio_impacto_financeiro.csv (impresso também no terminal)
"""

import logging
from pathlib import Path

import pandas as pd

from utils import carregar_previsoes, classificar_faixa_risco

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# ── Premissas de negócio (ajuste conforme o cenário real) ──────────────
TAXA_SUCESSO_CAMPANHA_RETENCAO = 0.15  # 15% dos clientes de alto risco respondem à campanha
# ─────────────────────────────────────────────────────────────────────


def calcular_impacto(df_previsoes: pd.DataFrame) -> pd.DataFrame:
    df = df_previsoes.copy()
    df["faixa_risco"] = df["probabilidade_churn"].apply(classificar_faixa_risco)

    resumo = (
        df.groupby("faixa_risco")
        .agg(
            quantidade_clientes=("customer_unique_id", "count"),
            ticket_medio=("valor_total_itens", "mean"),
            receita_total_em_risco=("valor_total_itens", "sum"),
        )
        .reset_index()
    )

    resumo["receita_recuperavel_com_campanha"] = (
        resumo["receita_total_em_risco"] * TAXA_SUCESSO_CAMPANHA_RETENCAO
    )

    return resumo


def main():
    df_previsoes = carregar_previsoes()

    resumo = calcular_impacto(df_previsoes)

    logger.info("\n📊 Resumo de impacto financeiro por faixa de risco:\n")
    logger.info("\n" + resumo.to_string(index=False))

    alto_risco = resumo[resumo["faixa_risco"] == "Alto risco"]
    if not alto_risco.empty:
        receita_recuperavel = alto_risco["receita_recuperavel_com_campanha"].iloc[0]
        quantidade = alto_risco["quantidade_clientes"].iloc[0]
        logger.info(
            f"\n💡 Interpretação: assumindo uma taxa de sucesso de "
            f"{TAXA_SUCESSO_CAMPANHA_RETENCAO:.0%} numa campanha de retenção "
            f"direcionada aos {quantidade} clientes de alto risco, o modelo "
            f"aponta um potencial de recuperação de aproximadamente "
            f"R$ {receita_recuperavel:,.2f} em receita de segunda compra."
        )

    caminho_saida = OUTPUT_DIR / "relatorio_impacto_financeiro.csv"
    resumo.to_csv(caminho_saida, index=False)
    logger.info(f"\nRelatório salvo em: {caminho_saida}")


if __name__ == "__main__":
    main()
