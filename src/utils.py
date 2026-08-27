"""
utils.py

Funções auxiliares compartilhadas entre os scripts do projeto.
"""

from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def carregar_previsoes() -> pd.DataFrame:
    """Carrega o conjunto de teste com as probabilidades de churn previstas."""
    caminho = OUTPUT_DIR / "previsoes_teste.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}\n"
            f"Rode primeiro: python src/data_prep.py && python src/train_model.py"
        )
    return pd.read_csv(caminho)


def classificar_faixa_risco(probabilidade: float) -> str:
    """Converte a probabilidade numérica em uma faixa de risco legível."""
    if probabilidade >= 0.7:
        return "Alto risco"
    elif probabilidade >= 0.4:
        return "Risco médio"
    else:
        return "Baixo risco"
