"""
train_model.py

Treina dois modelos para prever churn:
    1. Regressão Logística (baseline, interpretável)
    2. XGBoost (modelo mais forte, geralmente melhor performance)

Usa split TEMPORAL não é possível aqui porque o dataset já foi filtrado
por elegibilidade de janela em data_prep.py — por isso usamos split
aleatório estratificado pela variável churn, prática padrão quando as
features não têm componente de série temporal direta no treino.

Avalia com métricas focadas no problema de negócio (não só acurácia,
que engana em datasets desbalanceados).

Saída:
    outputs/modelo_baseline.joblib
    outputs/modelo_xgboost.joblib
    outputs/metricas_modelos.csv
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

FEATURES_NUMERICAS = [
    "valor_total_itens",
    "valor_total_frete",
    "quantidade_itens",
    "parcelas",
    "prazo_entrega_dias",
    "dias_adiantado_atrasado",
]

FEATURES_CATEGORICAS = [
    "customer_state",
    "categoria_principal",
    "forma_pagamento",
]

ALVO = "churn"


def carregar_dataset() -> pd.DataFrame:
    caminho = OUTPUT_DIR / "dataset_features.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {caminho}. Rode primeiro: python src/data_prep.py"
        )
    return pd.read_csv(caminho)


def montar_pipeline_preprocessamento() -> ColumnTransformer:
    """
    Preenche valores faltantes, padroniza numéricas e faz one-hot
    encoding nas categóricas.

    Valores faltantes são esperados neste dataset: por exemplo,
    'dias_adiantado_atrasado' fica vazio quando o pedido não tem data de
    entrega registrada. Preenchemos numéricas com a mediana (robusta a
    outliers) e categóricas com uma categoria explícita "desconhecido".
    """
    pipeline_numerico = Pipeline(
        steps=[
            ("imputacao", SimpleImputer(strategy="median")),
            ("padronizacao", StandardScaler()),
        ]
    )

    pipeline_categorico = Pipeline(
        steps=[
            ("imputacao", SimpleImputer(strategy="constant", fill_value="desconhecido")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", pipeline_numerico, FEATURES_NUMERICAS),
            ("cat", pipeline_categorico, FEATURES_CATEGORICAS),
        ]
    )


def treinar_baseline(X_train, y_train, preprocessador) -> Pipeline:
    logger.info("Treinando modelo baseline (Regressão Logística)...")
    pipeline = Pipeline(
        steps=[
            ("preprocessamento", preprocessador),
            ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def treinar_xgboost(X_train, y_train, preprocessador) -> Pipeline:
    logger.info("Treinando XGBoost...")
    # peso pra classe minoritária, calculado a partir do desbalanceamento do treino
    proporcao_negativa_positiva = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    pipeline = Pipeline(
        steps=[
            ("preprocessamento", preprocessador),
            (
                "modelo",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    scale_pos_weight=proporcao_negativa_positiva,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def avaliar_modelo(pipeline: Pipeline, X_test, y_test, nome_modelo: str) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    relatorio = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"\n── {nome_modelo} ──")
    logger.info(f"AUC-ROC: {auc:.3f}")
    logger.info(
        f"Precision (churn): {relatorio['1']['precision']:.3f} | "
        f"Recall (churn): {relatorio['1']['recall']:.3f}"
    )

    return {
        "modelo": nome_modelo,
        "auc_roc": auc,
        "precision_churn": relatorio["1"]["precision"],
        "recall_churn": relatorio["1"]["recall"],
        "f1_churn": relatorio["1"]["f1-score"],
    }


def main():
    df = carregar_dataset()

    X = df[FEATURES_NUMERICAS + FEATURES_CATEGORICAS]
    y = df[ALVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    logger.info(f"Treino: {len(X_train)} clientes | Teste: {len(X_test)} clientes")

    preprocessador = montar_pipeline_preprocessamento()

    modelo_baseline = treinar_baseline(X_train, y_train, preprocessador)
    modelo_xgb = treinar_xgboost(X_train, y_train, preprocessador)

    metricas = [
        avaliar_modelo(modelo_baseline, X_test, y_test, "Regressão Logística"),
        avaliar_modelo(modelo_xgb, X_test, y_test, "XGBoost"),
    ]

    pd.DataFrame(metricas).to_csv(OUTPUT_DIR / "metricas_modelos.csv", index=False)
    joblib.dump(modelo_baseline, OUTPUT_DIR / "modelo_baseline.joblib")
    joblib.dump(modelo_xgb, OUTPUT_DIR / "modelo_xgboost.joblib")

    # salva também o conjunto de teste com as probabilidades do melhor modelo,
    # para o dashboard e o script de impacto financeiro usarem depois
    df_teste_resultado = X_test.copy()
    df_teste_resultado["customer_unique_id"] = df.loc[X_test.index, "customer_unique_id"]
    df_teste_resultado["valor_total_itens"] = df.loc[X_test.index, "valor_total_itens"]
    df_teste_resultado["churn_real"] = y_test.values
    df_teste_resultado["probabilidade_churn"] = modelo_xgb.predict_proba(X_test)[:, 1]
    df_teste_resultado.to_csv(OUTPUT_DIR / "previsoes_teste.csv", index=False)

    logger.info("Modelos, métricas e previsões salvos em outputs/")


if __name__ == "__main__":
    main()
