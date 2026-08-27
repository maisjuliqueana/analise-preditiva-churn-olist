"""
data_prep.py

Junta as tabelas do dataset Olist, define o rótulo de churn e calcula as
features (estilo RFM) usando APENAS informações disponíveis no momento da
primeira compra de cada cliente — evitando vazamento de dados (data leakage).

Definição de churn usada:
    Um cliente é "churn" (1) se, depois da primeira compra, não fez uma
    segunda compra dentro dos 180 dias seguintes. Caso contrário, é "não
    churn" (0).

Para que o rótulo seja calculável com confiança, só entram no dataset
final clientes cuja primeira compra aconteceu com pelo menos 180 dias de
antecedência da última data disponível no dataset (senão não dá pra saber
se ainda vão comprar de novo dentro da janela).

Saída: outputs/dataset_features.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

JANELA_CHURN_DIAS = 180


def carregar_tabelas() -> dict:
    """Carrega os CSVs brutos da Olist em um dicionário de DataFrames."""
    logger.info("Carregando tabelas brutas da Olist...")

    arquivos = {
        "orders": "olist_orders_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
        "categoria_traducao": "product_category_name_translation.csv",
    }

    tabelas = {}
    for nome, arquivo in arquivos.items():
        caminho = DATA_DIR / arquivo
        if not caminho.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {caminho}\n"
                f"Confira as instruções de download em data/README.md"
            )
        tabelas[nome] = pd.read_csv(caminho)
        logger.info(f"  {nome}: {len(tabelas[nome])} linhas")

    return tabelas


def identificar_primeira_compra(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica a data da primeira compra de cada cliente único
    (a Olist usa customer_unique_id para identificar a mesma pessoa
    em pedidos diferentes — customer_id muda a cada pedido).
    """
    orders = orders.copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

    pedidos_com_cliente = orders.merge(
        customers[["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id",
        how="left",
    )

    primeira_compra = (
        pedidos_com_cliente.sort_values("order_purchase_timestamp")
        .groupby("customer_unique_id")
        .first()
        .reset_index()
    )

    return primeira_compra, pedidos_com_cliente


def calcular_rotulo_churn(primeira_compra: pd.DataFrame, pedidos_com_cliente: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada cliente, verifica se ele fez uma SEGUNDA compra dentro da
    janela de JANELA_CHURN_DIAS após a primeira. Só mantém clientes cuja
    primeira compra teve tempo suficiente para observar essa janela.

    Implementação otimizada: em vez de varrer a base inteira de pedidos
    para cada cliente (O(n²), inviável para dezenas de milhares de
    clientes), agrupamos as datas de compra por cliente uma única vez e
    comparamos apenas dentro do próprio grupo (rápido mesmo com muitos
    clientes).
    """
    data_maxima_dataset = pedidos_com_cliente["order_purchase_timestamp"].max()
    limite_elegivel = data_maxima_dataset - pd.Timedelta(days=JANELA_CHURN_DIAS)

    logger.info(f"Data máxima no dataset: {data_maxima_dataset.date()}")
    logger.info(f"Só entram clientes com 1ª compra até: {limite_elegivel.date()}")

    elegiveis = primeira_compra[primeira_compra["order_purchase_timestamp"] <= limite_elegivel].copy()

    # todas as datas de compra de cada cliente, agrupadas uma única vez
    logger.info("Agrupando datas de compra por cliente...")
    datas_por_cliente = (
        pedidos_com_cliente.groupby("customer_unique_id")["order_purchase_timestamp"]
        .apply(lambda s: s.sort_values().to_numpy())
        .to_dict()
    )

    def teve_segunda_compra(row) -> bool:
        datas_cliente = datas_por_cliente.get(row["customer_unique_id"])
        if datas_cliente is None or len(datas_cliente) < 2:
            return False

        data_primeira = np.datetime64(row["order_purchase_timestamp"])
        limite = data_primeira + np.timedelta64(JANELA_CHURN_DIAS, "D")

        # existe alguma outra data de compra estritamente depois da primeira
        # e dentro da janela de JANELA_CHURN_DIAS?
        return bool(np.any((datas_cliente > data_primeira) & (datas_cliente <= limite)))

    logger.info(f"Calculando rótulo de churn para {len(elegiveis)} clientes elegíveis...")
    elegiveis["fez_segunda_compra"] = elegiveis.apply(teve_segunda_compra, axis=1)
    elegiveis["churn"] = (~elegiveis["fez_segunda_compra"]).astype(int)

    taxa_churn = elegiveis["churn"].mean()
    logger.info(f"Taxa de churn observada: {taxa_churn:.1%}")

    return elegiveis


def calcular_features(
    elegiveis: pd.DataFrame,
    order_items: pd.DataFrame,
    payments: pd.DataFrame,
    products: pd.DataFrame,
    categoria_traducao: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula as features da primeira compra de cada cliente — tudo
    disponível no momento em que a compra foi feita, sem nenhuma
    informação futura.
    """
    logger.info("Calculando features da primeira compra...")

    # valor e quantidade de itens do primeiro pedido
    itens_pedido = (
        order_items.groupby("order_id")
        .agg(
            valor_total_itens=("price", "sum"),
            valor_total_frete=("freight_value", "sum"),
            quantidade_itens=("order_item_id", "count"),
        )
        .reset_index()
    )

    # categoria do produto mais comprado no primeiro pedido (em português traduzido)
    categoria_produto = (
        order_items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
        .merge(categoria_traducao, on="product_category_name", how="left")
        .groupby("order_id")["product_category_name_english"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "desconhecida")
        .reset_index()
        .rename(columns={"product_category_name_english": "categoria_principal"})
    )

    # forma de pagamento predominante e número de parcelas
    pagamento = (
        payments.sort_values("payment_installments", ascending=False)
        .groupby("order_id")
        .first()[["payment_type", "payment_installments"]]
        .reset_index()
        .rename(columns={"payment_type": "forma_pagamento", "payment_installments": "parcelas"})
    )

    features = (
        elegiveis[
            [
                "customer_unique_id",
                "order_id",
                "customer_state",
                "order_purchase_timestamp",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "churn",
            ]
        ]
        .merge(itens_pedido, on="order_id", how="left")
        .merge(categoria_produto, on="order_id", how="left")
        .merge(pagamento, on="order_id", how="left")
    )

    # prazo de entrega real, em dias (feature forte de satisfação)
    features["order_delivered_customer_date"] = pd.to_datetime(features["order_delivered_customer_date"])
    features["prazo_entrega_dias"] = (
        features["order_delivered_customer_date"] - features["order_purchase_timestamp"]
    ).dt.days

    # diferença entre prazo estimado e prazo real (negativo = atrasou)
    features["order_estimated_delivery_date"] = pd.to_datetime(features["order_estimated_delivery_date"])
    features["dias_adiantado_atrasado"] = (
        features["order_estimated_delivery_date"] - features["order_delivered_customer_date"]
    ).dt.days

    colunas_finais = [
        "customer_unique_id",
        "customer_state",
        "valor_total_itens",
        "valor_total_frete",
        "quantidade_itens",
        "categoria_principal",
        "forma_pagamento",
        "parcelas",
        "prazo_entrega_dias",
        "dias_adiantado_atrasado",
        "churn",
    ]

    dataset_final = features[colunas_finais].dropna(subset=["prazo_entrega_dias"])
    logger.info(f"Dataset final: {len(dataset_final)} clientes, {dataset_final.shape[1] - 1} features")

    return dataset_final


def main():
    tabelas = carregar_tabelas()

    primeira_compra, pedidos_com_cliente = identificar_primeira_compra(
        tabelas["orders"], tabelas["customers"]
    )

    elegiveis = calcular_rotulo_churn(primeira_compra, pedidos_com_cliente)

    dataset_final = calcular_features(
        elegiveis,
        tabelas["order_items"],
        tabelas["payments"],
        tabelas["products"],
        tabelas["categoria_traducao"],
    )

    caminho_saida = OUTPUT_DIR / "dataset_features.csv"
    dataset_final.to_csv(caminho_saida, index=False)
    logger.info(f"Dataset de features salvo em: {caminho_saida}")


if __name__ == "__main__":
    main()
