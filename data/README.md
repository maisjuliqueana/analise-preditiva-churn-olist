# Como baixar o dataset

Este projeto usa o **Brazilian E-Commerce Public Dataset by Olist**, disponível gratuitamente no Kaggle.

## Passo a passo

1. Crie uma conta gratuita em [kaggle.com](https://www.kaggle.com) (se ainda não tiver)
2. Acesse: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
3. Clique em **Download** (arquivo `.zip`, ~50 MB)
4. Extraia o `.zip` e copie **todos os arquivos `.csv`** para dentro desta pasta (`data/raw/`)

## Arquivos esperados

Depois de extrair, a pasta `data/raw/` deve conter:

```
data/raw/
├── olist_customers_dataset.csv
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
├── olist_geolocation_dataset.csv
└── product_category_name_translation.csv
```

## Alternativa: baixar via linha de comando (Kaggle API)

Se preferir automatizar:

```bash
pip install kaggle
# Configure sua API key seguindo: https://www.kaggle.com/docs/api
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip
```

> Os arquivos CSV **não são versionados no Git** (ver `.gitignore`) — cada pessoa que for rodar o projeto precisa baixá-los localmente.
