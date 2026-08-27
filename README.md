# 🔮 Análise Preditiva de Churn — E-commerce (Dataset Olist)

Modelo de machine learning que prevê a probabilidade de um cliente **não voltar a comprar** depois da primeira compra, usando o dataset público brasileiro da Olist, com features estilo RFM e avaliação de impacto financeiro do modelo — não só métricas técnicas.

## 🎯 Objetivo

Ir além de "treinar um modelo" — o projeto simula um problema real de negócio: **quais clientes de primeira compra têm maior risco de não retornar, e quanto isso representa em receita?** Isso conecta ciência de dados a uma decisão de marketing/CRM (para quem vale a pena mandar um cupom de segunda compra, por exemplo).

## ⚠️ Sobre a definição de "churn" usada aqui

O Olist é um **marketplace**, não uma assinatura — a maioria dos clientes compra uma única vez. Por isso, "churn" aqui foi definido de forma explícita e documentada (importante deixar isso claro em qualquer entrevista sobre o projeto):

> **Churn = cliente que fez sua primeira compra e NÃO fez uma segunda compra nos 180 dias seguintes.**

As features usadas para prever isso são calculadas **apenas com informações disponíveis no momento da primeira compra** (valor do pedido, forma de pagamento, prazo de entrega, categoria do produto, estado do cliente etc.) — nunca com dados do futuro. Isso evita **vazamento de dados (data leakage)**, um erro comum em projetos de churn.

## 🗂️ Fonte de dados

**Brazilian E-Commerce Public Dataset by Olist**, disponível gratuitamente no Kaggle:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

O dataset não está incluído neste repositório (arquivos grandes, e a licença do Kaggle pede que cada pessoa baixe diretamente). Instruções de download em `data/README.md`.

## 🏗️ Arquitetura / Pipeline

```
CSVs brutos da Olist (data/raw/)
        │
        ▼
┌─────────────────────────┐
│  src/data_prep.py         │  → junta as tabelas, define o cliente "primeira
│                            │    compra", calcula o rótulo de churn (180 dias)
│                            │    e as features estilo RFM
└─────────────┬─────────────┘
              ▼
┌─────────────────────────┐
│  src/train_model.py       │  → split temporal (não aleatório!), treina
│                            │    Regressão Logística (baseline) e XGBoost,
│                            │    avalia com métricas de negócio
└─────────────┬─────────────┘
              ▼
┌─────────────────────────┐
│  src/evaluate_business_   │  → traduz o modelo em R$: quanto de receita
│  impact.py                 │    está em risco, e quanto uma campanha de
│                            │    retenção poderia recuperar
└─────────────┬─────────────┘
              ▼
┌─────────────────────────┐
│  dashboard/app_streamlit  │  → lista os clientes de maior risco e o
│  .py                       │    impacto financeiro estimado
└─────────────────────────┘
```

## 📁 Estrutura do repositório

```
analise-preditiva-churn-olist/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md                    → como baixar o dataset do Kaggle
│   └── raw/                         → (vazio no repo — você coloca os CSVs aqui)
├── src/
│   ├── data_prep.py                 → junção das tabelas + feature engineering
│   ├── train_model.py               → treino e avaliação dos modelos
│   ├── evaluate_business_impact.py  → tradução do modelo em impacto de receita
│   └── utils.py                     → funções auxiliares compartilhadas
├── dashboard/
│   └── app_streamlit.py             → dashboard interativo dos clientes em risco
├── docs/
│   └── decisoes_arquitetura.md      → por que essa definição de churn, por que
│                                        split temporal, por que essas métricas
└── outputs/                         → modelos e relatórios gerados (gitignored)
```

## 🚀 Como reproduzir

### 1. Baixar os dados
Siga as instruções em `data/README.md` e coloque os CSVs em `data/raw/`.

### 2. Preparar o ambiente

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Rodar o pipeline

```bash
python src/data_prep.py           # gera outputs/dataset_features.csv
python src/train_model.py         # treina os modelos, salva em outputs/
python src/evaluate_business_impact.py   # gera o relatório de impacto financeiro
```

### 4. Rodar o dashboard

```bash
streamlit run dashboard/app_streamlit.py
```

## 📈 O que o modelo entrega

| Métrica | Por que importa (não só acurácia) |
|---|---|
| **AUC-ROC** | Mede a capacidade geral do modelo de separar quem volta de quem não volta |
| **Recall na classe "churn"** | De todo mundo que realmente ia embora, quantos o modelo identificou a tempo |
| **Precision na classe "churn"** | De quem o modelo aponta como risco, quantos realmente eram risco (evita gastar verba de retenção à toa) |
| **Receita em risco (R$)** | Soma do valor médio de segunda compra dos clientes classificados como alto risco |

## 🧠 Como a IA foi usada neste projeto

- Definição e documentação da metodologia de churn (evitando vazamento de dados)
- Geração e revisão do código de feature engineering e treino dos modelos
- Estruturação do cálculo de impacto financeiro
- Redação deste README e da documentação de decisões

## 🔜 Próximos passos (evolução do projeto)

- Testar outras janelas de churn (90 dias, 365 dias) e comparar
- Adicionar SHAP values para explicar as previsões individualmente
- Testar reamostragem (SMOTE) já que a classe "não-churn" costuma ser bem menor
- Publicar o dashboard no Streamlit Cloud (gratuito) para link direto no portfólio

---
📌 Projeto criado como parte de um portfólio de Engenharia de Dados / Ciência de Dados / BI.
