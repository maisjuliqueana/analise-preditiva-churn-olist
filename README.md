# ANÁLISE PREDITIVA DE CHURN — E-commerce (Dataset Olist)

Modelo de machine learning que prevê a probabilidade de um cliente **não voltar a comprar** depois da primeira compra, usando o dataset público brasileiro da Olist, com features estilo RFM e avaliação de impacto financeiro do modelo — não só métricas técnicas.

## OBJETIVO

Ir além de "treinar um modelo" — o projeto simula um problema real de negócio: **quais clientes de primeira compra têm maior risco de não retornar, e quanto isso representa em receita?** Isso conecta ciência de dados a uma decisão de marketing/CRM (para quem vale a pena mandar um cupom de segunda compra, por exemplo).

## SOBRE A DEFINIÇÃO DE CHURN USADA AQUI

O Olist é um **marketplace**, não uma assinatura — a maioria dos clientes compra uma única vez. Por isso, "churn" aqui foi definido de forma explícita e documentada (importante deixar isso claro em qualquer entrevista sobre o projeto):

> **Churn = cliente que fez sua primeira compra e NÃO fez uma segunda compra nos 180 dias seguintes.**

As features usadas para prever isso são calculadas **apenas com informações disponíveis no momento da primeira compra** (valor do pedido, forma de pagamento, prazo de entrega, categoria do produto, estado do cliente etc.) — nunca com dados do futuro. Isso evita **vazamento de dados (data leakage)**, um erro comum em projetos de churn.

## FONTE DE DADOS

**Brazilian E-Commerce Public Dataset by Olist**, disponível gratuitamente no Kaggle:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

O dataset não está incluído neste repositório (arquivos grandes, e a licença do Kaggle pede que cada pessoa baixe diretamente). Instruções de download em `data/README.md`.

## ARQUITETURA E PIPELINE

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

## ESTRUTURA DE REPOSITÓRIO

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

## COMO REPRODUZIR

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

## O QUE O MODELO ENTREGA? 

| Métrica | Por que importa (não só acurácia) |
|---|---|
| **AUC-ROC** | Mede a capacidade geral do modelo de separar quem volta de quem não volta |
| **Recall na classe "churn"** | De todo mundo que realmente ia embora, quantos o modelo identificou a tempo |
| **Precision na classe "churn"** | De quem o modelo aponta como risco, quantos realmente eram risco (evita gastar verba de retenção à toa) |
| **Receita em risco (R$)** | Soma do valor médio de segunda compra dos clientes classificados como alto risco |

## RESULTADOS OBTIDOS

### Panorama geral
- **66.692 clientes** elegíveis analisados (com primeira compra antiga o suficiente para observar a janela de 180 dias)
- **Taxa de churn observada: 97,3%** — a esmagadora maioria dos clientes de primeira compra não retorna, o que é consistente com o comportamento típico de um marketplace (compra pontual, não recorrente)

### Performance dos modelos

| Modelo | AUC-ROC | Precision (churn) | Recall (churn) |
|---|---|---|---|
| Regressão Logística | 0,594 | 0,978 | 0,611 |
| **XGBoost** | **0,596** | 0,977 | **0,697** |

O XGBoost superou a Regressão Logística principalmente em **recall** (69,7% vs. 61,1%) — ou seja, identifica uma fatia maior dos clientes que realmente vão dar churn, o que é o que mais importa para uma campanha de retenção (melhor errar prevendo risco a mais do que deixar passar um cliente que ia embora).

O AUC-ROC relativamente baixo (~0,60) indica que as features disponíveis na primeira compra (valor, prazo de entrega, categoria, forma de pagamento) têm **poder preditivo moderado** — um resultado honesto, e esperado dado o desbalanceamento extremo da classe. Isso abre espaço real para evolução (ver seção "Próximos passos").

### Impacto financeiro estimado

| Faixa de risco | Clientes | Ticket médio | Receita total em risco | Recuperável com campanha (15%) |
|---|---|---|---|---|
| Alto risco | 1.039 | R$ 277,85 | R$ 288.688,82 | R$ 43.303,32 |
| Risco médio | 14.327 | R$ 128,87 | **R$ 1.846.383,40** | **R$ 276.957,51** |
| Baixo risco | 1.307 | R$ 129,60 | R$ 168.613,65 | R$ 25.292,05 |

**Insight de negócio:** embora os clientes de "alto risco" tenham o maior ticket médio individual, é a faixa de **"risco médio" que concentra a maior receita total exposta** — simplesmente por ter um volume muito maior de clientes (14.327 vs. 1.039). Isso sugere que uma campanha de retenção bem desenhada pode gerar mais retorno se não focar só nos poucos clientes de risco extremo, mas também alcançar esse grupo intermediário, bem maior.

### Limitação assumida
O AUC modesto reforça um ponto já documentado em `docs/decisoes_arquitetura.md`: com dados exclusivos da primeira compra, o modelo tem um teto de performance. Uma evolução natural seria incorporar dados de comportamento de navegação (se disponíveis) ou testar janelas de churn diferentes (90 ou 365 dias).

## COMO A ISA FOI USADA AQUI

- Ajuste na documentação da metodologia de churn (evitando vazamento de dados)
- Revisão do código de feature engineering e treino dos modelos
- Revisão na estruturação do cálculo de impacto financeiro
- Ajuste do README e da documentação de decisões

## PRÓXIMOS PASSOS

- Testar outras janelas de churn (90 dias, 365 dias) e comparar
- Adicionar SHAP values para explicar as previsões individualmente
- Testar reamostragem (SMOTE) já que a classe "não-churn" costuma ser bem menor
- Publicar o dashboard no Streamlit Cloud (gratuito) para link direto no portfólio

---
Projeto criado como parte de um portfólio de Engenharia de Dados / Ciência de Dados / BI.


by Juliana Araújo
