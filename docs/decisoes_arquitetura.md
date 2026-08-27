# Decisões de Arquitetura

## 1. Por que essa definição específica de churn (180 dias sem 2ª compra)?

O Olist é um marketplace, não uma assinatura — não existe um evento óbvio de "cancelamento". Por isso, qualquer definição de churn em dados desse tipo é uma **escolha de negócio**, não um fato dos dados. Decidi documentar essa escolha explicitamente (em vez de esconder como se fosse óbvia), porque isso é exatamente o tipo de pergunta que um analista sênior faz numa entrevista: *"o que você considerou churn, e por quê?"*

180 dias foi escolhido como uma janela que:
- É longa o suficiente pra capturar boa parte do comportamento de recompra natural de e-commerce (a maioria das recompras que vão acontecer, acontece nesse intervalo)
- Ainda deixa uma amostra razoável de clientes "elegíveis" no dataset (que tiveram tempo suficiente para observar o resultado)

## 2. Por que só entram no dataset clientes com 1ª compra "antiga o suficiente"?

Esse é o ponto mais importante do projeto do ponto de vista metodológico: **evitar vazamento de dados (data leakage)**. Se eu incluísse um cliente que comprou 10 dias antes do fim do dataset, eu não teria como saber se ele "não voltou" (churn) ou apenas "ainda não teve tempo de voltar". Rotular esse cliente como churn seria simplesmente errado — não é uma medição, é um artefato do corte dos dados.

Por isso, o pipeline filtra: só entram clientes cuja primeira compra aconteceu com pelo menos 180 dias de antecedência da última data disponível no dataset. Isso reduz o tamanho da amostra, mas garante que o rótulo é confiável.

## 3. Por que as features usam só informação "do momento da primeira compra"?

Mesmo princípio do item 2, mas aplicado às features: se eu usasse, por exemplo, "nota média de todas as avaliações do cliente" (que inclui avaliações de compras futuras), o modelo estaria "trapaceando" — usando informação que não existiria no momento real de decisão (logo depois da primeira compra, quando a empresa precisaria decidir se manda uma campanha de retenção ou não).

## 4. Por que dois modelos (Regressão Logística + XGBoost)?

- **Regressão Logística**: serve de baseline interpretável — os coeficientes mostram diretamente a direção do efeito de cada variável (ex: "prazo de entrega maior aumenta a chance de churn").
- **XGBoost**: geralmente captura relações não-lineares e interações entre variáveis melhor que um modelo linear, então costuma performar melhor em métricas de ranking (AUC).

Comparar os dois e mostrar a métrica de cada um é mais honesto do que só apresentar "o modelo" — mostra que houve um processo de comparação, não só uma escolha arbitrária.

## 5. Por que não usar acurácia como métrica principal?

A taxa de churn real neste tipo de dataset costuma ser bem alta (a maioria dos clientes de e-commerce brasileiro não recompra em 180 dias), o que torna a classe desbalanceada. Um modelo "preguiçoso" que sempre prevê "vai dar churn" pode ter acurácia alta só por isso, sem ser útil. Por isso o projeto foca em:
- **AUC-ROC**: mede a capacidade de ranquear corretamente quem tem mais risco, independente do desbalanceamento
- **Precision/Recall da classe churn**: mostra o trade-off real que o time de CRM enfrentaria (quantos clientes de risco real o modelo pega vs. quanto "desperdício" de campanha em clientes que não iam dar churn mesmo)

## 6. Por que Streamlit em vez de Power BI neste projeto?

Diferente dos outros dois projetos do portfólio (que usam Power BI), este foi pensado pra rodar 100% localmente em qualquer sistema operacional, sem depender de licença — o que também demonstra versatilidade: saber entregar uma camada de visualização adequada à ferramenta disponível, não só repetir a mesma stack em todo projeto.

## 7. Limitações conhecidas deste projeto

- A janela de 180 dias é um parâmetro escolhido, não uma verdade absoluta — o ideal seria testar múltiplas janelas (90, 180, 365 dias) e comparar, como listado em "próximos passos" no README.
- A taxa de sucesso de campanha de retenção (15%) usada no cálculo de impacto financeiro é uma **suposição ilustrativa**, não um dado medido — em um cenário real, isso viria de testes A/B históricos da empresa.
- O dataset da Olist tem uma limitação temporal (concentra-se majoritariamente em 2017-2018), então o modelo reflete o comportamento de compra daquele período específico.
