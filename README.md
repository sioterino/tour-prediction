# TOUR PREDICTION

Projeto de Machine Learning para prever o público de shows de K-pop com base em características do artista, da turnê, do local e do mercado.

## Objetivo

O objetivo é construir e avaliar modelos de regressão capazes de estimar a variável `attendance`, representando a quantidade de ingressos vendidos em shows de K-pop.

## Modelos utilizados (Regressão)

- Random Forest
- Árvore de Decisão
- Rede Neural (MLP)

## Métricas

- MAE
- RMSE
- R² Score

## Estrutura

- `data/`: dados brutos e processados
- `notebooks/`: análise exploratória e experimentos
- `src/`: scripts reutilizáveis
- `models/`: modelos treinados
- `reports/`: relatório final e figuras

## Como executar

```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

## Limitações

O modelo depende da qualidade dos dados disponíveis e pode apresentar viés por artista, país, venue ou nível de popularidade.
