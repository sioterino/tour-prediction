# TOUR PREDICTION (STAGELINE)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-blue)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-3.0.3-blue)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.6-blue)](https://numpy.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.3-lightgrey)](https://fastapi.tiangolo.com/)
[![Joblib](https://img.shields.io/badge/joblib-1.5.3-orange)](https://joblib.readthedocs.io/)
[![Status](https://img.shields.io/badge/status-Acad%C3%AAmico-orange)](#)
[![Licença](https://img.shields.io/badge/licença-MIT-green)](LICENSE)

## 1. Descrição

Projeto universitário de Machine Learning para previsão de público em shows de K-pop. O modelo estima a variável `attendance` a partir de atributos de artista, tour, local e mercado.

---

## 2. Objetivo

- Construir e avaliar um pipeline de regressão para previsão de público de shows.
- Comparar três modelos de regressão em seis versões processadas do dataset.
- Gerar resultados reproduzíveis com métricas de desempenho e exportar o melhor modelo para inferência.

---

## 3. Contexto Acadêmico

O projeto segue a estrutura de trabalho de um curso de IA/ML universitário.

---

## 4. Problema Abordado

Previsão de `attendance` (vendas de ingressos / público estimado) para eventos de K-pop usando variáveis de artista, local e show.


---

## 5. Tipo de Tarefa

- Aprendizado supervisionado;
- Regressão.

---

## 6. Dataset Utilizado

- Arquivo principal: `data/raw/dataset.csv`
- 1169 entradas e 29 colunas no dataset bruto
- A base contém atributos como:
  - `artist_name`, `gender`, `generation`, `members`, `company`, `years_since_debut`
  - `tour_name`, `tour_type`, `stage_setup`, `show_nights`
  - `continent`, `country`, `city`, `venue_name`, `venue_type`, `venue_capacity`
  - `attendance`, `fill_rate`, `box_score`, `avg_ticket_price`, `reporting_status`

> [!NOTE]
> Em [/docs/RAW.md](/docs/RAW.md) há uma explicação clara sobre cada uma das 29 colunas.

O repositório também inclui seis versões processadas dos dados em `data/processed/`:
- `attendance_v1_baseline.csv`
- `attendance_v2_artist.csv`
- `attendance_v3_artist_geo.csv`
- `attendance_v4_artist_geo_time.csv`
- `attendance_v5_full.csv`
- `attendance_v6_no_artist.csv`

> [!NOTE]
> Em [/docs/PREPROCESSING_PLAN.md](/docs/PREPROCESSING_PLAN.md) há uma explicação clara sobre a seleção de features para treino.

---

## 7. Tecnologias Utilizadas

- **Python:** linguagem principal do projeto;
- **pandas:** manipulação e análise de dados;
- **NumPy:** computação numérica e arrays;
- **scikit-learn:** treinamento e avaliação de modelos;
- **FastAPI:** criação da API de predição;
- **Uvicorn:** servidor para execução da API;
- **joblib:** salvamento e carregamento de modelos;
- **matplotlib:** criação de gráficos;
- **seaborn:** visualização estatística de dados.

---

## 8. Requisitos

- Python 3.11 ou superior;
- `pip`;
- Dependências listadas em `requirements.txt`.

---

## 9. Instalação

### 9.1. Linux / macOS

```bash
# clone o repositório
git clone https://github.com/sioterino/tour-prediction
cd tour-prediction

# ative o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# instale as dependências
pip install -r requirements.txt
```

### 9.2. Windows (PowerShell)

```powershell
# clone o repositório
git clone https://github.com/sioterino/tour-prediction
cd tour-prediction

# ative o ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# instale as dependências
python -m pip install -r requirements.txt
```

---

## 10. Estrutura de Pastas

```bash
Tour-Prediction/
├── LICENSE                             # licença mit
├── PROJETO.md                          # descrição acadêmica dos requisitos gerais do projeto
├── README.md                           # você está aqui!
├── requirements.txt                    # dependências do projeto
├── VENV.md                             # passo a passo de como usar o ambientes virtuais
├── data/
│   ├── processed/                      # versões processadas do dataset usadas em experimentos
│   ├── raw/
│   │   └── dataset.csv                 # dataset bruto original
│   ├── results/
│   │   └── metrics.csv                 # resultados dos experimentos de regressão
│   ├── static/                         # arquivos json usados pelo front end
│   └── predictions/                    # predições de attendance geradas
├── docs/
│   ├── PREPROCESSING_PLAN.md           # plano para pré processamento e seleção de features para o dataset
│   └── RAW.md                          # explicação geral sobre o dataset cru e as 29 features nele presentes
├── frontend/                           # interface de demonstração simples
├── models/                             # modelos serializados exportados para inferência
├── notebooks/                          # notebooks para análise de dados, experimentos e resultados
└── src/
├── ├── api/                            # API FastAPI para inferência de attendance
├── ├── analysis/                       # scripts de análise de dataset e gráficos
├── ├── etl/                            # scripts de extração de dados json para o frontend
├── ├── modeling/                       # pipeline de pré-processamento, modelos e métricas
├── ├── preprocess/                     # scripts que geram os datasets de treino
├── ├── utils/                          # scripts úteis de logger, data loader, data save, etc
├── ├── config.py                       # configura caminhos e conjunto de datasets
├── ├── export_model.py                 # treina o melhor modelo em todos os dados rotulados e exporta um bundle .joblib
├── ├── experiments.py                  # roda os 18 experimentos (6 datasets x 3 modelos) e salva métricas
└── └── predict.py                      # prediz attendance em linhas sem valor e salva em data/predictions/
```

---

## 11. Como Executar

### 11.1. Rodar o pré processamento

```bash
python -m src.process
```

Este comando processa o `data/raw/dataset.csv` e gera 6 datasets que serão usados para treino e avaliação dos modelos.

---

### 11.2. Rodar os experimentos

```bash
python -m src.experiments
```

Este comando:
- carrega cada dataset processado em `data/processed/`
- filtra linhas com `attendance` válido
- divide treino/teste (20% teste)
- executa `Decision Tree`, `Random Forest` e `MLP`
- salva `data/results/metrics.csv`

---

### 11.3. Exportar o melhor modelo

```bash
python src/export_model.py
```

Este comando encontra o melhor modelo em `data/results/metrics.csv`, treina-o em todos os dados rotulados e exporta um arquivo `.joblib` em `models/`.

---

### 11.4. Gerar predições para linhas sem `attendance`

```bash
python src/predict.py
```

O script re-treina o melhor modelo e grava as previsões em `data/predictions/`.

---

### 11.5. Executar a API de inferência

```bash
uvicorn src.api.main:app --reload
```

A API expõe `/predict/` e `/health` (além da documentação Swagger padrão em `/docs`)

---

### 11.6. Executar o frontend

```bash
python -m http.server 5500
```

Este comando inicia um servidor HTTP local para servir a interface web do projeto, permitindo acessar o frontend pelo navegador durante os testes e demonstrações.

---

### 11.7. Executar notebooks

- `notebooks/01_dataset_analysis.ipynb`
- `notebooks/02_experiments.ipynb`
- `notebooks/03_results_analysis.ipynb`

Se necessário, instale Jupyter separadamente:

```bash
pip install jupyter
```

---

## 12. Pipeline de Machine Learning

1. Coleta / Carregamento de dados
   - `data/raw/dataset.csv`
2. Limpeza
   - remoção de linhas sem `attendance`
   - remoção de linhas sem `venue_capacity`
   - filtro por `reporting_status == 'reported'` quando presente
3. Análise exploratória
   - notebooks e plots em `src/analysis/`
4. Pré-processamento
   - remoção de colunas leaky (`fill_rate`, `box_score`, `avg_ticket_price`)
   - one-hot encoding de variáveis categóricas
   - escala com `StandardScaler`
5. Treino
   - `Decision Tree`, `Random Forest`, `MLP`
6. Avaliação
   - `MAE`, `RMSE`, `R2`
7. Salvamento do modelo
   - `src/export_model.py` gera o bundle em `models/`

---

## 13. Modelos utilizados (Regressão)

- **Random Forest:** conjunto de múltiplas árvores de decisão para gerar previsões mais robustas;
- **Árvore de Decisão:** modelo baseado em regras de decisão organizadas em uma estrutura hierárquica;
- **Rede Neural (MLP):** rede neural multicamadas capaz de aprender padrões complexos nos dados.

---

## 14. Métricas Utilizadas

- **MAE (Mean Absolute Error):** média dos erros absolutos entre valores reais e previstos;
- **RMSE (Root Mean Squared Error):** mede o erro médio, penalizando mais fortemente erros maiores;
- **R² (Coeficiente de Determinação):** indica o quanto o modelo consegue explicar a variabilidade dos dados.

---

## 15. Resultados Principais

O arquivo `data/results/metrics.csv` contém os resultados dos 18 experimentos.

O melhor experimento registrado é:

- Modelo: `Random Forest`;
- Dataset: `v2_artist`;
- MAE: `4120.23`;
- RMSE: `6196.44`;
- R2: `0.9537`.

> [!NOTE]
> Caso queira uma avaliação das métricas mais profunda, rode o [notebook 3](notebooks/03_results_analysis.ipynb).

### 15.1 Top 3 resultados por R2

| Dataset              | Modelo        | MAE      | RMSE     | R2      |
|----------------------|---------------|----------|----------|---------|
| `v2_artist`          | Random Forest | 4120.23  | 6196.44  | 0.9537  |
| `v5_full`            | Random Forest | 4166.57  | 6215.86  | 0.9534  |
| `v4_artist_geo_time` | Random Forest | 4187.03  | 6307.05  | 0.9520  |

> [!IMPORTANT]
> A tabela acima usa valores reais extraídos de [`data/results/metrics.csv`](data/results/metrics.csv).

---

## 16. Limitações Conhecidas

- O dataset é específico ao contexto de shows de K-pop e pode não generalizar para outros gêneros;
- A validação é feita com divisão simples treino/teste, sem validação temporal ou cross-validation explícita;
- Locais de evento (`venues`) com menos entradas do dataset terão margem de erro superior, assim como Artistas com menos entradas;
- Locais de evento (`venues`) com capacidades menores terão margem de erro superior.

---

## 17. Melhorias Futuras

- Ajustar `src/config.py` para usar caminhos relativos e permitir execução em qualquer máquina.
- Adicionar validação cruzada e/ou validação temporal.
- Explorar mais engenharia de features e seleção automática de variáveis.

---

## 18. Autor

- Sofia Alves Toreti

---

## 19. Licença

Este projeto está licenciado sob a licença [**MIT**](LICENSE).