# PREPROCESSING PLAN
> Feature Engineering & Variable Selection

## 1. Estrutura de Diretórios

```text
data/
├── raw/
│   └── dataset.csv
│
└── processed/
    ├── attendance_v1_baseline.csv
    ├── attendance_v2_artist.csv
    ├── attendance_v3_artist_geo.csv
    ├── attendance_v4_artist_geo_time.csv
    ├── attendance_v5_full.csv
    └── attendance_v6_no_artist.csv
```

---

## 2. Experimentos de Feature Ablation

| Experimento | Conjunto de Features      | Hipótese Central                                   |
| ----------- | ------------------------- | -------------------------------------------------- |
| E1          | Venue + Geography         | Localização e tamanho explicam parte da attendance |
| E2          | E1 + Artist               | Popularidade do artista supera o venue             |
| E3          | E2 + Time                 | Sazonalidade e ciclo econômico têm efeito          |
| E4          | Full                      | Modelo completo — melhor performance esperada      |
| E5          | Full – Artist             | Teste de dependência da identidade do artista      |
| E6          | Full – Time               | Teste de relevância dos efeitos temporais          |

> [!NOTE]
> Esses experimentos permitem responder perguntas reais:
> quanto o **artista** importa? quanto a **localização** importa?
> existe efeito de **final de semana**? existe efeito **pós-2023**?

---

## 3. Datasets

### 3.1. V1: Baseline

**Objetivo:** Descobrir quanto da attendance é explicada apenas pelo local.

```
venue_type
venue_capacity
continent
country
city
show_nights
```

> [!IMPORTANT]
> **Hipótese:** Localização e tamanho do venue explicam parte importante da attendance.

---

### 3.2. V2: Artist Features

Adicionar ao V1:

```
artist
gender
generation
members
company
years_since_debut
```

> [!IMPORTANT]
> **Hipótese:** A popularidade do artista é mais importante que o venue.

---

### 3.3. V3: Artist + Geography

Adicionar interação completa entre artista e país, mantendo as features geográficas do V1.

Features novas:

```
artist × country  (interação)
```

> [!IMPORTANT]
> **Hipótese:** Certos artistas performam melhor em determinados mercados.
>
> Exemplos esperados:
> - TWICE > Japan
> - SM artists > Thailand
> - Stray Kids > Latin America

---

### 3.4. V4: Time Effects

Adicionar ao V3:

```
show_year
show_month
show_quarter
weekday_1
weekday_n
is_weekend
```

> [!IMPORTANT]
> **Hipótese:** Existe sazonalidade e efeito econômico.
> 
> Observação: fill rate médio de 2023 > 2026.

---

### 3.5. V5: Full Model

Todas as features combinadas:

```
artist           generation       tour_type
gender           members          stage_setup
company          continent        country
city             venue_type       venue_capacity
show_nights      show_year        show_month
weekday          years_since_debut
```

> [!IMPORTANT]
> Modelo esperado com melhor performance geral.

---

### 3.6. V6: No Artist

Remover do V5:

```
artist
company
```

Manter:

```
generation
members
years_since_debut
```

> [!IMPORTANT]
> **Hipótese:** O modelo depende demais da identidade do artista.
>
> Se `R² Full = 0.85` e `R² No Artist = 0.35`, conclui-se que
> attendance é dominada pela popularidade específica do artista.

---

## 4. Features a Criar

### 4.1. `years_since_debut`

```python
years_since_debut = show_date - debut_date
```

Usar diferença de datas em vez do campo `debut` bruto.

---

### 4.2. `is_weekend`

```python
is_weekend = 1  # Fri, Sat, Sun
is_weekend = 0  # Mon, Tue, Wed, Thu
```

---

### 4.3. `show_year` e `show_month`

Extrair de `day_1`.

```python
show_year  = day_1.year
show_month = day_1.month
```

---

### 4.4. `season`

Criar a partir de `show_month`.

```
Summer | Autumn | Winter | Spring
```

> [!WARNING]
> Shows ocorrem em hemisférios diferentes.
> 
> **Alternativa:** usar `month` como numérico e deixar o modelo aprender sozinho.

---

### 4.5. `artist_age_bucket`

Transformar `years_since_debut` em categoria ordinal:

```
rookie       → 0–2 anos
growing      → 3–5 anos
established  → 6–10 anos
legacy       → 11+ anos
```

Testar como feature categórica vs. numérica.

---

## 5. Features a Remover (Target Leakage)

As seguintes colunas **não devem ser usadas** como features preditoras:

```
attendance        # target
fill_rate         # derivado do target
box_score         # derivado do target
avg_ticket_price  # correlacionado com o target
```

---

## 6. Perguntas que os Experimentos Respondem

- Quanto o artista importa para a attendance?
- Quanto a localização geográfica importa?
- Quanto o momento da indústria (ano/mês) importa?
- Existe efeito de final de semana?
- Existe queda estrutural de fill rate após 2023?
- O modelo generaliza para artistas desconhecidos?