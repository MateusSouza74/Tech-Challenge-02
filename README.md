# Tech Challenge — Fase 02 · Sistema de Recomendação

Sistema de recomendação de produtos baseado no comportamento de navegação
dos usuários. Uma rede neural (NCF — Neural Collaborative Filtering, em
PyTorch) é comparada a baselines de Scikit-Learn, com pipeline
reprodutível em DVC, experimentos e Model Registry no MLflow.

Documentação do modelo: [Model Card](docs/MODEL_CARD.md).

## Estrutura do projeto

```
src/recsys/
  data/           # download, preprocessamento, feature engineering, Dataset PyTorch
  models/         # NCF, baselines (popularidade, KNN de usuarios) e ModelFactory
  preprocessing/  # estrategias de encoding (padrao Strategy)
  training/       # Trainer (early stopping), treino, experimentos e Model Registry
  evaluation/     # metricas de ranking e avaliacao comparativa
  utils/          # configuracoes centralizadas (Pydantic Settings)
tests/            # testes unitarios (pytest)
docs/             # Model Card
data/, models/    # artefatos versionados via DVC (fora do git)
configs/          # configuracoes externas
```

## Requisitos

- Python 3.11 ou superior
- [Poetry](https://python-poetry.org/) para gerenciamento de dependências
- Docker e Docker Compose (opcional, para rodar o pipeline em container)

## Instalação

```bash
# 1. Instale o Poetry (se ainda não tiver)
pip install poetry

# 2. Na raiz do repositório, instale as dependências
poetry install

# 3. Crie o arquivo de configuração local
cp .env.example .env        # Windows: copy .env.example .env

# 4. Valide o ambiente
poetry run python scripts/validate_env.py
```

## Obtenção dos dados

O projeto usa o MovieLens-20M como proxy de navegação de e-commerce
(avaliação >= 3.5 é tratada como interesse no produto). O download é
feito via kagglehub:

```bash
poetry run python -m recsys.data.download_data
```

Os CSVs ficam em `data/raw/` (cerca de 700 MB para o `rating.csv`).

## Executando o pipeline (DVC)

```bash
poetry run dvc repro
```

O DVC executa apenas os estágios cujas dependências mudaram:

| Estágio | O que faz | Saídas principais |
|---|---|---|
| `preprocess` | binariza, amostra usuários ativos, encoding, split temporal | `data/processed/{train,val,test}.csv` |
| `feature_eng` | gera negativos (4:1) para treino e validação | `train_with_negatives.csv`, `val_with_negatives.csv` |
| `train` | treina o NCF com early stopping e loga no MLflow | `models/ncf_model.pt` |
| `evaluate` | compara NCF vs baselines com 4 métricas de ranking | `models/evaluation_report.json` |

Para rodar um estágio específico: `poetry run dvc repro train`.
