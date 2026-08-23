# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Couche de dependances separee : le code change bien plus souvent que les deps.
COPY pyproject.toml README.md ./
COPY src/quant_engine/__init__.py src/quant_engine/__init__.py
RUN pip install --upgrade pip && pip install -e ".[dev,yfinance]"

COPY . .
RUN pip install -e ".[dev,yfinance]"

# Utilisateur non privilegie : un conteneur d'analyse n'a aucune raison
# de tourner en root.
RUN useradd --create-home --uid 1000 quant && chown -R quant:quant /app
USER quant

CMD ["python", "examples/step1_data.py"]


FROM base AS ci
USER quant
CMD ["sh", "-lc", "ruff check . && mypy && pytest -q --cov=quant_engine --cov-report=term-missing"]
