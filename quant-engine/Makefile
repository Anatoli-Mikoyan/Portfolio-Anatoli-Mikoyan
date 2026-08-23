.DEFAULT_GOAL := help
PY := python3

.PHONY: help install lint type test cov check example clean docker-ci

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Installe le projet en mode editable avec les outils de dev
	$(PY) -m pip install -e ".[dev]"

lint:  ## Analyse statique (ruff)
	$(PY) -m ruff check .

format:  ## Corrige ce qui peut l'etre automatiquement
	$(PY) -m ruff check --fix .

type:  ## Verification de types (mypy, mode strict)
	$(PY) -m mypy

test:  ## Tests unitaires
	$(PY) -m pytest -q

cov:  ## Tests avec rapport de couverture
	$(PY) -m pytest -q --cov=quant_engine --cov-report=term-missing --cov-report=html

check: lint type cov  ## Chaine complete, telle qu'executee en CI

example:  ## Exemple reproductible de bout en bout (hors ligne)
	PYTHONPATH=src $(PY) examples/step1_data.py

network-test:  ## Tests necessitant un acces reseau reel (exclus par defaut)
	$(PY) -m pytest -q -m network

docker-ci:  ## Chaine complete dans un conteneur
	docker compose run --rm ci

clean:  ## Supprime les artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
