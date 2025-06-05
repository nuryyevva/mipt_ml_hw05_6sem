SHELL := /bin/sh

install:
	poetry install

test:
	poetry run pytest tests

lint:
	poetry run pre-commit run --show-diff-on-failure --color=always --all-files

hooks:
	poetry run pre-commit install --install-hooks

install_dataset:
	poetry run python src/data/data_loader.py

train:
	poetry run python src/main.py

inference:
	poetry run python notebooks/inference.py
