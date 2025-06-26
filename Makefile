.PHONY: format lint

format:
	uvx ruff check --select I --fix
	uvx ruff check --fix
	uvx ruff format


lint:
	uvx ruff check
	uvx ruff format --check