.PHONY: dev-setup test fmt lint

dev-setup:
	bash scripts/dev_setup.sh

test:
	pytest -q

fmt:
	isort . --profile black
	black .

lint:
	ruff check . --fix || true
