default: check

# Install the package with every optional extra and dev tools.
install:
    uv sync --all-extras

lint:
    uv run ruff check src

fmt:
    uv run ruff format src

type:
    uv run mypy

test:
    uv run pytest

check: lint type test
