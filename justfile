# Justfile for temps
current_dir := justfile_directory()
# Determine workspace path based on hostname
WORKSPACE_PATH := "workspace/runtime/audio-pipeline"

# Rsync exclusions (one per line for easy modification)
RSYNC_EXCLUDES := \
'''
--exclude=*.log
--exclude=.venv
--exclude=data
--exclude=.mypy_cache
--exclude=notebooks
--exclude=.ruff_cache
--exclude=*.egg-info
--exclude=__pycache__
--exclude=.pytest_cache
--exclude=dist
--exclude=build
--exclude=.git
'''


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


# Internal: rsync helper with exclusions
_rsync-to source_path target_path:
    #!/usr/bin/env bash
    echo "Syncing source code to remote cluster..."
    EXCLUDES=$(echo "{{ RSYNC_EXCLUDES }}" | tr '\n' ' ')
    if [ -z "${SSHPASS}" ]; then
        rsync -azP --delete ${EXCLUDES} "{{source_path}}/" "{{target_path}}/"
    else
        sshpass -e rsync -azP --delete ${EXCLUDES} "{{source_path}}/" "{{target_path}}/"
    fi

sync-oberon:
    @echo "Deploying to oberon2 ..."
    just _rsync-to "{{current_dir}}/" "oberon2:{{WORKSPACE_PATH}}/"
    @echo "✓ Code synced successfully!"
