# Fessel-local entry point for build/test/lint. Runnable standalone
# (make -C src/fessel test) regardless of the host repo's build system.
# Each sub-project has its own dependency manifest and builds independently.

SHELL := /bin/bash
HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PY_PROJECTS := schemas pi/shared pi/supervisor pi/uploader webui/backend
# pi/video is handled separately: it needs a --system-site-packages venv
# for PyGObject (gi). See dev-env docs and I0.4.

.PHONY: test lint lint-py lint-frontend format types check-types sync \
        test-py test-video test-frontend \
        package-dpkg package-docker package-tanka

sync:
	@for p in $(PY_PROJECTS); do \
	  echo "== uv sync $$p =="; \
	  (cd "$(HERE)$$p" && uv sync) || exit 1; \
	done

test: test-py test-frontend

test-py:
	@for p in $(PY_PROJECTS); do \
	  echo "== pytest $$p =="; \
	  (cd "$(HERE)$$p" && uv run pytest -q) || exit 1; \
	done

# video tests require the system GStreamer/PyGObject venv (see docs).
test-video:
	cd "$(HERE)pi/video" && ./.venv/bin/python -m pytest -q

test-frontend:
	cd "$(HERE)webui/frontend" && npm ci && npm run test

lint: lint-py lint-frontend

lint-py:
	@for p in $(PY_PROJECTS) pi/video; do \
	  echo "== ruff $$p =="; \
	  (cd "$(HERE)$$p" && uvx ruff check .) || exit 1; \
	done

# Mirrors the CI frontend job's lint step (eslint via npm run lint).
lint-frontend:
	cd "$(HERE)webui/frontend" && npm ci && npm run lint

format:
	@for p in $(PY_PROJECTS) pi/video; do \
	  (cd "$(HERE)$$p" && uvx ruff format .) || exit 1; \
	done

types:
	cd "$(HERE)webui/frontend" && npm ci
	"$(HERE)tools/generate-types.sh"

# Sync guard: regenerate the TS types + Zod validators and fail if the
# committed output is stale (i.e. someone edited a model in schemas/ without
# re-running `make types`). The generated artifacts are load-bearing — the
# frontend's compile-time types AND its runtime validators come from them — so
# drift must break the build, not surface as a silent mismatch at runtime.
check-types: types
	@git diff --exit-code -- \
	  "$(HERE)webui/shared/schemas.ts" \
	  "$(HERE)webui/frontend/src/generated/validators.ts" \
	  || { echo "ERROR: generated types/validators are stale — run 'make types' and commit"; exit 1; }

# --- packaging placeholders (arrive in a later slice) ---
package-dpkg:
	@echo "dpkg packaging: not implemented yet (later slice)"

package-docker:
	@echo "docker image: not implemented yet (later slice)"

package-tanka:
	@echo "Tanka library: not implemented yet (later slice)"
