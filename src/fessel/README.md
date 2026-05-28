# Fessel

Robot-arm monitoring system. This tree is self-contained under
`src/fessel/` and builds independently of the host repository.

- **supervisor** — Pi-side safety/control-plane process (Slice 1: live
  activate/deactivate relay + heartbeat).
- **video** — Pi-side GStreamer pipeline + live state machine.
- **webui** — cluster-side backend (FastAPI) + frontend (React) + mediamtx.
- **schemas** — canonical wire shapes (single source of truth).

See `docs/fessel/` for the requirements, architecture, and slice plans.

## Layout

```
schemas/            canonical payload + topic schemas (Python source of truth)
pi/supervisor/      supervisor process (clean venv)
pi/video/           video process (system-site-packages venv for gi)
pi/shared/          shared Pi-side code (MQTT client, config, topics)
pi/deploy/          systemd units, mosquitto.conf, config examples
webui/backend/      FastAPI: signed WHEP URL minting
webui/frontend/     React WHEP client
webui/shared/       generated TS types (do not hand-edit)
webui/deploy/       mediamtx config template + k8s manifests
tools/              generate-types.sh, render-mediamtx-config.sh
Makefile            build/test/lint entry point
Procfile            local dev (mosquitto + supervisor + video + backend + frontend)
```

## Build & test

```
make -C src/fessel sync          # uv sync each clean-venv project
make -C src/fessel test          # pytest (clean-venv projects) + frontend tsc
make -C src/fessel types         # regenerate webui/shared/schemas.ts
make -C src/fessel test-video    # video tests (needs the gi venv, below)
```

## PyGObject (gi) venv for `video` — IMPORTANT

`video` drives GStreamer through PyGObject (`gi`), which binds to the
**system** GObject-introspection libraries. Its venv **must** be created
with `--system-site-packages` against the **system-managed** Python, and
the process **must** be invoked via the venv binary directly. A wrapper
that re-resolves the interpreter (e.g. `uv run`) ignores
`--system-site-packages` and fails to find `gi`.

macOS (development):

```
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad pygobject3
cd src/fessel/pi/video
UV_PYTHON=/opt/homebrew/bin/python3.13 uv venv --system-site-packages .venv
./.venv/bin/python -m pip install -e . -e ../../schemas -e ../shared pytest
# If a system pydantic shadows the venv copy (Homebrew installs one as a
# namespace package), force a real one into the venv:
./.venv/bin/python -m pip install --ignore-installed "pydantic>=2.6" pydantic-core
# run it (note: the binary directly, NOT `uv run`):
VIDEO_CONFIG=../deploy/video.yaml.example ./.venv/bin/video
```

Pi (apt-provided gi + GStreamer):

```
sudo apt install python3-gi gir1.2-gst-plugins-bad-1.0 \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-tools v4l-utils mosquitto
cd src/fessel/pi/video
python3 -m venv --system-site-packages .venv
./.venv/bin/pip install -e . -e ../../schemas -e ../shared pytest
```

For development without a camera, set `camera.use_test_source: true` in
`video.yaml` to use a `videotestsrc` pattern. Hardware-encoder selection
still fails loud if the platform encoder is missing — it never silently
falls back to software encoding.

## Local dev

`supervisor`/`backend` use clean uv venvs:

```
cd src/fessel/pi/supervisor && uv sync
cd src/fessel/webui/backend && uv sync
```

Then run everything with a Procfile runner:

```
cd src/fessel && honcho start   # or foreman
```
