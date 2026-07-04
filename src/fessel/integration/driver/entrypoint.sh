#!/usr/bin/env bash
# Run the driver; exit code propagates so the Job fails on assertion failure.
# (The Xvfb/Chrome setup died with mediamtx — the WHEP client is aiortc now.)
set -euo pipefail
exec python /driver/run_tests.py
