#!/usr/bin/env bash
# Run the driver; exit code propagates so the Job fails on assertion failure.
# (The WHEP client is aiortc; no Xvfb/Chrome needed.)
set -euo pipefail
exec python /driver/run_tests.py
