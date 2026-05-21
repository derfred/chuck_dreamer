#!/usr/bin/env bash
# Sync the local working tree to a remote host over ssh, re-syncing on any change.
#
# Usage:
#   scripts/rsync_watch.sh <user@host>:<remote/path> [--ignore <glob>]... [--ssh-port N] [--initial-only] [--progress]
#
# Examples:
#   scripts/rsync_watch.sh me@gpu-box:~/world_model_pusher \
#       --ignore '.git/' --ignore '__pycache__/' --ignore '*.safetensors'
#
# Ignore globs use rsync --exclude syntax (trailing / means directory).
# Requires: rsync. Uses fswatch if installed (event-driven); otherwise falls back
# to a polling loop.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,14p' "$0"
  exit 1
fi

REMOTE="$1"; shift
SSH_PORT=22
INITIAL_ONLY=0
PROGRESS=0

# --- Default excludes (edit these) -------------------------------------------
# rsync --exclude syntax: trailing '/' means directory, '*' is a glob.
# Paths are matched relative to the sync root (the current working directory).
EXCLUDES=(
  --exclude=.git/
  --exclude=__pycache__/
  --exclude='*.safetensors'
  --exclude=.venv/
  --exclude='*.pyc'

  --exclude=data/eval-real
  --exclude=data/generated-pushes
  --exclude=data/new-generated-pushes
  --exclude=data/oldtraces
  --exclude=data/warmup-real
  --exclude=data/eval-old
  --exclude=data/warmup-old
  --exclude=checkpoints
  --exclude=data/real_imported
  --exclude=data/task2_t
)
# -----------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ignore)
      [[ $# -lt 2 ]] && { echo "--ignore needs a value" >&2; exit 1; }
      EXCLUDES+=("--exclude=$2"); shift 2 ;;
    --ssh-port)
      [[ $# -lt 2 ]] && { echo "--ssh-port needs a value" >&2; exit 1; }
      SSH_PORT="$2"; shift 2 ;;
    --initial-only)
      INITIAL_ONLY=1; shift ;;
    --progress)
      PROGRESS=1; shift ;;
    -h|--help)
      sed -n '2,14p' "$0"; exit 0 ;;
    *)
      echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

SRC_DIR="$(pwd)"

RSYNC_OPTS=(
  -az
  --stats
  --itemize-changes
  -e "ssh -p ${SSH_PORT} -o ControlMaster=auto -o ControlPath=/tmp/rsync_watch_%r@%h:%p -o ControlPersist=60s"
)

do_sync() {
  local ts start end elapsed out status
  ts="$(date +%H:%M:%S)"
  printf '[%s] syncing -> %s\n' "$ts" "$REMOTE"

  start=$(date +%s)

  if [[ $PROGRESS -eq 1 ]]; then
    # Stream rsync output live so big transfers show progress.
    # Use --progress (per-file) since macOS' bundled rsync 2.6 lacks --info=progress2.
    set +e
    rsync "${RSYNC_OPTS[@]}" --progress "${EXCLUDES[@]}" "${SRC_DIR}/" "${REMOTE}/"
    status=$?
    set -e
    end=$(date +%s)
    elapsed=$((end - start))

    if [[ $status -ne 0 ]]; then
      printf '[%s] rsync failed (exit %d, %ds) — continuing\n\n' "$(date +%H:%M:%S)" "$status" "$elapsed" >&2
      return
    fi

    printf '[%s] done in %ds\n\n' "$(date +%H:%M:%S)" "$elapsed"
    return
  fi

  set +e
  out="$(rsync "${RSYNC_OPTS[@]}" "${EXCLUDES[@]}" "${SRC_DIR}/" "${REMOTE}/" 2>&1)"
  status=$?
  set -e
  end=$(date +%s)
  elapsed=$((end - start))

  if [[ $status -ne 0 ]]; then
    printf '%s\n' "$out" >&2
    printf '[%s] rsync failed (exit %d, %ds) — continuing\n\n' "$(date +%H:%M:%S)" "$status" "$elapsed" >&2
    return
  fi

  # Itemized lines start with a YXcstpoguax-style flag block; first char is the update type.
  # Show up to 10 changed paths, then a count if there are more.
  local changed total_changed
  changed="$(printf '%s\n' "$out" | grep -E '^[<>ch.*][fdLDS]' || true)"
  total_changed=0
  if [[ -n "$changed" ]]; then
    total_changed=$(printf '%s\n' "$changed" | wc -l | tr -d ' ')
    printf '%s\n' "$changed" | head -n 10 | sed 's/^/  /'
    if (( total_changed > 10 )); then
      printf '  ... and %d more\n' "$((total_changed - 10))"
    fi
  fi

  # Pull the human-readable totals out of --stats.
  local files_xfer bytes_sent
  files_xfer="$(printf '%s\n' "$out" | sed -n 's/^Number of .*files transferred: *//p' | head -n 1)"
  bytes_sent="$(printf '%s\n' "$out"  | sed -n 's/^Total bytes sent: *//p')"

  printf '[%s] done in %ds — %s changed, %s files transferred, %s bytes sent\n\n' \
    "$(date +%H:%M:%S)" "$elapsed" "${total_changed:-0}" "${files_xfer:-0}" "${bytes_sent:-0}"
}

# Initial sync.
do_sync

if [[ $INITIAL_ONLY -eq 1 ]]; then
  exit 0
fi

# Build a regex of excludes for fswatch so we don't fire on ignored paths.
# fswatch --exclude takes an ERE; we translate simple globs.
FSWATCH_EXCLUDES=()
for e in "${EXCLUDES[@]}"; do
  pat="${e#--exclude=}"
  # Strip leading / if present.
  pat="${pat#/}"
  # Escape dots, then turn glob * into .*, ? into .
  pat_re="${pat//./\\.}"
  pat_re="${pat_re//\*/.*}"
  pat_re="${pat_re//\?/.}"
  FSWATCH_EXCLUDES+=("--exclude=${pat_re}")
done

# Debounce: collect bursts of changes within a short window.
DEBOUNCE_SECS="${DEBOUNCE_SECS:-0.5}"

# Clean shutdown: kill the whole process group so fswatch + subshells exit on Ctrl-C.
FSWATCH_PID=""
cleanup() {
  trap - INT TERM EXIT
  [[ -n "$FSWATCH_PID" ]] && kill "$FSWATCH_PID" 2>/dev/null || true
  echo
  echo "stopped."
  exit 0
}
trap cleanup INT TERM

if command -v fswatch >/dev/null 2>&1; then
  echo "watching for changes with fswatch (Ctrl-C to stop)..."
  # Run fswatch in the background so the shell keeps handling signals directly.
  # -o batches events and emits a single line per batch; -l sets latency.
  FIFO="$(mktemp -u /tmp/rsync_watch_fifo.XXXXXX)"
  mkfifo "$FIFO"
  fswatch -o -l "$DEBOUNCE_SECS" "${FSWATCH_EXCLUDES[@]}" "$SRC_DIR" > "$FIFO" &
  FSWATCH_PID=$!
  # Reading from the FIFO in the main shell means SIGINT reaches us directly.
  while read -r _ <"$FIFO"; do
    do_sync
  done
  rm -f "$FIFO"
else
  echo "fswatch not found; falling back to polling every 2s (brew install fswatch for event-driven mode)"
  LAST_HASH=""
  while true; do
    # Cheap change signal: mtime of the most-recently-modified non-excluded file.
    # We let rsync itself decide what actually needs transferring.
    find_args=(-type f)
    for e in "${EXCLUDES[@]}"; do
      pat="${e#--exclude=}"
      pat="${pat%/}"
      find_args+=(! -path "*/${pat}/*" ! -name "${pat}")
    done
    CUR_HASH="$(find "$SRC_DIR" "${find_args[@]}" -newer /tmp/.rsync_watch_marker 2>/dev/null | head -n 1)"
    if [[ -n "$CUR_HASH" || -z "$LAST_HASH" ]]; then
      do_sync
      touch /tmp/.rsync_watch_marker
      LAST_HASH="synced"
    fi
    sleep 2
  done
fi
