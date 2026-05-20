#!/bin/bash
# Live-monitor a remote autoresearch run by pulling lineage.jsonl files
# from the Mac at intervals and regenerating the dashboard locally.
#
# Usage:
#   ./scripts/monitor_remote.sh <state_root_glob> [out_dir]
#
# Example:
#   ./scripts/monitor_remote.sh 'synth_run_parallel_*' /tmp/dashboard
#
# Then open out_dir/dashboard.html in a browser (use python -m http.server
# from out_dir if you want auto-refresh; the HTML has --refresh 30 set).
#
# ctrl-c to stop.

set -e

GLOB="${1:-synth_run_*}"
OUT_DIR="${2:-/tmp/convex_dashboard}"
INTERVAL="${3:-30}"
REMOTE="${REMOTE:-revantkasichainula@100.98.66.89}"

mkdir -p "$OUT_DIR/lineage"
cd "$(dirname "$0")/.."  # repo root

echo "live-monitoring $REMOTE:$GLOB every ${INTERVAL}s"
echo "dashboard: $OUT_DIR/dashboard.html"
echo "ctrl-c to stop"

while true; do
    # Pull just the lineage files (not the bulky runs/ artifacts).
    rsync -az --include="*/" --include="lineage.jsonl" --exclude="*" \
        "$REMOTE:convexkernels/$GLOB/" \
        "$OUT_DIR/lineage/" 2>/dev/null || true

    # Find dirs with lineage files and render dashboard.
    dirs=$(find "$OUT_DIR/lineage" -name "lineage.jsonl" -exec dirname {} \;)
    if [ -n "$dirs" ]; then
        # shellcheck disable=SC2086
        .venv/bin/python scripts/lineage_dashboard.py $dirs \
            --out "$OUT_DIR/dashboard.html" --refresh "$INTERVAL" \
            > /dev/null 2>&1 \
            && echo "$(date +%H:%M:%S) updated dashboard ($(echo "$dirs" | wc -l) lineage files)"
    else
        echo "$(date +%H:%M:%S) no lineage files yet under $OUT_DIR/lineage/"
    fi

    sleep "$INTERVAL"
done
