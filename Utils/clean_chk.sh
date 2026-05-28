#!/bin/bash
# =============================================================================
# clean_chk.sh — Clean CHK checkpoint files, keeping every Nth step
#
# Usage:
#   bash clean_chk.sh [CHK_DIR] [KEEP_INTERVAL]
#
# Examples:
#   bash clean_chk.sh CHK 10000       # keep only steps divisible by 10000
#   bash clean_chk.sh CHK 50000       # keep only steps divisible by 50000
#   bash clean_chk.sh                 # default: CHK dir, keep every 10000
#
# What it does:
#   - Scans for chk-{step}-b*.h5 files
#   - Keeps files where step % KEEP_INTERVAL == 0
#   - Deletes files where step % KEEP_INTERVAL != 0
#   - Always keeps the latest (highest step) checkpoint
#   - Dry-run by default: shows what would be deleted without actually deleting
#   - Pass --delete as 3rd arg to actually remove files
# =============================================================================

set -euo pipefail

CHK_DIR="${1:-CHK}"
KEEP_INTERVAL="${2:-10000}"
DO_DELETE="${3:---dry-run}"

if [ ! -d "$CHK_DIR" ]; then
    echo "ERROR: Directory '$CHK_DIR' not found."
    exit 1
fi

echo "═══════════════════════════════════════════════════════"
echo "  CHK Checkpoint Cleaner"
echo "═══════════════════════════════════════════════════════"
echo "  Directory:     $CHK_DIR"
echo "  Keep interval: every $KEEP_INTERVAL steps"
echo "  Mode:          $DO_DELETE"
echo "═══════════════════════════════════════════════════════"

# Extract unique step numbers from chk-{step}-b*.h5 filenames
steps=$(ls "$CHK_DIR"/chk-*-b*.h5 2>/dev/null \
    | sed -E 's|.*/chk-([0-9]+)-b[0-9]+\.h5|\1|' \
    | sort -un)

if [ -z "$steps" ]; then
    echo "  No checkpoint files found."
    exit 0
fi

# Find the latest step (always keep it)
max_step=$(echo "$steps" | tail -1)
total_steps=$(echo "$steps" | wc -l)
echo "  Total steps:   $total_steps"
echo "  Latest step:   $max_step (always kept)"
echo ""

delete_count=0
delete_bytes=0
keep_count=0

for step in $steps; do
    if [ "$step" -eq "$max_step" ]; then
        keep_count=$((keep_count + 1))
        continue
    fi

    if [ $((step % KEEP_INTERVAL)) -eq 0 ]; then
        keep_count=$((keep_count + 1))
    else
        files=$(ls "$CHK_DIR"/chk-${step}-b*.h5 2>/dev/null || true)
        if [ -n "$files" ]; then
            for f in $files; do
                fsize=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
                delete_bytes=$((delete_bytes + fsize))
            done
            delete_count=$((delete_count + 1))

            if [ "$DO_DELETE" = "--delete" ]; then
                echo "  [DEL] step $step"
                rm -f $files
            else
                echo "  [DRY] step $step  (would delete)"
            fi
        fi
    fi
done

echo ""
echo "───────────────────────────────────────────────────────"
echo "  Steps to keep:   $keep_count"
echo "  Steps to delete: $delete_count"
echo "  Space to free:   $(echo "scale=2; $delete_bytes / 1073741824" | bc) GB"

if [ "$DO_DELETE" != "--delete" ]; then
    echo ""
    echo "  ⚠  DRY RUN — no files were deleted."
    echo "  To actually delete, run:"
    echo "    bash $0 $CHK_DIR $KEEP_INTERVAL --delete"
fi
echo "═══════════════════════════════════════════════════════"
