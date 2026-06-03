#!/bin/bash
# run_verification.sh — Run multi-block TGV verification
# Usage: bash debug/run_verification.sh [nsteps]

set -e

NSTEP=${1:-5}

echo "========================================"
echo "  Multi-Block TGV Verification"
echo "  Steps: $NSTEP"
echo "========================================"

# Step 1: Commit debug scripts
echo ""
echo "Step 1: Committing debug scripts..."
git add debug/ .gitignore
git commit -m "Add multi-block TGV verification scripts for Y-mesh"

# Step 2: Run TGV verification
echo ""
echo "Step 2: Running TGV verification on Y-mesh..."
julia debug/run_y_mesh_tgv.jl $NSTEP

# Step 3: Check results
echo ""
echo "Step 3: Checking results..."
if [ -f debug/tgv_verification.txt ]; then
    cat debug/tgv_verification.txt
    echo ""
    if grep -q "PASS" debug/tgv_verification.txt; then
        echo "✓ VERIFICATION PASSED: No numerical artifacts detected."
        exit 0
    else
        echo "✗ VERIFICATION FAILED: Numerical artifacts detected."
        exit 1
    fi
else
    echo "ERROR: Verification output not found."
    exit 1
fi
