#!/usr/bin/env bash
set -euo pipefail

export TF_ENABLE_ONEDNN_OPTS=0

PASS=0
FAIL=0

run_test() {
    local name="$1"
    shift
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  TEST: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if "$@"; then
        echo "  ✓ PASSED: $name"
        PASS=$((PASS + 1))
    else
        echo "  ✗ FAILED: $name"
        FAIL=$((FAIL + 1))
    fi
}

# ---------------------------------------------------------------------------
# 1. Create / sync venv via uv
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setting up environment"
echo "============================================================"

rm -rf .venv
uv venv

# Activate the venv so plain `python` resolves to it in this session.
# On Windows/MSYS the activate script lives under Scripts/, not bin/.
if [ -f ".venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

uv pip install -e .

echo "  Python: $(python --version)"
python -c "import tensorflow; print('Tensorflow version:', tensorflow.__version__)"
echo "  uv venv ready."

run_test "GMM (no epochs needed)" \
    python main.py --model gmm --ticker SPY --samples 10

run_test "CTGAN (1 epoch)" \
    python main.py --model ctgan --ticker SPY --epochs 1 --samples 10

run_test "TimeGAN (1 epoch)" \
    python main.py --model timegan --ticker SPY --epochs 1 --samples 10

run_test "DRAGAN (1 epoch)" \
    python main.py --model dragan --ticker SPY --epochs 1 --samples 10

# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "============================================================"

[ "$FAIL" -eq 0 ]   # exit 1 if any test failed
