#!/usr/bin/env bash
# Run compute_fisher_matrix() locally (in this container/machine) and time it,
# for direct comparison against run_remote_fisher.sh.
#
# Note: the OpenMP parallel-for loops in detail/ComputeCl.h hardcode
# num_threads(10), so this will use at most 10 threads here too, regardless
# of how many cores this machine has.
#
# Usage:
#   ./run_local_fisher.sh <local_mask_path> <local_cl_path> <local_fisher_out_path>
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <local_mask_path> <local_cl_path> <local_fisher_out_path>" >&2
    exit 1
fi

MASK_LOCAL=$1
CL_LOCAL=$2
FISHER_OUT_LOCAL=$3

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB_PATH="$REPO_ROOT/cpp/build-release/libQMLShearLib.so"

if [[ ! -f "$LIB_PATH" ]]; then
    echo "==> Building locally (Release)"
    cmake -S "$REPO_ROOT/cpp" -B "$REPO_ROOT/cpp/build-release" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$REPO_ROOT/cpp/build-release" -j"$(nproc)"
else
    echo "==> Using existing build at $LIB_PATH"
fi

echo "==> Running compute_fisher_matrix locally"
START_TIME=$(date +%s.%N)

python3 - "$LIB_PATH" "$MASK_LOCAL" "$CL_LOCAL" "$FISHER_OUT_LOCAL" <<'PYEOF'
import ctypes
import sys

lib_path, mask_path, cl_path, out_path = sys.argv[1:5]
lib = ctypes.CDLL(lib_path)
lib.compute_fisher_matrix(mask_path.encode(), cl_path.encode(), out_path.encode())
PYEOF

END_TIME=$(date +%s.%N)
echo "Done. Fisher matrix saved to $FISHER_OUT_LOCAL"
echo "Elapsed: $(echo "$END_TIME - $START_TIME" | bc) seconds"
