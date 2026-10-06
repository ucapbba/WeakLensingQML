#!/usr/bin/env bash
# Offload compute_fisher_matrix() to a remote machine over SSH.
#
# Only the cpp/ source tree and the two input files are copied - no Python
# packages, notebooks, or other data are needed on the remote side.
#
# Prerequisites on the remote machine: cmake, a C++17 compiler, pkg-config,
# Eigen3, healpix_cxx (+ cfitsio), OpenMP, and python3 (stdlib only).
#
# Usage:
#   ./run_remote_fisher.sh <user@host> <remote_workdir> <local_mask_path> <local_cl_path> <local_fisher_out_path>
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <user@host> <remote_workdir> <local_mask_path> <local_cl_path> <local_fisher_out_path>" >&2
    exit 1
fi

REMOTE=$1
REMOTE_DIR=$2
MASK_LOCAL=$3
CL_LOCAL=$4
FISHER_OUT_LOCAL=$5

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Creating remote directories"
ssh "$REMOTE" "mkdir -p '$REMOTE_DIR/cpp' '$REMOTE_DIR/data'"

echo "==> Syncing C++ source to $REMOTE:$REMOTE_DIR/cpp"
rsync -avz --exclude 'build*' "$REPO_ROOT/cpp/" "$REMOTE:$REMOTE_DIR/cpp/"

echo "==> Syncing input data files"
rsync -avz "$MASK_LOCAL" "$REMOTE:$REMOTE_DIR/data/mask.fits"
rsync -avz "$CL_LOCAL" "$REMOTE:$REMOTE_DIR/data/cl_theory.dat"

echo "==> Building on remote (Release)"
ssh "$REMOTE" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_DIR/cpp"
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j"\$(nproc)"
EOF

echo "==> Running compute_fisher_matrix on remote"
ssh "$REMOTE" python3 - "$REMOTE_DIR" <<'PYEOF'
import ctypes
import sys

remote_dir = sys.argv[1]
lib = ctypes.CDLL(f"{remote_dir}/cpp/build-release/libQMLShearLib.so")
lib.compute_fisher_matrix(
    f"{remote_dir}/data/mask.fits".encode(),
    f"{remote_dir}/data/cl_theory.dat".encode(),
    f"{remote_dir}/data/fisher_matrix_out.dat".encode(),
)
PYEOF

echo "==> Copying result back to $FISHER_OUT_LOCAL"
rsync -avz "$REMOTE:$REMOTE_DIR/data/fisher_matrix_out.dat" "$FISHER_OUT_LOCAL"

echo "Done. Fisher matrix saved to $FISHER_OUT_LOCAL"
