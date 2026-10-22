#!/usr/bin/env bash
# Offload compute_fisher_matrix() to a remote machine over SSH.
#
# Only the cpp/ source tree and the two input files are copied (with --checksum: the per-combo
# masks all have the same size and can share an mtime second, which defeats rsync's quick check) - no Python
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

# Retry an idempotent transfer through transient network drops (e.g. the local machine sleeping):
# up to 30 attempts, 60 s apart
retry() {
    local attempt
    for attempt in $(seq 1 30); do
        "$@" && return 0
        echo "    attempt $attempt/30 failed; retrying in 60 s" >&2
        sleep 60
    done
    echo "Giving up after 30 attempts: $*" >&2
    return 1
}

echo "==> Creating remote directories"
retry ssh "$REMOTE" "mkdir -p '$REMOTE_DIR/cpp' '$REMOTE_DIR/data'"

echo "==> Syncing C++ source to $REMOTE:$REMOTE_DIR/cpp"
retry rsync -avz --checksum --exclude 'build*' "$REPO_ROOT/cpp/" "$REMOTE:$REMOTE_DIR/cpp/"

echo "==> Syncing input data files"
retry rsync -avz --checksum "$MASK_LOCAL" "$REMOTE:$REMOTE_DIR/data/mask.fits"
retry rsync -avz --checksum "$CL_LOCAL" "$REMOTE:$REMOTE_DIR/data/cl_theory.dat"

echo "==> Building on remote (Release)"
ssh "$REMOTE" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_DIR/cpp"
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j"\$(nproc)"
EOF

echo "==> Running compute_fisher_matrix on remote (detached, so it survives SSH drops)"
ssh "$REMOTE" bash -s -- "$REMOTE_DIR" <<'EOF'
set -euo pipefail
REMOTE_DIR=$1
rm -f "$REMOTE_DIR/data/fisher_matrix_out.dat" "$REMOTE_DIR/data/fisher.done"
cat > "$REMOTE_DIR/data/run_fisher.py" <<'PYEOF'
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
nohup bash -c "python3 '$REMOTE_DIR/data/run_fisher.py' '$REMOTE_DIR' > '$REMOTE_DIR/data/fisher.log' 2>&1; echo \$? > '$REMOTE_DIR/data/fisher.done'" < /dev/null > /dev/null 2>&1 &
EOF

# Poll with short, independent SSH connections; tolerate up to an hour of the server being unreachable
echo "==> Waiting for remote Fisher matrix (log: $REMOTE:$REMOTE_DIR/data/fisher.log)"
unreachable=0
while true; do
    sleep 60
    if ! status=$(ssh -o ConnectTimeout=20 "$REMOTE" "cat '$REMOTE_DIR/data/fisher.done' 2>/dev/null || echo running"); then
        unreachable=$((unreachable + 1))
        echo "    remote unreachable ($unreachable/60)"
        [[ $unreachable -lt 60 ]] || { echo "Giving up: remote unreachable for an hour" >&2; exit 1; }
        continue
    fi
    unreachable=0
    case $status in
        running) ;;
        0) break ;;
        *) echo "Remote Fisher computation failed (exit code $status); tail of log:" >&2
           ssh "$REMOTE" "tail -20 '$REMOTE_DIR/data/fisher.log'" >&2 || true
           exit 1 ;;
    esac
done

echo "==> Copying result back to $FISHER_OUT_LOCAL"
retry rsync -avz --checksum "$REMOTE:$REMOTE_DIR/data/fisher_matrix_out.dat" "$FISHER_OUT_LOCAL"

echo "Done. Fisher matrix saved to $FISHER_OUT_LOCAL"
