# Source from the PileFlow repo root.

export REPO="$(pwd)"
export PYTHONPATH="$REPO:$REPO/generator/src:${PYTHONPATH:-}"
export MG5_PATH="$(which mg5_aMC)"
