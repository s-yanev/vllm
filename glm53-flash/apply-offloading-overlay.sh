#!/usr/bin/env bash
# Overlay patches onto matching nightly wheel (g9cd956c7e / 9cd956c7e6…):
# - #54756 SimpleCPUOffloadWorker mixed page-size registration (required for GLM)
# - #54743 OffloadingConnector scratch-group scoping (optional native path)
set -euo pipefail

GLM_HOME="${GLM_HOME:-/mnt/data/workspace/glm53}"
VENV="${GLM_HOME}/.venv"
SITE="$("${VENV}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY="${SCRIPT_DIR}/overlay"

echo "Overlaying into ${SITE}"
while IFS= read -r rel; do
  [[ -z "${rel}" ]] && continue
  src="${OVERLAY}/${rel}"
  dst="${SITE}/${rel}"
  mkdir -p "$(dirname "${dst}")"
  cp -v "${src}" "${dst}"
done < "${SCRIPT_DIR}/PATCHED_FILES.txt"

"${VENV}/bin/python" - <<'PY'
import inspect
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)

worker_src = inspect.getsource(SimpleCPUOffloadWorker.register_kv_caches)
assert "kv_cache_tensor.block_stride" in worker_src
assert "layer_outermost" in worker_src
cfg_src = inspect.getsource(build_offloading_config)
assert "prefix_cacheable" in cfg_src
assert "Skipping non-prefix-cacheable" in cfg_src
print("SimpleCPU (#54756) + OffloadingConnector (#54743) overlay: ok")
PY
