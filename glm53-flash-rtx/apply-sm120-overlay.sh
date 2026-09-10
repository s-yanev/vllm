#!/usr/bin/env bash
# Overlay onto matching nightly wheel (g9cd956c7e / 9cd956c7e6…):
# - #53969-style SM120 NoPE pad + effective topk (required for GLM on RTX)
# - #54756 SimpleCPUOffloadWorker mixed page-size registration (required for GLM)
# - #54743 OffloadingConnector scratch-group scoping (optional native path)
#
# Verify by reading installed files (no vLLM import). Importing vLLM here can
# fail on unrelated torch/torchvision mismatches and hide a successful overlay.
set -euo pipefail

GLM_HOME="${GLM_HOME:-/mnt/data/workspace/glm53}"
VENV="${GLM_HOME}/.venv"
SITE="$("${VENV}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY="${SCRIPT_DIR}/overlay"

echo "Overlaying into ${SITE}"
while IFS= read -r rel; do
  [[ -z "${rel}" ]] && continue
  [[ "${rel}" == \#* ]] && continue
  src="${OVERLAY}/${rel}"
  dst="${SITE}/${rel}"
  mkdir -p "$(dirname "${dst}")"
  cp -v "${src}" "${dst}"
done < "${SCRIPT_DIR}/PATCHED_FILES.txt"

SITE="${SITE}" "${VENV}/bin/python" - <<'PY'
from pathlib import Path
import os

site = Path(os.environ["SITE"])
sparse = (
    site / "vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py"
).read_text()
sm120 = (
    site / "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
).read_text()
worker = (site / "vllm/v1/simple_kv_offload/worker.py").read_text()
off_cfg = (
    site
    / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py"
).read_text()

assert "index_kpool" in sparse, "missing effective-topk check in sparse.py"
assert "eff_width" in sparse, "missing eff_width in sparse.py"
assert "2044" in sparse and "2048" in sparse, (
    "missing 2044 override hint / 2048 topk requirement in sparse.py"
)
assert "effective topk" in sparse, "missing effective topk marker in sparse.py"

assert "_nope_pad" in sm120, "missing _nope_pad in sm120.py"
assert "_DS_ROPE_DIM" in sm120, "missing _DS_ROPE_DIM in sm120.py"
assert "do_kv_cache_update" in sm120, "missing do_kv_cache_update in sm120.py"
assert "eff_topk" in sm120, "missing eff_topk in sm120.py"
assert "_SM120_GLM_NSA_TOPK" in sm120, "missing SM120 topk clamp in sm120.py"

assert "kv_cache_tensor.block_stride" in worker, (
    "missing #54756 block_stride in simple_kv_offload/worker.py"
)
assert "layer_outermost" in worker, (
    "missing #54756 layer_outermost in simple_kv_offload/worker.py"
)
assert "prefix_cacheable" in off_cfg, (
    "missing #54743 prefix_cacheable in offloading/config.py"
)
assert "Skipping non-prefix-cacheable" in off_cfg, (
    "missing #54743 skip log in offloading/config.py"
)

print("SM120 NoPE/topk + SimpleCPU (#54756) + OffloadingConnector (#54743): ok")
for rel in (
    "vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py",
    "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    "vllm/v1/simple_kv_offload/worker.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py",
):
    print(f"  {site / rel}")
PY
