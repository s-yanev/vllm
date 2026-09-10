#!/usr/bin/env bash
# Install FlashInfer + matching cubins for SM120 sparse MLA on RTX PRO 6000.
# Without flashinfer-cubin (and without nvcc on PATH), vLLM sets
# has_flashinfer()=False even when decode APIs import successfully.
#
# Important: flashinfer-cubin==0.6.18 is NOT on PyPI (latest there is 0.6.13)
# and NOT under https://flashinfer.ai/whl/cu130. Matching cubins ship on the
# GitHub release assets (and sometimes https://flashinfer.ai/whl).
set -euo pipefail

GLM_HOME="${GLM_HOME:-/mnt/data/workspace/glm53}"
VENV="${GLM_HOME}/.venv"
FI_INDEX="${FI_INDEX:-https://flashinfer.ai/whl/cu130}"
FI_CUBIN_INDEX="${FI_CUBIN_INDEX:-https://flashinfer.ai/whl}"
FI_VERSION="${FI_VERSION:-0.6.18}"
# Direct GitHub asset (authoritative for released cubin wheels)
FI_CUBIN_URL="${FI_CUBIN_URL:-https://github.com/flashinfer-ai/flashinfer/releases/download/v${FI_VERSION}/flashinfer_cubin-${FI_VERSION}-py3-none-any.whl}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "Installing flashinfer-python==${FI_VERSION}"
uv pip install -U "flashinfer-python==${FI_VERSION}" \
  --extra-index-url "${FI_INDEX}"

echo "Installing matching flashinfer-cubin==${FI_VERSION}"
# 1) Prefer exact GitHub release wheel (same version as python)
if uv pip install -U "${FI_CUBIN_URL}"; then
  echo "Installed flashinfer-cubin from ${FI_CUBIN_URL}"
# 2) Fallback: FlashInfer cubin index (CUDA-agnostic; not cu130)
elif uv pip install -U "flashinfer-cubin==${FI_VERSION}" \
  --index-url "${FI_CUBIN_INDEX}" \
  --extra-index-url https://pypi.org/simple; then
  echo "Installed flashinfer-cubin==${FI_VERSION} from ${FI_CUBIN_INDEX}"
else
  echo "error: could not install flashinfer-cubin==${FI_VERSION}" >&2
  echo "  Tried: ${FI_CUBIN_URL}" >&2
  echo "  Tried: ${FI_CUBIN_INDEX}" >&2
  echo "  PyPI only has older cubins (e.g. 0.6.13) which mismatch 0.6.18." >&2
  echo "  Do NOT leave flashinfer-cubin==0.6.13 installed with python 0.6.18." >&2
  exit 1
fi

# Optional JIT cache (CUDA-specific index). Exact pin may be unavailable.
uv pip install -U "flashinfer-jit-cache==${FI_VERSION}" \
  --extra-index-url "${FI_INDEX}" || \
  echo "warn: flashinfer-jit-cache==${FI_VERSION} install skipped (optional)"

# TileLang MHC (GLM mhc_pre_tilelang) breaks with apache-tvm-ffi>=0.1.12
# (__ffi_repr__ already registered). Pin a known-good version.
uv pip install -U "apache-tvm-ffi==0.1.11" || \
  echo "warn: could not pin apache-tvm-ffi==0.1.11"

python - <<'PY'
import importlib.metadata as md

import flashinfer
from flashinfer.autotuner import autotune
from flashinfer.decode import (
    trtllm_batch_decode_sparse_mla_dsv4,
    trtllm_batch_decode_with_kv_cache_mla,
)
from vllm.utils.flashinfer import (
    has_flashinfer,
    has_flashinfer_cubin,
    has_flashinfer_sparse_mla_sm120,
)

fi_ver = getattr(flashinfer, "__version__", "?")
cubin_ver = md.version("flashinfer-cubin")
print("flashinfer =", fi_ver)
print("flashinfer-cubin =", cubin_ver)
# Allow 0.6.18 vs 0.6.18.post1 style if major.minor.patch matches
fi_base = fi_ver.split("+")[0].split(".post")[0]
cubin_base = cubin_ver.split("+")[0].split(".post")[0]
assert cubin_base == fi_base or cubin_ver.startswith(fi_base), (
    f"version mismatch: flashinfer={fi_ver} cubin={cubin_ver}"
)
print("has_flashinfer_cubin() =", has_flashinfer_cubin())
print("has_flashinfer() =", has_flashinfer())
print(
    "sparse APIs =",
    callable(trtllm_batch_decode_sparse_mla_dsv4),
    callable(trtllm_batch_decode_with_kv_cache_mla),
    callable(autotune),
)
print("has_flashinfer_sparse_mla_sm120() =", has_flashinfer_sparse_mla_sm120())
assert has_flashinfer_cubin()
assert has_flashinfer()
assert has_flashinfer_sparse_mla_sm120()
print("FlashInfer SM120 sparse MLA: ok")
PY
