#!/usr/bin/env bash
# Pin GLM-5.3-Flash-capable vLLM nightly (cu130) on the serve host.
# Target commit: 9cd956c7e6cf54efa366b803cafa15ec6c2df827
# Wheel: vllm-0.28.1rc1.dev396+g9cd956c7e
set -euo pipefail

GLM_HOME="${GLM_HOME:-/mnt/data/workspace/glm53}"
VENV="${GLM_HOME}/.venv"
PIN_SHA="9cd956c7e6cf54efa366b803cafa15ec6c2df827"
WHEEL_URL="https://wheels.vllm.ai/${PIN_SHA}/vllm-0.28.1rc1.dev396%2Bg9cd956c7e-cp38-abi3-manylinux_2_28_x86_64.whl"

cd "$GLM_HOME"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "Installing pinned vLLM wheel from ${WHEEL_URL}"
uv pip install --upgrade "${WHEEL_URL}" --torch-backend=auto

python - <<'PY'
import vllm
print("vllm.__version__ =", vllm.__version__)
from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration  # noqa: F401
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: F401
    SimpleCPUOffloadConnector,
)
print("glm5next + SimpleCPUOffloadConnector imports: ok")
PY
