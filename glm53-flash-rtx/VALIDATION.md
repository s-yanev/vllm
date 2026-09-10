# Server validation checklist (RTX PRO 6000 / SM120)

Default home: `GLM_HOME=/mnt/data/workspace/glm53`

## 1. Wheel pin

```bash
./install-nightly-wheel.sh
# expect: 0.28.1rc1.dev396+g9cd956c7e
```

## 2. FlashInfer + cubin

```bash
./install-flashinfer.sh
# expect:
#   flashinfer = 0.6.18
#   flashinfer-cubin = 0.6.18   # must match; 0.6.13 + 0.6.18 fails on import
#   has_flashinfer_cubin() = True
#   has_flashinfer() = True
#   has_flashinfer_sparse_mla_sm120() = True
```

If you already have a mismatched cubin (e.g. PyPI 0.6.13 with python 0.6.18):

```bash
# Matching 0.6.18 cubin is on the GitHub release, not PyPI / cu130
uv pip install -U 'flashinfer-python==0.6.18'
uv pip install -U \
  'https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18/flashinfer_cubin-0.6.18-py3-none-any.whl'
# then re-run ./install-flashinfer.sh
```

If cubin install fails but CUDA is installed:

```bash
# Prefer a toolkit that actually has nvcc on this host (often 13.1, not 13.2):
export CUDA_HOME=/usr/local/cuda-13.1   # or /usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
which nvcc   # must print a path
# then re-run the python asserts from install-flashinfer.sh
```

## 3. SM120 + offload overlay

```bash
./apply-sm120-overlay.sh
# expect: SM120 NoPE/topk + SimpleCPU (#54756) + OffloadingConnector (#54743): ok
```

If you already copied the overlay and only the *import-based* verify failed
(`torchvision::nms does not exist`), the files are likely already in place.
Confirm without importing vLLM:

```bash
rg -n "_nope_pad|eff_topk|index_kpool|eff_width|block_stride|prefix_cacheable" \
  "$GLM_HOME/.venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse"*.py \
  "$GLM_HOME/.venv/lib/python3.12/site-packages/vllm/v1/simple_kv_offload/worker.py" \
  "$GLM_HOME/.venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py"
```

Then re-run the updated `./apply-sm120-overlay.sh` (file-content asserts).

If `vllm serve` later hits the same `torchvision::nms` error, the venv has a
**torch / torchvision ABI mismatch** (common after FlashInfer / wheel upgrades).
GLM text serve does not need torchvision — either uninstall it or reinstall a
build that matches torch:

```bash
# Diagnose
python -c "import torch; print('torch', torch.__version__, torch.__file__)"
uv pip show torch torchvision | rg '^(Name|Version|Location):'

# Fastest for text-only GLM (recommended):
uv pip uninstall -y torchvision

# Or reinstall a matching wheel (keeps vision deps):
uv pip install -U --force-reinstall torchvision --torch-backend=auto
```

Then retry `./run.sh`. Confirm import works:

```bash
python -c "from vllm.engine.arg_utils import EngineArgs; print('vllm import ok')"
```

### TileLang MHC / `apache-tvm-ffi` (`__ffi_repr__` already registered)

If model loads and attention selects `FLASHINFER_MLA_SPARSE_SM120`, but
`profile_run` dies in `mhc_pre_tilelang` with:

```text
TypeAttr `__ffi_repr__` is already registered for type index 132
```

pin `apache-tvm-ffi` below 0.1.12 (known TileLang conflict):

```bash
uv pip install 'apache-tvm-ffi==0.1.11'
# or: uv pip install 'apache-tvm-ffi==0.1.10'
python -c "import importlib.metadata as m; print(m.version('apache-tvm-ffi'))"
```

Then restart `./run.sh`.

### DeepGEMM needs `nvcc` (`std::filesystem::exists(nvcc_path)`)

If MHC profile fails with:

```text
Assertion error (.../jit/compiler.hpp): std::filesystem::exists(nvcc_path)
```

DeepGEMM is JIT-compiling `tf32_hc_prenorm_gemm` and cannot find the CUDA
compiler. `flashinfer-cubin` does **not** cover this. Put the toolkit on PATH:

```bash
ls /usr/local/cuda*/bin/nvcc
export CUDA_HOME=/usr/local/cuda-13.1   # or /usr/local/cuda — not a missing 13.2 path
export PATH="${CUDA_HOME}/bin:${PATH}"
which nvcc && nvcc --version
./run.sh   # refuses to start if nvcc is still missing
```

If the host has no CUDA toolkit (runtime-only drivers), install the matching
CUDA 13.x toolkit / `cuda-nvcc` package for your distro.

### FlashInfer rejects `topk=2176` (RedHatAI default)

```text
Unsupported sparse-MLA prefill configuration: model=GLM_NSA ... topk=2176
```

SM120 GLM_NSA kernels only ship for `topk=2048`. RedHatAI’s
`index_topk=2048` + `index_kpool=4` pads the index buffer to 2176. Force the
official width (`2044` → pads to 2048):

```bash
# in run.sh / vllm serve:
--hf-overrides '{"text_config":{"index_topk":2044}}'
```

Re-apply the SM120 overlay (clamp + clearer support-check message), then restart.

## 4. Serve (primary — no offload)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
./run.sh
```

Boot checks:

- `world_size=4` / TP ranks 0–3
- attention backend ≈ `FLASHINFER_MLA_SPARSE_SM120`
- must **not** see: `pe_dim must be 64 for fp8_ds_mla`
- must **not** see: `Unsupported sparse-MLA prefill configuration: ... topk=2176`
  (RedHatAI defaults to `index_topk=2048` → buffer 2176; FlashInfer SM120
  only instantiates `topk=2048`. Use `--hf-overrides '{"text_config":{"index_topk":2044}}'`)
- must **not** see FlashInfer shape errors with topk / block_tables ≈ 2176
- after override, effective buffer width should be 2048

Smoke:

```bash
curl -s http://127.0.0.1:8101/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}'
```

## 5. SimpleCPU offload (after Step 4 works)

```bash
# stop the no-offload server first, then:
CPU_GIB_PER_RANK=100 ./run-offload.sh
# expect: SimpleCPUOffloadWorker [CPU]: … CPU blocks
# must NOT see: shape '[-1, N, block_bytes]' is invalid
```

## 6. Optional native OffloadingConnector

```bash
KV_OFFLOAD_GIB=400 ./run-offloading-connector.sh
# expect: Skipping non-prefix-cacheable KV cache group … CircularBufferSpec
# must NOT see: tokens_per_block=4 not divisible by tokens_per_hash
```

## Fallback: CUDA pe_dim=0 (#55277)

Only if Python rope-pad is insufficient (decode still rejects NoPE geometry):

```bash
# on a checkout at 9cd956c7e (or glm53-flash/rtx-pin):
git fetch upstream pull/55277/head
git cherry-pick 0fa2b3308c2292b9e4d07a8f7bca17f0b2e6067d   # cache_kernels.cu
# then rebuild vLLM CUDA ops (VLLM_USE_PRECOMPILED=0 / incremental build)
# and reinstall into the venv; re-apply Python overlay afterward if needed
```
