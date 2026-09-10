# GLM-5.3-Flash on RTX PRO 6000 (SM120)

Separate pack from [`../glm53-flash/`](../glm53-flash/) (H200). Same vLLM wheel pin;
different FlashInfer requirement, overlays, and serve flags. **Do not** mix
H200 `run.sh` onto this host, and **do not** apply this SM120 overlay on H200.

## Pin

- Commit: `9cd956c7e6cf54efa366b803cafa15ec6c2df827`
- Wheel: `vllm-0.28.1rc1.dev396+g9cd956c7e` (cu130 nightly)
- FlashInfer: `flashinfer-python==0.6.18` + matching **`flashinfer-cubin==0.6.18`**
  (cubin wheel from GitHub release — **not** on PyPI / `cu130`; PyPI stops at 0.6.13)
- CuteDSL: pin `nvidia-cutlass-dsl[cu13]==4.6.2` (do not mix 4.7.x Python with 4.6 libs)

## Local branch

- `glm53-flash/rtx-pin` @ wheel SHA + overlays:
    - [#53969](https://github.com/vllm-project/vllm/pull/53969)-style NoPE pad + effective topk
    - [#54756](https://github.com/vllm-project/vllm/pull/54756) SimpleCPU mixed page-size
    - [#54743](https://github.com/vllm-project/vllm/pull/54743) OffloadingConnector group skip

No CUDA rebuild required for the SM120 overlay (rope pad is done in Python).

## Install on RTX server

```bash
cd /path/to/glm53-flash-rtx   # or rsync this folder next to the venv
export GLM_HOME=/mnt/data/workspace/glm53

./install-nightly-wheel.sh
./install-flashinfer.sh      # must print has_flashinfer_sparse_mla_sm120 = True
./apply-sm120-overlay.sh     # SM120 + SimpleCPU + OffloadingConnector
./run.sh                     # TP=4, fp8 KV, no offload (clean boot)
# or:
./run-offload.sh             # same + SimpleCPUOffloadConnector
```

Also required at serve time:

- `CUDA_HOME` with `nvcc` on `PATH` (DeepGEMM MHC JIT)
- `--hf-overrides '{"text_config":{"index_topk":2044}}'` (in `run*.sh`; RedHatAI default is 2048 → buffer 2176 unsupported)

Expect: `FLASHINFER_MLA_SPARSE_SM120`, `world_size=4`, no `pe_dim must be 64`.

## Overlay file list (`PATCHED_FILES.txt`)

```text
vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py
vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
vllm/v1/simple_kv_offload/worker.py
vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
vllm/v1/kv_offload/base.py
vllm/v1/kv_offload/config.py
```

## H200 vs RTX (do not mix run scripts)

| | H200 `glm53-flash/` | RTX `glm53-flash-rtx/` |
| --- | --- | --- |
| TP | 2 | 4 |
| KV | BF16 (no fp8 flag) | `--kv-cache-dtype fp8` |
| Overlay | SimpleCPU / OffloadingConnector | SM120 NoPE/topk **+** same offload patches |
| FlashInfer | cubin optional if `nvcc` present | **cubin required** |
| Serve | `run.sh` (SimpleCPU) | `run.sh` (no offload) or `run-offload.sh` |

## Fallback if pad overlay is not enough

If FlashInfer still rejects query dim 512 after this overlay, apply upstream
[#55277](https://github.com/vllm-project/vllm/pull/55277) `cache_kernels.cu`
(`pe_dim == 0` native) and rebuild CUDA ops (incremental / non-precompiled).
See `VALIDATION.md`.
