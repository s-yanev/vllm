# GLM-5.3-Flash wheel pin + KV offload (H200 deliverables)

For **4× RTX PRO 6000 (SM120)** use the sibling pack
[`../glm53-flash-rtx/`](../glm53-flash-rtx/) instead. Same wheel pin; different
FlashInfer cubin requirement, NoPE/SM120 overlays, and `run.sh` (TP=4 + FP8 KV).
**Do not** mix this H200 `run.sh` onto the RTX host (wrong TP / KV dtype).
RTX offload patches live in `../glm53-flash-rtx/` and are applied with that
pack’s `./apply-sm120-overlay.sh`.

## Pin

- Commit: `9cd956c7e6cf54efa366b803cafa15ec6c2df827`
- Wheel: `vllm-0.28.1rc1.dev396+g9cd956c7e` (cu130 nightly)
- Install on server: `./install-nightly-wheel.sh`

## Local branch

- `glm53-flash/offload-pin` @ wheel SHA + patches:
    - [#54756](https://github.com/vllm-project/vllm/pull/54756) — SimpleCPU mixed page-size registration (DSA/GLM)
    - [#54743](https://github.com/vllm-project/vllm/pull/54743) — OffloadingConnector scratch-group scoping + skip log

## Required overlay for SimpleCPU (stock nightly crashes)

Unpatched `g9cd956c7e` dies in `SimpleCPUOffloadWorker.register_kv_caches` with:

```text
RuntimeError: shape '[-1, 1036, 2228224]' is invalid for input of size …
```

Copy runtime files, then restart with `run.sh`:

```bash
./apply-offloading-overlay.sh   # also installs SimpleCPU worker.py
./run.sh
```

Look for: `SimpleCPUOffloadWorker [CPU]: … CPU blocks` (not the reshape error).

## Optional OffloadingConnector

After the same overlay: `./run-offloading-connector.sh`  
Expect skip logs for `CircularBufferSpec` / `tail_cache`; no `tokens_per_block=4 not divisible`.

## Overlay file list (`PATCHED_FILES.txt`)

```text
vllm/v1/simple_kv_offload/worker.py
vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
vllm/v1/kv_offload/base.py
vllm/v1/kv_offload/config.py
```

Do not overlay onto a different `g<sha>` without rebasing.
