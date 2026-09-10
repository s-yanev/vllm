# Server validation checklist

## 1. Wheel pin

```bash
cd /mnt/data/workspace/glm53
./install-nightly-wheel.sh
# expect: 0.28.1rc1.dev396+g9cd956c7e
```

## 2. Apply overlay (required for SimpleCPU on GLM)

Stock nightly hits:
`shape '[-1, N, block_bytes]' is invalid` in `simple_kv_offload/worker.py`.

```bash
./apply-offloading-overlay.sh
```

## 3. Primary: SimpleCPUOffloadConnector

```bash
./run.sh
# boot past KV init; logs: SimpleCPUOffloadWorker [CPU]: … CPU blocks
# cold long prompt → second shared-prefix request faster than full prefill
```

## 4. Optional: OffloadingConnector

```bash
./run-offloading-connector.sh
# "Skipping non-prefix-cacheable KV cache group … CircularBufferSpec"
# must NOT see: tokens_per_block=4 not divisible by tokens_per_hash
```
