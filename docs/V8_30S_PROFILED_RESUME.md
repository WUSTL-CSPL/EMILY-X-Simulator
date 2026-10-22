# Historical machine-specific resume with gzip export and performance records

For a new machine, use [HANDOFF.md](HANDOFF.md). The commands below document the original workstation and depend on its existing reference metadata. Its shutdown arrangement is not part of the portable workflow.

On October 15, 2026, the original workstation resumed through `generate_multireceiver_v8_30s_million_profiled.py`. It reused the existing SQLite checkpoint and retained all sampling, seeds, physical settings, detection requirements and split quotas. It used the original gzip exporter, producing `inputs.jsonl.gz`, `labels.jsonl.gz` and `ground_truth.jsonl.gz` parts. The intermediate uncompressed exporter remains local and is not part of this submission.

The wrapper records `runs/multireceiver_v8_30s_million/batch_timings.jsonl`: proposal generation, cache/dispatch, result collection, encoding, validation, database transaction/commit, worker CPU time, propagation/spectrogram/detector time, peak memory and available system memory. Collection overlaps work already dispatched and includes encoding; these measurements must not all be added together as disjoint phases. Worker CPU and wall times are summed over scenes and therefore exceed parallel elapsed time.

No worker recycling or physics change was introduced: the cause of the earlier slowdown has not yet been established. New timings provide evidence if it happens again.

Resume from the package directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../.venv/bin/python -u generate_multireceiver_v8_30s_million_profiled.py \
  --config configs/multireceiver_v8_30s_million.json --resume
```

The simulation manifest remains unchanged; `profiled_gzip_export_policy.json` records the wrapper hash and current output policy, and the wrapper is copied into the campaign's `code` directory. The existing output directories must not contain incompatible uncompressed exports. The new wrapper refuses to overwrite such outputs.

The original monitor used recent entries in `batch_timings.jsonl` for estimates,
excluding the old 450-second batches. Monitoring and shutdown were separate
workstation operations, not features of the portable generator. For a new
machine, use `generate_dataset.py` and `scripts/verify_portable_output.py` as
described in [HANDOFF.md](HANDOFF.md).
