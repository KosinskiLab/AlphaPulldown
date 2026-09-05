# Resident inference workflow

## Trigger

Run `run_structure_prediction_batch.py --manifest <jobs.jsonl>` when several
independent folds share one backend and model configuration. The existing
`run_structure_prediction.py` command remains the entry point for its current
single-invocation semantics.

## Public seam

`PredictionBatch` is the shared interface used by both commands. A
`PredictionJob` contains only `job_id`, `input`, and `output_directory`.
`PredictionBatch.from_jsonl(path)` loads one job per non-empty line; relative
filesystem paths are resolved from the manifest's parent directory.

The execution interface accepts the AlphaPulldown prediction adapter and returns
a summary containing completed job IDs and failures. The adapter owns backend
specific parsing and prediction; the batch owns validation, one-time setup,
job isolation, and the final status.

## Manifest contract

Each line is one JSON object:

```json
{"job_id":"A_and_B","input":"A+B","output_directory":"predictions/A_and_B"}
```

- All three fields are required non-empty strings; unknown fields are rejected.
- `job_id` and resolved `output_directory` must each be unique.
- Jobs are independent. Delimited chains inside one `input` still compose one
  fold according to the unchanged `alphapulldown-input-parser` behavior.
- Every job uses the same backend and model configuration. A heterogeneous batch
  is rejected rather than silently changing runners.
- Manifest validation completes before model setup starts.

## Run behavior

1. Validate the complete manifest.
2. Select one backend adapter and initialize its runners exactly once.
3. Parse and execute jobs in manifest order, one at a time.
4. Drop references to completed job-local inputs before preparing the next one.
   Model runners and JAX/XLA allocator state remain resident; this does not imply
   that host or device memory is returned to the operating system.
5. Record ordinary Python per-job exceptions and continue. Setup failures and
   process interrupts remain fatal. Native CUDA/XLA aborts, process termination,
   or a backend left unusable after an error cannot be isolated: they may stop the
   batch or cause its remaining jobs to fail.
6. Print one final summary for completed execution and rejected manifests. Native
   process termination cannot produce a Python-level summary. Exit nonzero after
   the summary if any job failed or the manifest was rejected.

The existing command preserves its current `--input` and `--output_directory`
semantics while delegating runner lifecycle and execution to the same module.

## Acceptance criteria

- A two-job AF2 or AF3 batch calls backend setup once and predicts both folds
  independently into their requested directories.
- A failure in the first job does not prevent the second job from running, and
  the final status is nonzero with the failed `job_id` in the summary.
- Duplicate IDs or output directories fail before backend setup.
- Relative output directories resolve against the manifest directory.
- Multiple chains in one job remain one complex; two manifest lines never merge.
- Existing command behavior and focused AF2/AF3 tests remain green.
- No GPU is needed for interface tests; only the heavy backend adapter is faked.

## Workflow integration

AlphaPulldownSnakemake writes one JSONL manifest for each existing size-binned
batch. `batch_size > 1` invokes the batch command once inside one Slurm
allocation; `batch_size <= 1` keeps the existing command for compatibility with
older containers. Existing token sorting, token caps, largest-fold memory sizing,
count-scaled runtime, retry behavior, and the batch sentinel are unchanged.
