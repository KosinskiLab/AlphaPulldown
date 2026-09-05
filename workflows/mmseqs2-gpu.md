# Experimental local MMseqs2-GPU feature generation

## Status and scope

This backend is experimental. It replaces AF3's protein MSA searches, so it may
change MSA depth, template hits, and prediction accuracy. The command-contract
tests prove that AlphaPulldown can consume real MMseqs2 output; they do **not**
establish scientific equivalence to jackhmmer. A representative full-database
MSA-depth/template-hit and downstream DockQ comparison remains a merge gate.

The interface is AF3-protein-only. AlphaPulldown's AF2 and AF3 images both carry
the same pinned binary so cluster images remain symmetric and an AF2 adapter can
be added without rebuilding the runtime; AF2 feature generation does not consume
it yet.

## Two-stage interface

The workflow deliberately releases its GPU before AF3 template search:

1. `create_batch_msas.py` searches the four MMseqs2 databases and atomically
   writes one durable `<name>_mmseqs_msa.json` bundle per protein. Its
   `--summary_path` is published only after the entire shard succeeds. If a
   shard fails, completed bundles remain valid and are reused by its retry.
2. `finalize_batch_features.py` reads those bundles, runs the native AF3 CPU
   template and feature pipeline, and writes standard
   `<name>_af3_input.json[.xz]` artifacts. These jobs can run independently and
   in parallel without reserving a GPU.
3. `create_batch_features.py` composes both stages for direct compatibility,
   but holds one allocation throughout and is not recommended on a scheduler.

The GPU entry point does not import AlphaFold or JAX. The core seams are
`MsaBatch.generate(requests)` and `FeatureFinalizer.generate(requests)`.

## Database prerequisites

The paths passed as `mmseqs_*_database_path` must be local MMseqs2 databases
prepared for GPU search. Build each target from its source FASTA, for example:

```bash
mmseqs createdb uniref90.fasta uniref90
mmseqs makepaddedseqdb uniref90 uniref90_gpu
```

Repeat this for UniRef90, MGnify, small BFD, and UniProt, then configure the
`*_gpu` prefixes. Give every build an immutable caller-managed identifier; the
identifier, MMseqs binary identity, E-value, hit cap, and execution mode are in
the MSA cache provenance.

All native AF3 databases are still required. MMseqs2 replaces only the four
protein MSA searches. AF3 still resolves PDB seqres/mmCIF for templates and its
RNA databases from `--data_dir`. Final artifact provenance therefore also
includes `--max_template_date`, `--template_seqres_database_id`, and
`--template_mmcif_database_id`.

GPU database memory can be substantial. Prefer fast node-local storage and an
Ampere-or-newer GPU. A database larger than VRAM can stream from host RAM, but
requires enough host memory and runs below peak throughput.

For repeated searches on one persistent node, MMseqs2's advanced `gpuserver`
mode can avoid repeated database upload. It additionally requires an index made
with `createindex --index-subset 2`, matching `--max-seqs` and prefilter options
between server and searches, `--gpu-server 1`, and `--db-load-mode 2`. Ordinary
Slurm shards cannot assume that a daemon remains on the same node, so the
workflow currently uses standalone searches.

MMseqs2 GPU search always uses maximum sensitivity; the CPU `-s` option is
ignored by its GPU prefilter. AlphaPulldown therefore exposes no sensitivity
knob and does not include one in cache provenance.

## MSA and template behavior

For each chunk, one query database is reused for UniRef90, MGnify, small BFD,
and paired UniProt searches. Aligned FASTA is converted to A3M while retaining
insertions and complete UniProt descriptions, including taxonomy metadata.
Unpaired hits from all three databases are merged and deduplicated.

The finalizer passes that merged unpaired MSA to native AF3 with `templates`
unset. This matches AF3's own pipeline: it merges UniRef90, small-BFD, and MGnify
before using the resulting unpaired MSA for template search.

Missing `unpackdb` output is a hard per-shard failure. A query-only alignment is
never fabricated or cached.

## Validation

Unit tests exercise the public stage seams, including all three unpaired hit
sources, missing unpack output, provenance invalidation, and partial results.
The opt-in real command test can be run with:

```bash
MMSEQS_INTEGRATION_BINARY=/path/to/mmseqs \
pytest -o addopts="-ra --strict-markers" \
  test/integration/test_mmseqs2_command_contract.py
```

It creates a tiny database, pads it, and checks the complete
`createdb -> search -> result2msa -> unpackdb` filename and content contract in
CPU contract mode. GPU validation requires a compatible allocated GPU and is
kept out of ordinary CI; add `MMSEQS_INTEGRATION_GPU=1` to run the same test
with `search --gpu 1`.

For a representative benchmark, generate matched native and MMseqs2 AF3 feature
directories and run:

```bash
compare_msa_backends.py \
  --reference_dir=/path/to/native_features \
  --candidate_dir=/path/to/mmseqs_features \
  --output_path=msa_comparison.json
```

The report records raw/unique MSA depth, mean non-gap coverage, template counts,
and exact paired artifact paths. Use those same inputs for matched inference and
external DockQ evaluation against experimental references; the report explicitly
does not treat MSA statistics as an accuracy result.
