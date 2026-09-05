# Structural templates with Foldseek and ESMFold

AlphaPulldown normally finds AlphaFold 2 templates by *sequence*: a profile built
from the uniref90 alignment is searched against `pdb_seqres` with hmmsearch (or
against PDB70 with HHsearch when `--use_hhsearch` is set), and the resulting
alignments become the `template_*` block of the features.

`--use_foldseek_templates` replaces that search with a structural one. ESMFold
predicts a structure for the query sequence, Foldseek searches that structure
against a local structure database, and each alignment it returns becomes a
template hit. It is useful when a template's sequence has drifted far enough that
a profile search no longer finds it but its fold is still recognisable.

The feature is off by default. With it off, nothing about feature generation
changes — not the search, not the features, not their metadata.

## What changes and what does not

Only the *source of the hits* changes. The hits are handed to AlphaFold 2's own
`HhsearchHitFeaturizer`, so:

- every template is still read out of `--template_mmcif_dir` as `<pdbid>.cif`;
- `--max_template_date`, the coverage and duplicate prefilters, and the cap of 20
  templates per query all still apply;
- MSA generation is untouched, so the run still needs the sequence databases.

The consequence worth planning around: **the Foldseek database must describe
chains that exist in `--template_mmcif_dir`.** A hit whose name does not reduce to
a PDB chain — an AlphaFold DB model, say — is skipped with a warning, because
there is no mmCIF file to featurise it from.

## What you have to install

None of this is installed with AlphaPulldown.

1. **Foldseek.** A release binary or a conda install; see
   <https://github.com/steineggerlab/foldseek>. Point `--foldseek_binary_path` at
   it, or leave it on `PATH`.
2. **PyTorch and transformers**, for ESMFold: `pip install torch transformers`.
3. **ESMFold weights** in a local directory — a download of the published
   `facebook/esmfold_v1` checkpoint, for example. AlphaPulldown loads it with
   `local_files_only`, so nothing is fetched at run time and a missing checkpoint
   is a local error rather than a silent multi-gigabyte download on a compute
   node.
4. **A Foldseek structure database** (below).

## Building the structure database

Build it from the same mmCIF directory the featuriser reads, and the two cannot
disagree:

```bash
foldseek createdb /path/to/pdb_mmcif/mmcif_files /path/to/foldseek/pdb
foldseek createindex /path/to/foldseek/pdb /path/to/scratch   # optional
```

Foldseek names its targets after the files it was built from, so entries come out
as `1abc.cif.gz_A`. AlphaPulldown reduces those to `1abc_A`, which is what
AlphaFold 2 needs in order to open `1abc.cif`. The common decorations
(`pdb1abc.ent.gz_A`, `1abc-assembly1.cif.gz_A`) are understood too.

## Running it

```bash
create_individual_features.py \
  --fasta_paths=proteins.fasta \
  --data_dir=$ALPHAFOLD_DATA_DIR \
  --output_dir=features \
  --max_template_date=2024-01-01 \
  --use_foldseek_templates \
  --foldseek_database_path=/path/to/foldseek/pdb \
  --foldseek_database_id=pdb-mmcif-2024-01 \
  --esmfold_model_dir=/path/to/esmfold_v1 \
  --esmfold_device=cuda
```

`--keep_msas` works with it: an existing feature set can have only its templates
replaced, using the structural search instead of the sequence one.

## Running the GPU step separately

Folding the query is the only part that wants a GPU, and it depends on nothing
but the sequence and the checkpoint. `predict_query_structures.py` does just that
step and caches the result:

```bash
predict_query_structures.py \
  --fasta_paths=proteins.fasta \
  --structural_template_cache_dir=/scratch/structural_templates \
  --esmfold_model_dir=/path/to/esmfold_v1 \
  --esmfold_device=cuda
```

Feature generation then reuses those structures, so it can run on CPU nodes.
Point it at the same `--structural_template_cache_dir` and the same
`--esmfold_model_dir`: the checkpoint directory is still read (its file sizes
identify the weights that produced a cached structure), but the model itself is
never loaded when every structure is already cached.

Running this stage is optional. Feature generation folds whatever it does not
find, so skipping it only moves the work, never breaks the run.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--use_foldseek_templates` | `False` | Turn structural template search on. |
| `--foldseek_binary_path` | `which foldseek` | The Foldseek executable. |
| `--foldseek_database_path` | — | Prefix of the Foldseek database. Required. |
| `--foldseek_database_id` | — | Immutable name of that database build. Required. |
| `--foldseek_temp_dir` | `<cache>/tmp` | Foldseek scratch directory. |
| `--foldseek_e_value` | `1e-3` | Search E-value cutoff. |
| `--foldseek_max_hits` | `100` | Structures Foldseek may return per query. |
| `--foldseek_min_alignment_tm_score` | `0.0` | Drop hits below this alignment TM-score. |
| `--foldseek_alignment_type` | `2` | `1` for TMalign, `2` for 3Di+AA. |
| `--foldseek_threads` | `8` | CPU threads for Foldseek. |
| `--structural_template_cache_dir` | `<output_dir>/structural_templates` | Where structures and alignments are cached. |
| `--esmfold_model_dir` | — | Local ESMFold checkpoint directory. Required. |
| `--esmfold_device` | `cuda` | Torch device for ESMFold. |
| `--esmfold_chunk_size` | unset | Chunk the trunk to trade speed for GPU memory. |
| `--esmfold_max_sequence_length` | `1500` | Refuse longer sequences rather than risk an out-of-memory kill. |

## Caching and provenance

Two caches live under `--structural_template_cache_dir`, keyed by the sequence:

- `<digest>_esmfold.json` — the predicted structure and the identity of the
  checkpoint that produced it.
- `<digest>_foldseek.json` — Foldseek's output and the full identity of the
  search: checkpoint, Foldseek version, database identifier and index size,
  E-value, hit cap, alignment type, TM-score threshold, and the exact columns
  requested.

A cache entry is reused only when that record matches exactly, so a rebuilt
database, a new checkpoint or a changed setting invalidates the alignments. The
two are separate on purpose: re-searching a refreshed database costs a Foldseek
run, not a GPU run.

`--foldseek_database_id` is yours to choose and should name the build, not the
path. Because an operator-supplied name cannot notice a database that was rebuilt
or half-copied under the same name, the index size is recorded alongside it as a
content-derived witness.

The raw Foldseek output is also written to `pdb_hits.m8` in the MSA output
directory, where the sequence-based search writes `pdb_hits.sto` or
`pdb_hits.hhr`.

Feature metadata records the structural-template flags only when the feature was
actually used, so a Foldseek binary that merely happens to be on `PATH` never
appears in the provenance of features it did not produce.

## Limitations

- **AlphaFold 2 features only.** AlphaFold 3 runs its own template search inside
  its data pipeline; combining the two is rejected with an error.
- **Not with `--use_mmseqs2`.** The remote MMseqs2 path receives MSAs and
  templates together and has no separable template search to replace.
- **Proteins only.**
- Multimeric templates (`--path_to_mmt`) are a separate mechanism and are
  unaffected.
- `template_sum_probs` carries the Foldseek bit score rather than an HHsearch
  probability. It is used to rank hits and is not consumed by the model.

## When something is missing

Each of these is reported as a plain message naming the fix, not a traceback:

- no Foldseek executable configured, or the configured path does not exist;
- PyTorch or transformers not installed;
- no weight files in `--esmfold_model_dir`, or a checkpoint that will not load;
- a query longer than `--esmfold_max_sequence_length`.

A Foldseek run that fails reports Foldseek's own stderr. Individual hits that
cannot be used — an unparsable row, a target with no mmCIF file, an alignment
that does not belong to this query — are dropped with a warning so that one bad
target does not cost the query its remaining templates.
