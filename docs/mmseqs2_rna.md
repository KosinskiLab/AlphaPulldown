# Local MMseqs2 features (AlphaFold 2 and 3)

An alternative to the per-protein jackhmmer/HHblits MSA search: proteins are split into
bounded shards searched with MMseqs2, and a separate CPU stage turns each chain's
alignment into standard features — an AF3 JSON with AlphaFold 3's own template search,
or an AF2 `MonomericObject` pickle; see [AlphaFold 2](#alphafold-2) below. Off by
default. Native feature generation and the remote `--use_mmseqs2` path are untouched.

RNA chains can use the same path once the RNA databases are configured; see
[RNA chains](#rna-chains) below.

## Databases

**Protein — four, all padded.** The GPU prefilter needs `makepaddedseqdb` output, so
each configured path must name a padded database. Build them from ordinary MMseqs2
databases, keeping source and destination prefixes different:

```bash
mmseqs makepaddedseqdb /source/uniref90 /db/mmseqs/uniref90
```

**RNA — three, not padded.** AlphaFold 3 searches Rfam, RNAcentral and NT-RNA with
`nhmmer` and merges them into one unpaired MSA; this stage searches the same three in
the same order.

| identifier | AlphaFold 3 FASTA |
| --- | --- |
| `rfam` | `rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta` |
| `rnacentral` | `rnacentral_active_seq_id_90_cov_80_linclust.fasta` |
| `nt_rna` | `nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta` |

All three ship with the standard AlphaFold 3 download, so they usually only need
converting, and a nucleotide search cannot use the GPU prefilter anyway — plain
`createdb` is enough, and padding them would cost hours for something never read:

```bash
bash scripts/setup_databases.sh --dest /path/to/databases --mmseqs --mmseqs-rna
```

All three are required together. A subset is an error, not a partial opt-in: AlphaFold 3
merges all three, so a missing one silently yields a shallower alignment than the same
input run elsewhere.

The four MMseqs2 protein databases are **additive**. The complete native AlphaFold 3
database tree is still required, since finalization keeps AF3's own template search.

## How the MSAs compare to the native pipeline

MMseqs2 has been used to build AlphaFold MSAs for years — ColabFold does exactly this —
so the approach is established. It is not the *same* search as jackhmmer, and the
difference is worth seeing before switching.

Eight *B. subtilis* proteins, same four databases, counting unique sequences. The
figure is a **depth ratio**: MMseqs2 sequence count divided by the native pipeline's,
not an overlap measure.

| protein | unpaired | paired |
| --- | --- | --- |
| P0CI78 | 99.2% | 98.8% |
| O32142 | 98.5% | 99.6% |
| O30472 | 99.5% | 101.3% |
| P80870 | 86.0% | 103.2% |
| O31537 | 83.8% | 81.7% |
| O31843 | 82.8% | 87.1% |
| O07542 | 68.1% | 78.3% |
| O31580 | 53.7% | 61.5% |
| **overall** | **90.2%** | **98.0%** |

**Read this carefully.** Values above 100% are not "better than complete" — they mean
MMseqs2 returned *more* sequences than jackhmmer for that chain. And because it counts
sequences rather than comparing sets, a ratio near 100% does **not** prove the two found
the *same* sequences: an entirely different set of equal size would also score 100%. Use
the table to see where depth collapses (shallow families), not as an equivalence claim.

Template counts were identical (32 vs 32). Depth is close to complete on well-populated
families and falls off on shallow ones — the expected shape for a single-pass search
against an iterative profile search.

What none of this tells you is whether the difference costs prediction accuracy; that
needs matched inference and DockQ against experimental structures. Treat it as a reason
to spot-check your own targets, not as a verdict.

## RNA chains

An AlphaFold 3 RNA chain needs one field this stage produces:

```json
{"rna": {"id": "A", "sequence": "GGCUAUAGCUCAG...", "unpairedMsa": "..."}}
```

There is no paired MSA and no template search — AlphaFold 3 pairs protein chains by
UniProt taxon and searches templates for protein only — so the bundle's `pairedMsa` is
empty.

- **The alphabet matters.** AlphaFold 3 tokenises RNA over `A/C/G/U` and maps every
  other letter to the unknown nucleotide, so hits are transcribed to the RNA alphabet on
  the way out. Queries must be given as RNA (`U`, not `T`); a `T` is rejected rather
  than guessed at. Internally the query is respelled as DNA before `createdb`, because
  `A/C/G/U` are also valid amino-acid codes and MMseqs2 would otherwise read a U-spelled
  RNA as a protein and map every uracil to `X`.
- **RNA searches run on CPU** (`--gpu 0 --search-type 3`) regardless of
  `--mmseqs_use_gpu`, for the prefilter reason above.

FASTA entries are classified as elsewhere in AlphaPulldown: a sequence over `ACGUN`
containing `U` is RNA; an ambiguous one needs `RNA` or `protein` in its description. DNA
is not supported by this path.

## Running it directly

The workflow normally drives this, but the stages are ordinary scripts. Which database
flags are required depends on what the FASTA contains, so an RNA-only shard need not
point at the protein databases, or vice versa.

```bash
python -m alphapulldown.scripts.create_batch_msas \
  --fasta_paths complex.fasta --summary_path shard.json --msa_output_dir msas/ \
  --mmseqs_temp_dir /local-fast-scratch/mmseqs \
  --mmseqs_batch_max_sequences 64 --mmseqs_batch_max_residues 40000 \
  --mmseqs_uniref90_database_path  /db/mmseqs/uniref90_gpu  --mmseqs_uniref90_database_id  uniref90-2026-08 \
  --mmseqs_mgnify_database_path    /db/mmseqs/mgnify_gpu    --mmseqs_mgnify_database_id    mgnify-2026-08 \
  --mmseqs_small_bfd_database_path /db/mmseqs/small_bfd_gpu --mmseqs_small_bfd_database_id small-bfd-2026-08 \
  --mmseqs_uniprot_database_path   /db/mmseqs/uniprot_gpu   --mmseqs_uniprot_database_id   uniprot-2026-08 \
  --mmseqs_rfam_database_path       /db/mmseqs/rfam       --mmseqs_rfam_database_id       rfam-14.9 \
  --mmseqs_rnacentral_database_path /db/mmseqs/rnacentral --mmseqs_rnacentral_database_id rnacentral-2026-08 \
  --mmseqs_nt_rna_database_path     /db/mmseqs/nt_rna     --mmseqs_nt_rna_database_id     nt-rna-2023-02-23
```

Then `finalize_batch_features.py` as usual; it reads whatever the MSA stage produced and
needs no RNA configuration of its own.

Useful flags: `--mmseqs_rna_e_value` (default `1e-3`) and
`--mmseqs_<database>_max_sequences` (default `10000` per RNA database), both matching
AlphaFold 3's own settings.

## AlphaFold 2

The search is the same; only finalization differs. Pass `--data_pipeline alphafold2` to
`finalize_batch_features.py` and it writes `<name>.pkl` (or `.pkl.xz` with
`--compress_features`), the pickle the AlphaFold 2 backend reads:

```bash
python -m alphapulldown.scripts.finalize_batch_features \
  --data_pipeline alphafold2 --fasta_paths complex.fasta \
  --msa_input_dir msas/ --output_dir features/ --data_dir /db/alphafold2 \
  --max_template_date 2026-08-01 \
  --template_seqres_database_id pdb-seqres-2026-08 --template_mmcif_database_id pdb-mmcif-2026-08
```

Templates come from the AlphaFold 2 database tree under `--data_dir` (`pdb_seqres`,
`pdb_mmcif`), searched with hmmsearch, or with hhsearch against PDB70 under
`--use_hhsearch`. The MSA databases are the MMseqs2 ones the search stage used.

The features are built the way native AlphaFold 2 builds them from jackhmmer:

- uniref90 capped at 10 000 rows and MGnify at 501, counting the query as AlphaFold 2
  does; small BFD uncapped
- merged in AlphaFold 2's order, UniRef90, BFD, MGnify — row order matters, since it
  samples its MSA from the top
- templates searched from UniRef90 **alone**, never from the merged alignment; the
  bundle records which rows each database contributed so this is possible
- pairing features (`*_all_seq`) from the separate UniProt search, whose headers carry
  the species AlphaFold 2 pairs chains by

What differs, and why:

- **The recipe is `reduced_dbs`.** There is no BFD/UniRef30 HHblits arm, so compare
  against `--db_preset reduced_dbs`, not `full_dbs`.
- **Caps apply after cross-database deduplication**, where native AlphaFold 2 caps raw
  hits first. This only affects rows two databases both found.
- **The template profile has no insert columns.** A3M insertions are per row and not
  aligned to each other, so they cannot be turned back into Stockholm columns; the match
  columns, which decide the profile's states, are the same.
- **Accession identifiers are filled in** from the UniProt headers, where native
  features leave them empty. Nothing on this path queries UniProt over the network.

The pickles carry the same feature keys as native ones plus those two accession arrays,
and assemble into multimers alongside pickles from other sources. RNA and DNA chains are
refused: AlphaFold 2 has no MSA features for them.

### Measured against native AlphaFold 2

32 monomers spread over the quartiles of their native MSA depth, plus 12 heterodimers
released after AF2-multimer's training cutoff, featurized natively with `reduced_dbs` and
through this path, with the same databases and templates up to 2021-09-30:

| median, unless stated | native | local, GPU search | local, CPU search |
|---|---|---|---|
| MSA depth, shallowest quartile | 52 | 29 | 28 |
| MSA depth, deepest quartile | 12 247 | 11 756 | 5 181 |
| Neff, all monomers | 778 | 627 | 572 |
| templates per chain | 16 | 16 | 15 |
| AF2-multimer DockQ, top-ranked, mean of 12 | 0.59 | 0.56 | 0.56 |
| acceptable interfaces (DockQ ≥ 0.23) | 9 / 12 | 9 / 12 | 9 / 12 |

- The MSAs are shallower, most on the shallowest families, as for AlphaFold 3. A GPU
  search recovers about three quarters of native's unpaired hits.
- A CPU search, at MMseqs2's default sensitivity, finds far fewer distant hits than the
  GPU prefilter on deep families, without changing DockQ here.
- One interface (9HMX) lost about 0.3 DockQ against native; the other eleven moved by
  less than 0.1.
- Finalization is dominated by template featurization: median ~1 GB and 2 min per chain,
  but up to 19 GB and 90 min, set by which structures the templates come from rather
  than by chain length or MSA depth.

## The binary

Both maintained prediction images bundle the same pinned MMseqs2-GPU build at
`/opt/mmseqs/bin/mmseqs`, which is the only supported `binary_path` — arbitrary host
executables are not visible inside the image. `binary_id` records that exact commit;
update it only when deliberately changing the binary.

## Caching and provenance

Cache identity covers the binary, database identifiers and hit limits, E-value,
container identity and the template cutoff, so changing scientific provenance schedules
fresh outputs even with `rerun-triggers: mtime`.

Partial per-protein MSA bundles are deliberately **not** Snakemake outputs: a failed
shard loses only its completion summary, and a retry validates and reuses finished
bundles. Summaries record each bundle's size, mtime and SHA-256; DAG construction uses
the stat fields and streams the digest only when metadata changed. A missing, corrupt or
replaced bundle triggers a repair that reruns only that shard.

RNA bundles additionally record molecule type, search mode and the three RNA database
identities. A protein bundle records no molecule type at all — exactly what bundles
written before RNA support looked like — so every previously cached protein MSA still
matches.

## Tuning

Ampere or newer GPUs give full performance; Turing works at reduced speed. A database
larger than VRAM can stream from host RAM but needs enough node RAM and is slower. Put
database prefixes and `temp_dir` on fast local storage. GPU search always runs at
maximum sensitivity, so there is no `sensitivity` setting.

For repeated searches on a dedicated node, MMseqs2 recommends `createindex
--index-subset 2` with a same-node `gpuserver` and `--gpu-server 1 --db-load-mode 2`.
This adapter does not start one, because shards can land on different nodes; each shard
loads its databases once. Pinning shards to a resident service is a site-specific
optimisation.

Container binds are merged with existing `APPTAINER_BINDPATH`/`SINGULARITY_BINDPATH`
rather than replacing them, in all workflow modes including AF2.

## Caveats

- **Protein**: depth falls off on shallow families (see the table above).
- **RNA**: MMseqs2's nucleotide search is sequence-sequence, while AlphaFold 3 uses
  `nhmmer`, which is profile-based and finds remote homologues a sequence-sequence
  search does not. Close relatives look similar; divergent families will be shallower.
- **RNA, nt_rna**: MMseqs2 can crash (SIGSEGV) on particular queries against nt_rna
  while succeeding for the same query on the smaller databases and for other queries on
  nt_rna. That is a fault inside MMseqs2; the failure names the chain and says retrying
  will not help.

If MSA depth matters for your target, compare against the native pipeline before
committing to this path.
