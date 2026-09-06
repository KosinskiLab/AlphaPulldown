# RNA MSAs through the local MMseqs2 path

The local MMseqs2 feature stage was protein-only. It can now also build the unpaired
MSA that AlphaFold 3 expects for an RNA chain, so a fold containing RNA no longer has
to fall back to the native AlphaFold 3 data pipeline for that chain.

RNA is **off** unless you configure the three RNA databases. A run that configures
none of them behaves exactly as it did before, down to the cache signature written
into every MSA bundle, so nothing already cached is invalidated.

## What AlphaFold 3 expects for RNA

An AlphaFold 3 RNA chain carries one field this stage has to produce:

```json
{"rna": {"id": "A", "sequence": "GGCUAUAGCUCAG...", "unpairedMsa": "..."}}
```

There is no paired MSA and no template search: AlphaFold 3 pairs protein chains by
UniProt taxon and searches templates for protein only. RNA gets a single unpaired A3M
and nothing else. The bundle this stage writes therefore has an empty `pairedMsa`.

Two consequences worth knowing:

- **The alphabet matters.** AlphaFold 3 tokenises an RNA MSA over `A/C/G/U` and maps
  every other letter to the unknown nucleotide. A hit spelled with `T` would reach the
  model as a row of unknowns, so nucleotide hits are transcribed to the RNA alphabet
  on the way out. Query sequences must be given as RNA (`U`, not `T`); a `T` in an
  input RNA sequence is rejected rather than guessed at.
- **RNA searches run on CPU.** MMseqs2's GPU prefilter needs a padded protein
  database, so the nucleotide searches are issued with `--gpu 0 --search-type 3`
  regardless of `--mmseqs_use_gpu`. The RNA databases are correspondingly built with
  plain `createdb` and no `makepaddedseqdb`.

## Databases

AlphaFold 3 searches three RNA databases with `nhmmer` and merges them into one
unpaired MSA, in this order: Rfam, RNAcentral, NT-RNA. This stage searches the same
three, in the same order, so the merged MSA has the same composition.

| identifier | AlphaFold 3 FASTA |
| --- | --- |
| `rfam` | `rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta` |
| `rnacentral` | `rnacentral_active_seq_id_90_cov_80_linclust.fasta` |
| `nt_rna` | `nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta` |

All three come with the standard AlphaFold 3 database download, so if you already ran
`fetch_databases.sh` (or `scripts/setup_databases.sh --alphafold3`) the FASTAs are on
disk already and only need converting:

```bash
bash scripts/setup_databases.sh --dest /path/to/databases --mmseqs-rna
```

All three are required together. Configuring a subset is an error rather than a
partial opt-in, because AlphaFold 3 merges all three: a subset would quietly produce a
shallower alignment than the same input run elsewhere.

## Running it

Add the RNA flags to the ordinary MSA stage invocation. Everything else - the batch
limits, the scratch directory, the binary - is shared with the protein search.

```bash
python -m alphapulldown.scripts.create_batch_msas \
  --fasta_paths complex.fasta \
  --summary_path shard.json \
  --msa_output_dir msas/ \
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

Then `finalize_batch_features.py` as usual; it reads whatever the MSA stage produced
and needs no RNA configuration of its own.

FASTA entries are classified the way the rest of AlphaPulldown classifies them: a
sequence over `ACGUN` containing `U` is RNA, and an ambiguous one needs `RNA` or
`protein` in its description. DNA is still not supported by this path.

Which database flags are required now depends on what is in the FASTA, so an RNA-only
shard does not have to point at the protein databases, and a protein-only shard does
not have to point at the RNA ones.

Additional flags:

- `--mmseqs_rna_e_value` (default `1e-3`, matching AlphaFold 3's own RNA searches).
- `--mmseqs_<database>_max_sequences` (default `10000` per RNA database, again
  matching AlphaFold 3).

## Caching

RNA bundles carry their own provenance: molecule type, MMseqs2 version, search mode,
E-value and the identity of all three RNA databases. Changing any of them regenerates
the RNA MSA and leaves protein bundles alone, and vice versa. The same letters
submitted once as protein and once as RNA are two different searches with two
different bundles.

A protein bundle records no molecule type at all, which is exactly what bundles
written before RNA support looked like, so every cached protein MSA still matches.

## Caveat

MMseqs2's nucleotide search is a sequence-sequence search. AlphaFold 3 builds its RNA
MSAs with `nhmmer`, which is profile-based and finds remote RNA homologues that a
sequence-sequence search does not. For an RNA family with close relatives in the
databases the two will look similar; for a divergent one the MMseqs2 MSA will be
shallower. If RNA MSA depth matters for your target, compare against the native
AlphaFold 3 pipeline before committing to this path.
