# Batched local MMseqs2-GPU feature generation

## Goal

Generate standard per-protein AlphaFold 3 JSON feature artifacts while amortising local MMseqs2-GPU searches across unique protein sequences.

## Confirmed public seam

`FeatureBatch.generate(requests)` is the behavioral test seam.

- A `FeatureRequest` contains a unique output name and one protein sequence.
- The batch is configured with a GPU-capable MMseqs2 process adapter plus explicit UniRef90, MGnify, small-BFD, and paired UniProt GPU-compatible database paths and caller-supplied database identifiers. AlphaPulldown's AF2 and AF3 images bundle the same verified MMseqs2-GPU release, while this feature interface remains AF3-protein-only.
- The batch also receives the MMseqs2 settings, sequence-count and total-residue chunk limits, output directory, temporary directory, compression choice, native AlphaFold 3 data pipeline, and an MMseqs process adapter.
- The result identifies written artifacts, reused artifacts, and per-request failures. If any request failed, the caller raises a single nonzero summary after all recoverable work finishes.
- The observable outputs are standard `<name>_af3_input.json[.xz]` artifacts. MMseqs2 provenance is embedded in each artifact.

The command-line script and workflow are adapters to this seam. Existing AlphaFold 2, remote-MMseqs2, and native AlphaFold 3 paths retain their current behavior.

## Required behavior

1. Validate requests before launching MMseqs2. Only protein sequences are accepted; output names must be unique.
2. Reuse an existing artifact only when its sequence, MMseqs2 executable version, settings, and every database identifier match. A stale or unreadable artifact is regenerated.
3. Reuse a valid cached sequence across another missing request with the same sequence.
4. Deduplicate remaining work by sequence content and preserve first-seen order.
5. Pack unique sequences without exceeding either configured sequence count or configured total residues. A single sequence larger than the residue limit runs alone.
6. For each chunk, create one MMseqs2 query database and reuse it for all four searches.
7. Search UniRef90, MGnify, and small BFD for the unpaired MSA; search UniProt for the paired MSA. One GPU is used. Database discovery is forbidden.
8. Request aligned FASTA from MMseqs2 and convert query-gap columns to valid A3M insertions. Merge the unpaired alignments without duplicate rows, retain the query as the first row, and preserve complete UniProt descriptions (including taxon metadata) for species-aware pairing.
9. Pass both MSAs to the native AlphaFold 3 data pipeline with templates unset so its existing template search remains authoritative.
10. Persist each completed artifact atomically. A killed or failed write must not leave a cacheable partial artifact.
11. Continue after failures that are isolated to one sequence. Requests sharing a failed sequence fail together; unrelated sequences continue. Batch-wide MMseqs process failures fail only the affected chunk, then later chunks continue.

## Acceptance criteria

- Duplicate sequences are searched once and create separate standard artifacts.
- Count and residue limits produce deterministic chunks.
- Matching artifacts avoid MMseqs2; changed settings or database identifiers force regeneration.
- Unpaired and paired MSAs reach the native AlphaFold 3 pipeline and its resulting templates are preserved.
- Successful artifacts survive another request's failure, while the result reports the failed request.
- Compression and uncompressed output names remain compatible with existing AlphaPulldown consumers.
- Focused tests need no GPU or MMseqs2 installation; only the true external MMseqs process seam uses a fake adapter.

## Completion checks

- Core behavioral tests pass at the `FeatureBatch.generate` seam.
- Existing feature-generation regression tests pass.
- The Snakemake adapter requests one GPU only when local MMseqs2-GPU is enabled and supplies explicit batching/database settings.
- Snakemake helper and dry-run tests pass.
