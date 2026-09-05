# Domain glossary

- **Prediction job**: one independent fold request, including all chains that
  compose that fold and its output directory.
- **Prediction batch**: an ordered set of independent prediction jobs that share
  one homogeneous backend and model configuration.
- **Resident inference**: executing a prediction batch while keeping one set of
  model runners initialized for the lifetime of the batch.
- **Batch manifest**: a JSONL description of prediction jobs. It describes work;
  it does not change fold composition semantics.
- **Job failure**: a recoverable exception associated with one prediction job.
  It is reported after remaining jobs have been attempted.
- **Fold preparation**: building the object to model for one prediction job and its
  output directory, including AlphaPulldown-style naming and feature-metadata copying.
  Shared by the single-fold command and the resident batch so the two cannot diverge.
- **Feature request**: one named protein sequence requiring an AlphaFold 3 feature artifact.
- **Feature batch**: an ordered collection of feature requests handled as one operation.
- **MSA batch**: the GPU stage that searches MMseqs2 and durably publishes one reusable MSA bundle per feature request.
- **Feature finalization**: the CPU stage that reads an MSA bundle, performs native AF3 template search, and publishes the standard AF3 feature artifact.
- **MSA bundle**: an atomic intermediate JSON containing one sequence, merged unpaired A3M, paired A3M, and complete MMseqs/database provenance.
- **Database identifier**: the caller-supplied immutable identity of one MMseqs2 database build; cache validity depends on it, not only its filesystem path.
- **Feature artifact**: the standard AlphaFold 3 JSON (optionally LZMA-compressed) produced for one feature request.
- **MSA cache hit**: an existing MSA bundle whose sequence, MMseqs2 executable version, search settings, and database identifiers match the request.
- **Feature cache hit**: an existing feature artifact whose MSA provenance, maximum template date, PDB seqres identity, and mmCIF identity match the request.
- **Recoverable failure**: a failure isolated to one sequence; remaining requests continue and the batch reports a nonzero summary after writing successful artifacts.
- **Database role**: whether a configured MMseqs2 database supplies unpaired hits
  (uniref90, mgnify, small_bfd, merged into one MSA) or paired hits (uniprot, whose
  UniProt taxon headers let AlphaFold 3 pair chains by species). Roles are named, not
  inferred from a position in the configured list.

