# Domain language

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
