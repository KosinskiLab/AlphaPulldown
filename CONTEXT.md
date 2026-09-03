# Domain language

- **Feature request**: one named protein sequence requiring an AlphaFold 3 feature artifact.
- **Feature batch**: an ordered collection of feature requests handled as one operation.
- **Database identifier**: the caller-supplied immutable identity of one MMseqs2 database build; cache validity depends on it, not only its filesystem path.
- **Feature artifact**: the standard AlphaFold 3 JSON (optionally LZMA-compressed) produced for one feature request.
- **Cache hit**: an existing feature artifact whose sequence, MMseqs2 settings, and database identifiers match the request.
- **Recoverable failure**: a failure isolated to one sequence; remaining requests continue and the batch reports a nonzero summary after writing successful artifacts.
