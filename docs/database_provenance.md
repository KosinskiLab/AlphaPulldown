# Database provenance

Feature metadata records the databases selected for the search mode. AF2 full
and reduced presets, HHsearch/HMMsearch, query-only features and the ColabFold
service use different resources; unused default paths are omitted.

UniProt and UniRef90 release identifiers come from the **resolved filename**
(for example `uniprot_all_2021_04.fa` or `uniref90_2022_05.fa`) or an explicit
manifest. A generic `uniprot.fasta` does not identify a release. Its `version`
and `release_date` remain null unless a manifest supplies them. Filesystem dates
are recorded separately as `file_modified_date`; copying or touching a file does
not establish its release. Symlinks are resolved before reading provenance.

## Register a new database download

Keep each downloaded release in its own immutable, versioned location. Retain
the upstream release notes and download URLs, and verify the upstream checksum
when one is available. Register only a release that those records establish:

```bash
python -m alphapulldown.scripts.create_database_metadata \
  /databases/uniprot/2024_04/uniprot.fasta \
  --database UniProt --version 2024_04 \
  --source-url 'https://your-archive.example/uniprot/2024_04/uniprot.fasta'
```

Replace the example URL with the actual source. Use `--database UniRef90` for
UniRef90. Add `--release-date YYYY-MM-DD` only when the date is known.
This command streams the FASTA once to compute SHA-256 and writes
`<resolved-fasta>.metadata.json`. Hashing a large database can take minutes;
run it in a compute allocation on shared clusters. Existing manifests are not
overwritten unless `--overwrite` is supplied.

Feature generation reads the small manifest without hashing the entire database
for every protein. It checks database name, schema, file size and nanosecond mtime.
A stale or malformed manifest, or disagreement with a versioned filename, fails
explicitly. These inexpensive freshness checks require immutable database files;
they are not a substitute for verifying the stored SHA-256 after a transfer.
The manifest's release is an operator assertion, not something inferred from
the sequence content. Do not register a guessed release for an old download.

## Reused alignments and features

Raw precomputed alignments without metadata are marked `cached_alignments` with
`unknown_source: true`. Current database paths cannot establish which release
created those alignments. Fresh searches in the same run still record their
own selected databases.

AF2 pickles carry the same metadata as their JSON sidecars. Template-only
updates retain the original MSA database entries and source date/software while
recording the new template resources. Older pickles can recover provenance from
their latest sidecar; if neither source exists, it stays unknown. AF3 template
updates similarly retain the metadata embedded in chain descriptions.
When inheriting older metadata, null-version UniProt/UniRef90 modification dates
are moved out of `release_date`, and unsourced ColabFold request-date versions
and fixed environment-database URLs are cleared. Explicitly sourced releases
remain intact; repeated template updates do not reintroduce the legacy claims.

ColabFold is included only for a service request or documented inherited
provenance. Its database release remains unknown when the service does not
report one; neither the request date nor a historical environment-database URL
identifies the service's current backend.

Historical sidecars cannot reliably be repaired from today's database directory.
Keep their original bytes and an audit trail, correct demonstrably false entries,
and leave unrecoverable release identifiers unknown. Such metadata corrections
do not require regenerating the feature tensors.
