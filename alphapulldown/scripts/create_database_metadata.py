"""Register a known UniProt/UniRef90 release; hash the FASTA only once."""
import argparse
from alphapulldown.utils.database_provenance import write_database_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta")
    parser.add_argument("--database", choices=("UniProt", "UniRef90"), required=True)
    parser.add_argument("--version", required=True, help="Known release ID, never a guessed file date")
    parser.add_argument("--release-date", help="Known release date in YYYY-MM-DD format")
    parser.add_argument("--source-url", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(write_database_manifest(args.fasta, args.database, args.version,
                                  release_date=args.release_date,
                                  location_url=args.source_url, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
