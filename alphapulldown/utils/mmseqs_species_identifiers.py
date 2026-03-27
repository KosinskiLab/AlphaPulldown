"""Helpers for recovering species identifiers from mmseqs-derived A3Ms."""

from __future__ import annotations

import json
import re
from typing import Callable, Iterable, Sequence
from urllib import error
from urllib import parse
from urllib import request

from absl import logging
from alphafold.data import parsers
import numpy as np

_UNIPROT_HEADER_PATTERN = re.compile(
    r"""
    ^
    (?:tr|sp)
    \|
    (?P<accession>[A-Za-z0-9]{6,10})
    (?:_\d)?
    \|
    (?:[A-Za-z0-9]+)
    _
    (?P<species>[A-Za-z0-9]{1,5})
    (?:_\d+)?
    $
    """,
    re.VERBOSE,
)
_UNIREF_HEADER_PATTERN = re.compile(
    r'^UniRef\d+_(?P<accession>[A-Za-z0-9]+)$'
)
_UNIPARC_HEADER_PATTERN = re.compile(r'^(?P<accession>UPI[A-Z0-9]+)$')
_GENERIC_ACCESSION_PATTERN = re.compile(
    r'^(?P<accession>[A-Za-z0-9]{6,16})$'
)

_UNIPROT_BATCH_SIZE = 32
_UNIPARC_BATCH_SIZE = 16
_UNIPROT_TIMEOUT_SECONDS = 30
_SPECIES_ID_CACHE: dict[str, str] = {}


def _extract_sequence_identifier(description: str) -> str:
    split_description = description.split()
    if not split_description:
        return ""
    return split_description[0].partition('/')[0]


def _extract_accession_and_species(description: str) -> tuple[str, str]:
    sequence_identifier = _extract_sequence_identifier(description)
    if not sequence_identifier:
        return "", ""

    matches = _UNIPROT_HEADER_PATTERN.search(sequence_identifier.strip())
    if matches:
        return matches.group("accession"), matches.group("species")

    for pattern in (
        _UNIREF_HEADER_PATTERN,
        _UNIPARC_HEADER_PATTERN,
        _GENERIC_ACCESSION_PATTERN,
    ):
        matches = pattern.search(sequence_identifier.strip())
        if matches:
            return matches.group("accession"), ""

    return "", ""


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _is_transport_error(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, ConnectionError, OSError, error.URLError)) and not isinstance(
        exc, error.HTTPError
    )


def _query_uniprot_batch(
    accessions: Sequence[str],
    *,
    urlopen: Callable[..., object],
) -> dict[str, object]:
    query = " OR ".join(f"accession:{accession}" for accession in accessions)
    url = (
        "https://rest.uniprot.org/uniprotkb/search?query="
        f"{parse.quote(query)}"
        "&fields=accession,organism_id"
        "&format=json"
        f"&size={len(accessions)}"
    )
    with urlopen(url, timeout=_UNIPROT_TIMEOUT_SECONDS) as response:
        return json.load(response)


def _query_uniparc_batch(
    accessions: Sequence[str],
    *,
    urlopen: Callable[..., object],
) -> dict[str, object]:
    query = " OR ".join(f"upi:{accession}" for accession in accessions)
    url = (
        "https://rest.uniprot.org/uniparc/search?query="
        f"{parse.quote(query)}"
        "&fields=upi,organism_id"
        "&format=json"
        f"&size={len(accessions)}"
    )
    with urlopen(url, timeout=_UNIPROT_TIMEOUT_SECONDS) as response:
        return json.load(response)


def _query_uniprot_species_ids(
    accessions: Sequence[str],
    *,
    urlopen: Callable[..., object],
) -> tuple[dict[str, str], set[str]]:
    resolved: dict[str, str] = {}
    cacheable_misses: set[str] = set()
    for batch in _batched(sorted(set(accessions)), _UNIPROT_BATCH_SIZE):
        try:
            payload = _query_uniprot_batch(batch, urlopen=urlopen)
            cacheable_misses.update(batch)
        except Exception as exc:  # pragma: no cover - best-effort network fallback
            logging.warning(
                "Unable to resolve UniProtKB taxonomy for %d accessions: %s",
                len(batch),
                exc,
            )
            if _is_transport_error(exc):
                continue
            payload = {"results": []}
            for accession in batch:
                try:
                    single_payload = _query_uniprot_batch([accession], urlopen=urlopen)
                except Exception as single_exc:
                    if _is_transport_error(single_exc):
                        continue
                    continue
                cacheable_misses.add(accession)
                payload["results"].extend(single_payload["results"])
        for result in payload.get("results", []):
            accession = result.get("primaryAccession")
            taxon_id = result.get("organism", {}).get("taxonId")
            if accession and taxon_id is not None:
                resolved[accession] = str(taxon_id)
    return resolved, cacheable_misses


def _query_uniparc_species_ids(
    accessions: Sequence[str],
    *,
    urlopen: Callable[..., object],
) -> tuple[dict[str, str], set[str]]:
    resolved: dict[str, str] = {}
    cacheable_misses: set[str] = set()
    for batch in _batched(sorted(set(accessions)), _UNIPARC_BATCH_SIZE):
        try:
            payload = _query_uniparc_batch(batch, urlopen=urlopen)
            cacheable_misses.update(batch)
        except Exception as exc:  # pragma: no cover - best-effort network fallback
            logging.warning(
                "Unable to resolve UniParc taxonomy for %d accessions: %s",
                len(batch),
                exc,
            )
            if _is_transport_error(exc):
                continue
            payload = {"results": []}
            for accession in batch:
                try:
                    single_payload = _query_uniparc_batch([accession], urlopen=urlopen)
                except Exception as single_exc:
                    if _is_transport_error(single_exc):
                        continue
                    continue
                cacheable_misses.add(accession)
                payload["results"].extend(single_payload["results"])
        for result in payload.get("results", []):
            accession = result.get("uniParcId")
            organisms = result.get("organisms", [])
            taxon_ids = {
                str(organism.get("taxonId"))
                for organism in organisms
                if organism.get("taxonId") is not None
            }
            if accession and len(taxon_ids) == 1:
                resolved[accession] = next(iter(taxon_ids))
    return resolved, cacheable_misses


def resolve_species_ids_by_accession(
    accessions: Sequence[str],
    *,
    urlopen: Callable[..., object] = request.urlopen,
) -> dict[str, str]:
    unresolved = [
        accession
        for accession in sorted(set(accessions))
        if accession and accession not in _SPECIES_ID_CACHE
    ]
    if unresolved:
        uniprot_accessions = [
            accession for accession in unresolved if not accession.startswith("UPI")
        ]
        uniparc_accessions = [
            accession for accession in unresolved if accession.startswith("UPI")
        ]
        resolved, cacheable_misses = _query_uniprot_species_ids(
            uniprot_accessions, urlopen=urlopen
        )
        uniparc_resolved, uniparc_cacheable_misses = _query_uniparc_species_ids(
            uniparc_accessions, urlopen=urlopen
        )
        resolved.update(uniparc_resolved)
        cacheable_misses.update(uniparc_cacheable_misses)
        for accession, species_id in resolved.items():
            _SPECIES_ID_CACHE[accession] = species_id
        for accession in cacheable_misses:
            _SPECIES_ID_CACHE.setdefault(accession, "")

    return {
        accession: _SPECIES_ID_CACHE.get(accession, "")
        for accession in accessions
        if accession
    }


def build_mmseq_identifier_features(
    a3m_string: str,
    *,
    species_resolver: Callable[[Sequence[str]], dict[str, str]] = (
        resolve_species_ids_by_accession
    ),
) -> dict[str, np.ndarray]:
    msa = parsers.parse_a3m(a3m_string)
    seen_sequences: set[str] = set()
    accessions: list[str] = []
    species_ids: list[str] = []

    for sequence, description in zip(
        msa.sequences, msa.descriptions, strict=True
    ):
        if sequence in seen_sequences:
            continue
        seen_sequences.add(sequence)
        accession_id, species_id = _extract_accession_and_species(description)
        accessions.append(accession_id)
        species_ids.append(species_id)

    resolved_species_ids = species_resolver(
        [accession for accession, species_id in zip(accessions, species_ids, strict=True)
         if accession and not species_id]
    )
    species_ids = [
        species_id or resolved_species_ids.get(accession_id, "")
        for accession_id, species_id in zip(accessions, species_ids, strict=True)
    ]

    return {
        "msa_species_identifiers": np.array(
            [species_id.encode("utf-8") for species_id in species_ids],
            dtype=np.object_,
        ),
        "msa_uniprot_accession_identifiers": np.array(
            [accession_id.encode("utf-8") for accession_id in accessions],
            dtype=np.object_,
        ),
    }


def enrich_mmseq_feature_dict_with_identifiers(
    feature_dict: dict[str, np.ndarray],
    a3m_string: str,
    *,
    species_resolver: Callable[[Sequence[str]], dict[str, str]] = (
        resolve_species_ids_by_accession
    ),
) -> None:
    identifier_features = build_mmseq_identifier_features(
        a3m_string,
        species_resolver=species_resolver,
    )
    msa = feature_dict.get("msa")
    if msa is None:
        return
    if len(identifier_features["msa_species_identifiers"]) != msa.shape[0]:
        logging.warning(
            "Skipping mmseqs species enrichment because identifier rows do not "
            "match MSA rows: %d != %d",
            len(identifier_features["msa_species_identifiers"]),
            msa.shape[0],
        )
        return
    feature_dict.update(identifier_features)
