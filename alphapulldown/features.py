"""
Data classes for monomeric features objects (builders.py)
Copyright (c) 2025 European Molecular Biology Laboratory
Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from absl import logging

logging.set_verbosity(logging.INFO)

# Custom exceptions for clearer error handling
class AlphaPulldownError(Exception):
    """Base exception for AlphaPulldown errors."""

class InvalidRegionError(AlphaPulldownError):
    """Raised when a requested sequence region is invalid."""

class MissingFeatureError(AlphaPulldownError):
    """Raised when expected features are missing for a MonomericObject."""

class TemplateFeatureError(AlphaPulldownError):
    """Raised for errors processing multimeric template features."""

@dataclass
class ProteinSequence:
    """A protein sequence with identifier"""
    identifier: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)

    def get_region(self, start: int, end: int) -> 'ProteinSequence':
        if start < 1 or end > self.length or start > end:
            raise InvalidRegionError(f"Invalid region: {start}-{end} for length {self.length}")
        return ProteinSequence(
            identifier=f"{self.identifier}_{start}-{end}",
            sequence=self.sequence[start-1:end]
        )

@dataclass
class MSAFeatures:
    """Multiple sequence alignment features"""
    msa: np.ndarray
    deletion_matrix: np.ndarray
    species_identifiers: List[str]
    uniprot_accessions: Optional[List[str]] = None

    @property
    def num_sequences(self) -> int:
        return self.msa.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.msa.shape[1]

    def get_region(self, start: int, end: int) -> 'MSAFeatures':
        idx0 = start - 1
        return MSAFeatures(
            msa=self.msa[:, idx0:end],
            deletion_matrix=self.deletion_matrix[:, idx0:end],
            # copy lists so mutations on the slice don’t affect the original
            species_identifiers=list(self.species_identifiers),
            uniprot_accessions=(
                None if self.uniprot_accessions is None
                else list(self.uniprot_accessions)
            )
        )


@dataclass
class TemplateFeatures:
    """Template features for structure prediction"""
    aatype: np.ndarray
    all_atom_positions: np.ndarray
    all_atom_mask: np.ndarray
    template_domain_names: List[str]
    confidence_scores: Optional[np.ndarray] = None
    release_date: Optional[np.ndarray] = None

    @property
    def num_templates(self) -> int:
        return self.aatype.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.aatype.shape[1]

    def get_region(self, start: int, end: int) -> 'TemplateFeatures':
        idx0 = start - 1
        return TemplateFeatures(
            aatype=self.aatype[:, idx0:end, :],
            all_atom_positions=self.all_atom_positions[:, idx0:end, :, :],
            all_atom_mask=self.all_atom_mask[:, idx0:end, :],
            template_domain_names=list(self.template_domain_names),
            confidence_scores=(
                None if self.confidence_scores is None
                else self.confidence_scores[:, idx0:end]
            ),
            release_date=(
                None if self.release_date is None
                else self.release_date[:, idx0:end]
            )
        )


@dataclass
class ProteinFeatures:
    """Complete set of features for a protein"""
    sequence: ProteinSequence
    msa: MSAFeatures
    template: TemplateFeatures
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_region(self, start: int, end: int) -> 'ProteinFeatures':
        return ProteinFeatures(
            sequence=self.sequence.get_region(start, end),
            msa=self.msa.get_region(start, end),
            template=self.template.get_region(start, end),
            extra=self.extra.copy()
        )

    def get_regions(self, ranges: List[Tuple[int, int]]) -> 'ProteinFeatures':
        sliced = [self.get_region(s, e) for s, e in ranges]
        new_seq = ''.join(sf.sequence.sequence for sf in sliced)
        new_id = f"{self.sequence.identifier}_" + '_'.join(f"{s}-{e}" for s, e in ranges)
        new_sequence = ProteinSequence(identifier=new_id, sequence=new_seq)
        new_msa = MSAFeatures(
            msa=np.concatenate([sf.msa.msa for sf in sliced], axis=1),
            deletion_matrix=np.concatenate([sf.msa.deletion_matrix for sf in sliced], axis=1),
            # copy lists here
            species_identifiers=list(self.msa.species_identifiers),
            uniprot_accessions=(
                None if self.msa.uniprot_accessions is None
                else list(self.msa.uniprot_accessions)
            )
        )
        new_template = TemplateFeatures(
            aatype=np.concatenate([sf.template.aatype for sf in sliced], axis=1),
            all_atom_positions=np.concatenate([sf.template.all_atom_positions for sf in sliced], axis=1),
            all_atom_mask=np.concatenate([sf.template.all_atom_mask for sf in sliced], axis=1),
            # copy list here
            template_domain_names=list(self.template.template_domain_names),
            confidence_scores=(
                None if self.template.confidence_scores is None
                else np.concatenate([sf.template.confidence_scores for sf in sliced], axis=1)
            ),
            release_date=(
                None if self.template.release_date is None
                else np.concatenate([sf.template.release_date for sf in sliced], axis=1)
            )
        )
        return ProteinFeatures(
            sequence=new_sequence,
            msa=new_msa,
            template=new_template,
            extra=self.extra.copy()
        )
