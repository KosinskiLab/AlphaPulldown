"""
Create monomeric or multimeric objects using different pipelines
Copyright (c) 2025 European Molecular Biology Laboratory
Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from absl import logging

from alphapulldown.utils.file_handling import zip_msa_files, unzip_msa_files, remove_msa_files
from alphapulldown.utils.multimeric_template_utils import (
    extract_multimeric_template_features_for_single_chain,
    prepare_multimeric_template_meta_info,
)
from alphapulldown.providers import MSAProvider, TemplateProvider

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

# Helper to create a MonomericObject for a subsequence range
def make_monomer_from_range(
    fasta_path: str,
    chain_index: int,
    start: int,
    stop: int,
    msa_provider: MSAProvider,
    tpl_provider: TemplateProvider
) -> MonomericObject:
    """
    Parse a region from a FASTA file and return a MonomericObject for that subsequence.
    """
    content = Path(fasta_path).read_text()
    from alphafold.data import parsers
    seqs, descs = parsers.parse_fasta(content)
    try:
        seq = seqs[chain_index]
        desc = descs[chain_index]
    except IndexError:
        raise AlphaPulldownError(f"Chain index {chain_index} out of bounds for {fasta_path}")
    region = ProteinSequence(desc, seq).get_region(start, stop)
    return MonomericObject(region.identifier, region.sequence, msa_provider, tpl_provider)

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
            species_identifiers=self.species_identifiers,
            uniprot_accessions=self.uniprot_accessions
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
            aatype=self.aatype[:, idx0:end],
            all_atom_positions=self.all_atom_positions[:, idx0:end],
            all_atom_mask=self.all_atom_mask[:, idx0:end],
            template_domain_names=self.template_domain_names,
            confidence_scores=(None if self.confidence_scores is None else self.confidence_scores[:, idx0:end]),
            release_date=(None if self.release_date is None else self.release_date[:, idx0:end])
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
            species_identifiers=self.msa.species_identifiers,
            uniprot_accessions=self.msa.uniprot_accessions
        )
        new_template = TemplateFeatures(
            aatype=np.concatenate([sf.template.aatype for sf in sliced], axis=1),
            all_atom_positions=np.concatenate([sf.template.all_atom_positions for sf in sliced], axis=1),
            all_atom_mask=np.concatenate([sf.template.all_atom_mask for sf in sliced], axis=1),
            template_domain_names=self.template.template_domain_names,
            confidence_scores=None if self.template.confidence_scores is None else np.concatenate([sf.template.confidence_scores for sf in sliced], axis=1),
            release_date=None if self.template.release_date is None else np.concatenate([sf.template.release_date for sf in sliced], axis=1)
        )
        return ProteinFeatures(
            sequence=new_sequence,
            msa=new_msa,
            template=new_template,
            extra=self.extra.copy()
        )

class MonomericObject:
    """
    Build features for a single protein using MSA and template providers.
    """
    def __init__(
        self,
        description: str,
        sequence: str,
        msa_provider: MSAProvider,
        tpl_provider: TemplateProvider
    ):
        self.sequence = ProteinSequence(description, sequence)
        self.msa_provider = msa_provider
        self.tpl_provider = tpl_provider
        self.features: Optional[ProteinFeatures] = None

    @property
    def description(self) -> str:
        return self.sequence.identifier

    def make_features(
        self,
        output_dir: str,
        use_precomputed_msa: bool = False,
        save_msa: bool = True,
        compress_msa: bool = False
    ) -> ProteinFeatures:
        outdir = Path(output_dir) / self.description
        outdir.mkdir(parents=True, exist_ok=True)
        # unzip any existing MSAs
        zipped = unzip_msa_files(outdir)
        # generate MSA and template features
        msa_feats, dummy_tpl = self.msa_provider.run(
            seq_id=self.description,
            sequence=self.sequence.sequence,
            outdir=outdir,
            use_precomputed=use_precomputed_msa
        )
        tpl_feats = self.tpl_provider.run(self.sequence.sequence)
        # post-process cleaning
        if not save_msa:
            remove_msa_files(outdir)
        if compress_msa:
            zip_msa_files(outdir)
            if zipped:
                zip_msa_files(outdir)
        self.features = ProteinFeatures(
            sequence=self.sequence,
            msa=msa_feats,
            template=tpl_feats,
            extra={}
        )
        return self.features

class MultimericObject:
    """Combine multiple MonomericObject features into a multimeric feature set"""
    def __init__(
        self,
        interactors: List[MonomericObject],
        pair_msa: bool = True,
        multimeric_template: bool = False,
        multimeric_template_meta_data: Optional[str] = None,
        multimeric_template_dir: Optional[str] = None,
    ):
        self.interactors = interactors
        self.pair_msa = pair_msa
        self.multimeric_template = multimeric_template
        self.multimeric_template_dir = multimeric_template_dir
        # ensure each monomerised object has features
        for mono in interactors:
            if mono.features is None:
                raise MissingFeatureError(f"Missing features for interactor {mono.description}")
        # load and inject multimeric templates
        if multimeric_template and multimeric_template_meta_data and multimeric_template_dir:
            meta = prepare_multimeric_template_meta_info(multimeric_template_meta_data, multimeric_template_dir)
            for mon_id, mapping in meta.items():
                mono = next(m for m in self.interactors if m.sequence.identifier == mon_id)
                for cif, chain_id in mapping.items():
                    pdb_id = Path(cif).stem
                    ext = extract_multimeric_template_features_for_single_chain(
                        query_seq=mono.sequence.sequence,
                        pdb_id=pdb_id,
                        chain_id=chain_id,
                        mmcif_file=str(Path(multimeric_template_dir)/cif)
                    )
                    # update monomer raw
                    mono.features.extra.setdefault('multimeric', []).append(ext.features)
        # build multimeric features via pipeline_multimer
        from alphafold.data import parsers, pipeline_multimer, msa_pairing, feature_processing
        all_fasta = "".join(f">{m.description}\n{m.sequence.sequence}\n" for m in self.interactors)
        seqs, descs = parsers.parse_fasta(all_fasta)
        chain_map = pipeline_multimer._make_chain_id_map(seqs, descs)
        per_chain = {cid: pipeline_multimer.convert_monomer_features(mono.features.extra.get('raw', {}), cid)
                     for cid, mono in zip(chain_map, self.interactors)}
        assembled = pipeline_multimer.add_assembly_features(per_chain)
        chains = (msa_pairing.create_paired_features(assembled.values())
                  if pair_msa and not feature_processing._is_homomer_or_monomer(assembled.values())
                  else assembled.values())
        merged = msa_pairing.merge_chain_features(np_chains_list=chains, pair_msa_sequences=pair_msa, max_templates=4)
        processed = feature_processing.process_final(merged)
        self.feature_dict = pipeline_multimer.pad_msa(processed, max_msa=512)

    @staticmethod
    def remove_all_seq_features(chain_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{k: v for k, v in feats.items() if '_all_seq' not in k} for feats in chain_list]

    def save_binary_matrix(self, matrix: np.ndarray, file_path: str) -> None:
        from PIL import Image, ImageDraw, ImageFont
        h, w = matrix.shape
        img_data = np.zeros((h, w, 3), np.uint8)
        img_data[matrix==1] = [255,0,0]
        img_data[matrix==0] = [0,0,255]
        img = Image.fromarray(img_data)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial", 16)
        except OSError:
            font = ImageFont.load_default()
        for col in range(w-1):
            if matrix[:,col].any() != matrix[:,col+1].any():
                txt = str(col+1)
                tw, th = draw.textsize(txt, font=font)
                x = (col+0.5)*img.width/w - tw/2
                y = img.height - th
                draw.text((x,y), txt, font=font, fill=(0,0,0))
        img.save(file_path)
