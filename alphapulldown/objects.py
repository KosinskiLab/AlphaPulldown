"""
Create monomeric or multimeric objects using different pipelines
Copyright (c) 2025 European Molecular Biology Laboratory
Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from __future__ import annotations
import numpy as np
from absl import logging
from alphafold.data.tools import jackhmmer
from alphafold.data import parsers, pipeline, pipeline_multimer, msa_pairing, feature_processing
from colabfold.batch import get_msa_and_templates, msa_to_str, build_monomer_feature
from colabfold.utils import DEFAULT_API_SERVER

from alphapulldown.utils.file_handling import (
    temp_fasta_file,
    zip_msa_files,
    unzip_msa_files,
    remove_msa_files
)
from alphapulldown.utils.multimeric_template_utils import (
    extract_multimeric_template_features_for_single_chain,
    prepare_multimeric_template_meta_info,
)

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
    uniprot_runner: Optional[jackhmmer.Jackhmmer] = None
) -> 'MonomericObject':
    """
    Parse a region from a FASTA file and return a MonomericObject for that subsequence.

    Args:
        fasta_path: Path to input FASTA file.
        chain_index: Index of the sequence in the file (0-based).
        start: 1-based start position.
        stop: 1-based end position (inclusive).
        uniprot_runner: Optional Jackhmmer runner.
    Raises:
        AlphaPulldownError: for parsing or index errors.
    """
    content = Path(fasta_path).read_text()
    seqs, descs = parsers.parse_fasta(content)
    try:
        seq = seqs[chain_index]
        desc = descs[chain_index]
    except IndexError:
        raise AlphaPulldownError(f"Chain index {chain_index} out of bounds for {fasta_path}")
    region = ProteinSequence(desc, seq).get_region(start, stop)
    return MonomericObject(region.identifier, region.sequence, uniprot_runner)

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
            release_date=self.release_date
        )

@dataclass
class ProteinFeatures:
    """Complete set of features for a protein"""
    sequence: ProteinSequence
    msa: MSAFeatures
    template: Optional[TemplateFeatures] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_region(self, start: int, end: int) -> 'ProteinFeatures':
        return ProteinFeatures(
            sequence=self.sequence.get_region(start, end),
            msa=self.msa.get_region(start, end),
            template=None if self.template is None else self.template.get_region(start, end),
            extra=self.extra.copy()
        )

class MonomericObject:
    """
    Build features for a single protein using chosen backend ('uniprot' or 'mmseqs').
    """
    def __init__(
        self,
        description: str,
        sequence: str,
        uniprot_runner: Optional[jackhmmer.Jackhmmer] = None,
    ):
        self.sequence = ProteinSequence(description, sequence)
        self.uniprot_runner = uniprot_runner
        self.raw_features: Dict[str, Any] = {}
        self.features: Optional[ProteinFeatures] = None

    @property
    def description(self) -> str:
        return self.sequence.identifier

    def make_features(
        self,
        pipeline: Optional[pipeline.DataPipeline],
        output_dir: str,
        use_precomputed_msa: bool = False,
        save_msa: bool = True,
        use_mmseqs2: bool = False
    ) -> ProteinFeatures:
        method = 'mmseqs' if use_mmseqs2 else 'uniprot'
        try:
            return self.generate_features(
                method=method,
                output_dir=Path(output_dir),
                use_precomputed=use_precomputed_msa,
                save_msa=save_msa,
                compress_msa=False
            )
        except Exception as e:
            raise AlphaPulldownError(f"Feature generation failed for {self.description}: {e}") from e

    def generate_features(
        self,
        method: str,
        output_dir: Path,
        use_precomputed: bool = False,
        save_msa: bool = True,
        compress_msa: bool = False,
    ) -> ProteinFeatures:
        outdir = output_dir / self.sequence.identifier
        outdir.mkdir(parents=True, exist_ok=True)
        zipped = unzip_msa_files(outdir)

        if method == 'uniprot':
            msa_feats = self._run_uniprot_msa(outdir, use_precomputed)
            tpl_feats = None
        else:
            msa_feats, tpl_feats = self._run_mmseqs_msa(outdir, use_precomputed)

        if not save_msa and not use_precomputed:
            remove_msa_files(outdir)
        if compress_msa:
            zip_msa_files(outdir)
            if zipped:
                zip_msa_files(outdir)

        self.features = ProteinFeatures(
            sequence=self.sequence,
            msa=msa_feats,
            template=tpl_feats,
            extra={'raw': self.raw_features},
        )
        return self.features

    def _run_uniprot_msa(self, outdir: Path, use_precomputed: bool) -> MSAFeatures:
        fasta = f">{self.sequence.identifier}\n{self.sequence.sequence}"
        with temp_fasta_file(fasta) as fasta_file:
            result = pipeline.run_msa_tool(
                self.uniprot_runner, fasta_file, str(outdir / 'uniprot.sto'), 'sto', use_precomputed
            )
        msa = parsers.parse_stockholm(result['sto']).truncate(max_seqs=50000)
        feats = pipeline.make_msa_features([msa])
        self.raw_features = feats.copy()
        valid = msa_pairing.MSA_FEATURES + ('msa_species_identifiers','msa_uniprot_accession_identifiers')
        msa_dict = {k: feats[k] for k in valid}
        return MSAFeatures(
            msa=msa_dict['msa'],
            deletion_matrix=msa_dict['deletion_matrix_int_all_seq'],
            species_identifiers=list(msa_dict['msa_species_identifiers_all_seq']),
            uniprot_accessions=list(msa_dict['msa_uniprot_accession_identifiers_all_seq']),
        )

    def _run_mmseqs_msa(
        self, outdir: Path, use_precomputed: bool
    ) -> Tuple[MSAFeatures, Optional[TemplateFeatures]]:
        a3m = outdir / f"{self.sequence.identifier}.a3m"
        if use_precomputed and a3m.is_file():
            lines = [msa_to_str(get_msa_and_templates(
                jobname=self.sequence.identifier,
                query_sequences=self.sequence.sequence,
                a3m_lines=[a3m.read_text()],
                result_dir=str(outdir), msa_mode="mmseqs2_uniref_env",
                use_templates=True, host_url=DEFAULT_API_SERVER,
                user_agent="alphapulldown"
            )[0], None, None, None)]
            unpaired, *_ , tpl_raw = get_msa_and_templates(
                jobname=self.sequence.identifier,
                query_sequences=self.sequence.sequence,
                a3m_lines=lines,
                result_dir=str(outdir), msa_mode="mmseqs2_uniref_env",
                use_templates=True, host_url=DEFAULT_API_SERVER,
                user_agent="alphapulldown"
            )
        else:
            unpaired, *_ , tpl_raw = get_msa_and_templates(
                jobname=self.sequence.identifier,
                query_sequences=self.sequence.sequence,
                a3m_lines=None,
                result_dir=str(outdir), msa_mode="mmseqs2_uniref_env",
                use_templates=True, host_url=DEFAULT_API_SERVER,
                user_agent="alphapulldown"
            )
            serialized = msa_to_str(unpaired, None, None, None)
            a3m.write_text(serialized)

        fea = build_monomer_feature(self.sequence.sequence, unpaired[0], tpl_raw[0])
        self.raw_features = fea.copy()
        msa_feats = MSAFeatures(
            msa=fea['msa'],
            deletion_matrix=fea['deletion_matrix_int'],
            species_identifiers=[s.decode() for s in fea['msa_species_identifiers']],
            uniprot_accessions=[u.decode() for u in fea.get('msa_uniprot_accession_identifiers', [])],
        )
        tpl_feats = TemplateFeatures(
            aatype=fea['template_aatype'],
            all_atom_positions=fea['template_all_atom_positions'],
            all_atom_mask=fea['template_all_atom_masks'],
            template_domain_names=[dn.decode() for dn in fea['template_domain_names']],
            confidence_scores=fea.get('template_confidence_scores'),
            release_date=fea.get('template_release_date'),
        )
        return msa_feats, tpl_feats

    class MultimericObject:
        """Combine multiple MonomericObject features into multimeric feature set"""

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
            self.chain_id_map: Dict[str, Any] = {}
            self.multimeric_template_dir = multimeric_template_dir

            # load and attach multimeric‐template metadata if requested
            if multimeric_template and multimeric_template_meta_data and multimeric_template_dir:
                try:
                    self.multimeric_template_meta_data = prepare_multimeric_template_meta_info(
                        multimeric_template_meta_data,
                        multimeric_template_dir
                    )
                except Exception as e:
                    raise TemplateFeatureError(f"Failed to parse multimeric template metadata: {e}") from e

            # ensure each monomer has features
            for mono in interactors:
                if mono.features is None:
                    raise MissingFeatureError(f"Missing features for interactor {mono.description}")

            # insert per‐chain template features if requested
            if self.multimeric_template:
                for mon_id, mapping in self.multimeric_template_meta_data.items():
                    mono = next(m for m in self.interactors if m.sequence.identifier == mon_id)
                    for cif_file, chain_id in mapping.items():
                        pdb_id = Path(cif_file).stem
                        try:
                            ext_feats = extract_multimeric_template_features_for_single_chain(
                                query_seq=mono.sequence.sequence,
                                pdb_id=pdb_id,
                                chain_id=chain_id,
                                mmcif_file=str(Path(self.multimeric_template_dir) / cif_file)
                            )
                        except Exception as e:
                            raise TemplateFeatureError(
                                f"Error extracting template features from {cif_file}: {e}") from e
                        mono.raw_features.update(ext_feats.features)

            # build mapping from chain IDs to sequences
            all_fasta = "".join(f">{m.sequence.identifier}\n{m.sequence.sequence}\n" for m in self.interactors)
            seqs, descs = parsers.parse_fasta(all_fasta)
            self.chain_id_map = pipeline_multimer._make_chain_id_map(sequences=seqs, descriptions=descs)

            # merge into a single feature dict
            per_chain = {}
            for chain_id, mono in zip(self.chain_id_map, self.interactors):
                raw = mono.raw_features
                per_chain[chain_id] = pipeline_multimer.convert_monomer_features(raw, chain_id)

            # pair MSAs if desired, then pad and finalize
            chains_for_pairing = (
                msa_pairing.create_paired_features(per_chain.values())
                if pair_msa and not feature_processing._is_homomer_or_monomer(per_chain.values())
                else per_chain.values()
            )

            merged = msa_pairing.merge_chain_features(
                np_chains_list=chains_for_pairing,
                pair_msa_sequences=pair_msa,
                max_templates=4
            )
            processed = feature_processing.process_final(merged)
            self.feature_dict = pipeline_multimer.pad_msa(processed, max_msa=512)
