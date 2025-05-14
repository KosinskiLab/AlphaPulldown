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
    mono = MonomericObject(region.identifier, region.sequence, uniprot_runner)
    return mono

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
        """
        Extract and concatenate multiple discontiguous regions.

        Args:
            ranges: list of (start, end) 1-based inclusive indices.
        Returns:
            A new ProteinFeatures with concatenated sequence, MSA, and template.
        """
        # slice each region
        sliced = [self.get_region(s, e) for s, e in ranges]
        # build new sequence and identifier
        new_seq = ''.join(sf.sequence.sequence for sf in sliced)
        new_id = f"{self.sequence.identifier}_" + '_'.join(f"{s}-{e}" for s, e in ranges)
        new_sequence = ProteinSequence(identifier=new_id, sequence=new_seq)
        # concatenate MSA
        new_msa_arr = np.concatenate([sf.msa.msa for sf in sliced], axis=1)
        new_del_mat = np.concatenate([sf.msa.deletion_matrix for sf in sliced], axis=1)
        new_msa = MSAFeatures(
            msa=new_msa_arr,
            deletion_matrix=new_del_mat,
            species_identifiers=self.msa.species_identifiers,
            uniprot_accessions=self.msa.uniprot_accessions
        )
        # concatenate template
        if self.template is not None:
            new_aatype = np.concatenate([sf.template.aatype for sf in sliced], axis=1)
            new_positions = np.concatenate([sf.template.all_atom_positions for sf in sliced], axis=1)
            new_mask = np.concatenate([sf.template.all_atom_mask for sf in sliced], axis=1)
            new_conf = (None if self.template.confidence_scores is None else
                        np.concatenate([sf.template.confidence_scores for sf in sliced], axis=1))
            new_dates = (None if self.template.release_date is None else
                         np.concatenate([sf.template.release_date for sf in sliced], axis=1))
            new_template = TemplateFeatures(
                aatype=new_aatype,
                all_atom_positions=new_positions,
                all_atom_mask=new_mask,
                template_domain_names=self.template.template_domain_names,
                confidence_scores=new_conf,
                release_date=new_dates
            )
        else:
            new_template = None
        return ProteinFeatures(
            sequence=new_sequence,
            msa=new_msa,
            template=new_template,
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
        uniprot_runner: Optional[jackhmmer.Jackhmmer] = None
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
        compress_msa: bool = False,
        use_mmseqs2: bool = False
    ) -> ProteinFeatures:
        method = 'mmseqs' if use_mmseqs2 else 'uniprot'
        try:
            return self.generate_features(
                method=method,
                output_dir=Path(output_dir),
                use_precomputed=use_precomputed_msa,
                save_msa=save_msa,
                compress_msa=compress_msa
            )
        except Exception as e:
            raise AlphaPulldownError(f"Feature generation failed for {self.description}: {e}") from e

    def generate_features(
        self,
        method: str,
        output_dir: Path,
        use_precomputed: bool = False,
        save_msa: bool = True,
        compress_msa: bool = False
    ) -> ProteinFeatures:
        outdir = output_dir / self.sequence.identifier
        outdir.mkdir(parents=True, exist_ok=True)
        zipped = unzip_msa_files(outdir)

        if method == 'uniprot':
            msa_feats = self._run_uniprot_msa(outdir, use_precomputed)
            seq_len = self.sequence.length
            tpl_feats = TemplateFeatures(
                aatype=np.zeros((1, seq_len), dtype=np.int32),
                all_atom_positions=np.zeros((1, seq_len, 3), dtype=np.float32),
                all_atom_mask=np.zeros((1, seq_len, 3), dtype=np.float32),
                template_domain_names=[],
                confidence_scores=np.ones((1, seq_len), dtype=np.float32),
                release_date=np.array(['none'])
            )
        else:
            msa_feats, tpl_feats = self._run_mmseqs_msa(outdir, use_precomputed)

        if not save_msa:
            if use_precomputed:
                logging.warning(
                    f"You chose not to save MSA files but still want to use precomputed ones; these will not be removed."
                )
            else:
                remove_msa_files(outdir)
        if compress_msa:
            zip_msa_files(outdir)
            if zipped:
                zip_msa_files(outdir)

        self.features = ProteinFeatures(
            sequence=self.sequence,
            msa=msa_feats,
            template=tpl_feats,
            extra={'raw': self.raw_features}
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
    ) -> Tuple[MSAFeatures, TemplateFeatures]:
        a3m = outdir / f"{self.sequence.identifier}.a3m"
        # prepare input lines for mmseqs
        if use_precomputed and a3m.is_file():
            lines = [msa_to_str(get_msa_and_templates(
                jobname=self.sequence.identifier,
                query_sequences=self.sequence.sequence,
                a3m_lines=[a3m.read_text()],
                result_dir=str(outdir),
                msa_mode="mmseqs2_uniref_env",
                use_templates=True,
                host_url=DEFAULT_API_SERVER,
                user_agent="alphapulldown"
            )[0], None, None, None)]
            unpaired, *_ , tpl_raw = get_msa_and_templates(
                jobname=self.sequence.identifier,
                query_sequences=self.sequence.sequence,
                a3m_lines=lines,
                result_dir=str(outdir),
                msa_mode="mmseqs2_uniref_env",
                use_templates=True,
                host_url=DEFAULT_API_SERVER,
                user_agent="alphapulldown"
            )
        else:
            unpaired, *_ , tpl_raw = get_msa_and_templates(
                jobname=self.sequence.identifier,
                query_sequences=self.sequence.sequence,
                a3m_lines=None,
                result_dir=str(outdir),
                msa_mode="mmseqs2_uniref_env",
                use_templates=True,
                host_url=DEFAULT_API_SERVER,
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
        self.chain_id_map: Dict[str, Any] = {}
        self.multimeric_template_dir = multimeric_template_dir

        # load multimeric template metadata
        if multimeric_template and multimeric_template_meta_data and multimeric_template_dir:
            try:
                self.multimeric_template_meta_data = prepare_multimeric_template_meta_info(
                    multimeric_template_meta_data,
                    multimeric_template_dir
                )
            except Exception as e:
                raise TemplateFeatureError(f"Failed to parse multimeric template metadata: {e}") from e

        # ensure features exist
        for mono in interactors:
            if mono.features is None:
                raise MissingFeatureError(f"Missing features for interactor {mono.description}")

        # inject multimeric template features
        if self.multimeric_template:
            for mon_id, mapping in self.multimeric_template_meta_data.items():
                mono = next(m for m in self.interactors if m.sequence.identifier == mon_id)
                for cif_file, chain_id in mapping.items():
                    pdb_id = Path(cif_file).stem
                    try:
                        ext = extract_multimeric_template_features_for_single_chain(
                            query_seq=mono.sequence.sequence,
                            pdb_id=pdb_id,
                            chain_id=chain_id,
                            mmcif_file=str(Path(self.multimeric_template_dir) / cif_file)
                        )
                    except Exception as e:
                        raise TemplateFeatureError(f"Error extracting template features from {cif_file}: {e}") from e
                    mono.raw_features.update(ext.features)

        # chain ID mapping
        all_fasta = "".join(
            f">{m.sequence.identifier}\n{m.sequence.sequence}\n" for m in self.interactors
        )
        seqs, descs = parsers.parse_fasta(all_fasta)
        self.chain_id_map = pipeline_multimer._make_chain_id_map(seqs, descs)

        # convert monomers to chain features
        per_chain = {}
        for chain_id, mono in zip(self.chain_id_map, self.interactors):
            per_chain[chain_id] = pipeline_multimer.convert_monomer_features(mono.raw_features, chain_id)

        # add assembly features
        per_chain = pipeline_multimer.add_assembly_features(per_chain)

        # optionally strip all_seq for unpaired
        chains = (
            msa_pairing.create_paired_features(per_chain.values())
            if pair_msa and not feature_processing._is_homomer_or_monomer(per_chain.values())
            else per_chain.values()
        )

        merged = msa_pairing.merge_chain_features(
            np_chains_list=chains,
            pair_msa_sequences=pair_msa,
            max_templates=4
        )
        processed = feature_processing.process_final(merged)
        self.feature_dict = pipeline_multimer.pad_msa(processed, max_msa=512)

    @staticmethod
    def remove_all_seq_features(chain_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip out all '_all_seq' keys for pure unpaired multimer modelling"""
        return [{k:v for k,v in feats.items() if '_all_seq' not in k} for feats in chain_list]

    def save_binary_matrix(self, matrix: np.ndarray, file_path: str) -> None:
        """Save a binary interaction/gap matrix as PNG for QC"""
        from PIL import Image, ImageDraw, ImageFont
        h, w = matrix.shape
        image_data = np.zeros((h, w, 3), dtype=np.uint8)
        image_data[matrix==1] = [255,0,0]
        image_data[matrix==0] = [0,0,255]
        img = Image.fromarray(image_data)
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

    def create_multichain_mask(self) -> np.ndarray:
        """Construct binary mask for inter-chain template overlap/gaps"""
        pdb_map = []
        no_gap = []
        for mono in self.interactors:
            length = len(mono.sequence.sequence)
            # domain name or placeholder
            domain = mono.raw_features.get('template_domain_names', [''])[0]
            pdb_map.extend([domain]*length)
            gaps = [True]*length
            for t in mono.raw_features.get('template_sequence', []):
                seq = t.decode('utf-8').strip()
                is_not_gap = [c!='-' for c in seq]
                gaps = [a and b for a,b in zip(gaps, is_not_gap)]
            no_gap.extend(gaps)
        size = len(pdb_map)
        mask = np.zeros((size,size), dtype=int)
        for i,a in enumerate(pdb_map):
            for j,b in enumerate(pdb_map):
                if a[:4]==b[:4]: mask[i,j]=1
        return mask
