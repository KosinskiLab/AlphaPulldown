"""
Create monomeric or multimeric objects using different pipelines
Copyright (c) 2025 European Molecular Biology Laboratory
Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from absl import logging

from alphapulldown.utils.file_handling import zip_msa_files, unzip_msa_files, remove_msa_files
from alphapulldown.utils.multimeric_template_utils import (
    extract_multimeric_template_features_for_single_chain,
    prepare_multimeric_template_meta_info,
)
from alphapulldown.providers import MSAProvider, TemplateProvider, MMseqsMSAProvider
from alphapulldown.features import ProteinSequence, ProteinFeatures, AlphaPulldownError, MissingFeatureError

logging.set_verbosity(logging.INFO)

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

        # unzip if needed
        unzip_msa_files(outdir)

        # always call the MSA provider
        msa_res, tpl_res = self.msa_provider.run(
            seq_id=self.description,
            sequence=self.sequence.sequence,
            outdir=outdir,
            use_precomputed=use_precomputed_msa
        )

        if isinstance(self.msa_provider, MMseqsMSAProvider):
            # MMseqs2 already gave us real templates
            msa_feats, tpl_feats = msa_res, tpl_res
        else:
            # ignore the dummy tpl_res and get real templates separately
            msa_feats = msa_res
            tpl_feats = self.tpl_provider.run(self.sequence.sequence)

        # cleanup and optional re‐zipping
        if not save_msa:
            remove_msa_files(outdir)
        if compress_msa:
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
