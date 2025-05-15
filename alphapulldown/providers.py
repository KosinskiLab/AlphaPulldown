""" Abstract interface and concrete implementations for MSA backends

    Copyright (c) 2025 European Molecular Biology Laboratory
    Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple
from absl import logging
from alphafold.data.tools import jackhmmer
from alphafold.data import parsers, pipeline
from colabfold.batch import get_msa_and_templates, msa_to_str, build_monomer_feature
from colabfold.utils import DEFAULT_API_SERVER

from alphapulldown.features import MSAFeatures, TemplateFeatures
from alphapulldown.utils.file_handling import temp_fasta_file
import numpy as np

EMPTY_TEMPLATES = TemplateFeatures(
    aatype=np.zeros((1, 1), dtype=np.int32),
    all_atom_positions=np.zeros((1, 1, 3), dtype=np.float32),
    all_atom_mask=np.zeros((1, 1, 3), dtype=np.float32),
    template_domain_names=[],
    confidence_scores=None,
    release_date=None
)

class MSAProvider(ABC):
    """Abstract base class for MSA/template feature providers."""

    @abstractmethod
    def run(
        self,
        seq_id: str,
        sequence: str,
        outdir: Path,
        use_precomputed: bool
    ) -> Tuple[MSAFeatures, TemplateFeatures]:
        """
        Generate and return (msa_features, template_features) for a sequence.
        """
        pass


class UniprotMSAProvider(MSAProvider):
    """Jackhmmer-based Uniprot MSA provider"""

    def __init__(
        self,
        jackhmmer_binary: str,
        uniprot_db: str,
        max_hits: int = 50000
    ):
        self.runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary,
            database_path=uniprot_db,
        )
        self._max_hits = max_hits


    def run(
        self,
        seq_id: str,
        sequence: str,
        outdir: Path,
        use_precomputed: bool
    ) -> Tuple[MSAFeatures, TemplateFeatures]:
        fasta = f">{seq_id}\n{sequence}"
        with temp_fasta_file(fasta) as fasta_file:
            result = pipeline.run_msa_tool(
                self.runner,
                fasta_file,
                str(outdir / f"{seq_id}.sto"),
                'sto',
                use_precomputed
            )
        msa = parsers.parse_stockholm(result['sto']).truncate(max_seqs=self._max_hits)
        feats = pipeline.make_msa_features([msa])
        valid = pipeline.MSA_FEATURES + ('msa_species_identifiers',)
        msa_dict = {k: feats[k] for k in valid}
        msa_feats = MSAFeatures(
            msa=msa_dict['msa'],
            deletion_matrix=msa_dict['deletion_matrix_int_all_seq'],
            species_identifiers=list(msa_dict['msa_species_identifiers_all_seq']),
            uniprot_accessions=None
        )
        # return msa + dummy template
        return msa_feats, EMPTY_TEMPLATES



class MMseqsMSAProvider(MSAProvider):
    """ColabFold MMseqs2-based MSA provider"""

    def __init__(self, api_server: str = DEFAULT_API_SERVER):
        self.api = api_server

    def run(
        self,
        seq_id: str,
        sequence: str,
        outdir: Path,
        use_precomputed: bool
    ) -> Tuple[MSAFeatures, TemplateFeatures]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        a3m_path = outdir / f"{seq_id}.a3m"

        def _strip_headers(txt: str) -> str:
            return "\n".join(ln for ln in txt.splitlines() if not ln.startswith("#"))

        # Decide if we reuse the A3M
        if use_precomputed and a3m_path.is_file():
            a3m_lines = [_strip_headers(a3m_path.read_text())]
        else:
            a3m_lines = None

        # Unpack all five returns
        a3m_lines, paired_lines, uniq, card, tpl_list = get_msa_and_templates(
            jobname=seq_id,
            query_sequences=sequence,
            a3m_lines=a3m_lines,
            result_dir=outdir,
            msa_mode="mmseqs2_uniref_env",
            use_templates=True,
            custom_template_path=None,
            pair_mode="none",
            host_url=self.api,
            user_agent="alphapulldown",
        )

        # If we didn’t reuse the A3M, write it out for future runs
        if a3m_lines is not None and not use_precomputed:
            from colabfold.batch import msa_to_str
            a3m_path.write_text(msa_to_str(a3m_lines, paired_lines, uniq, card))

        # Grab the first MSA string and the first template dict
        unpaired0 = a3m_lines[0]
        tpl0       = tpl_list[0]

        # Warn if no templates were found
        names = tpl0.get("template_domain_names", [])
        if not names or all(n == b"none" for n in names):
            logging.warning("No templates found for %s", seq_id)

        # Build the feature dict
        from colabfold.batch import build_monomer_feature
        fea = build_monomer_feature(sequence, unpaired0, tpl0)

        # Wrap in our dataclasses
        msa_feats = MSAFeatures(
            msa=fea["msa"],
            deletion_matrix=fea["deletion_matrix_int"],
            species_identifiers=[s.decode() for s in fea["msa_species_identifiers"]],
            uniprot_accessions=None
        )
        tpl_feats = TemplateFeatures(
            aatype=fea["template_aatype"],
            all_atom_positions=fea["template_all_atom_positions"],
            all_atom_mask=fea["template_all_atom_masks"],
            template_domain_names=[dn.decode() for dn in fea["template_domain_names"]],
            confidence_scores=fea.get("template_confidence_scores"),
            release_date=fea.get("template_release_date")
        )

        return msa_feats, tpl_feats

# ---------- Template providers ----------

from alphafold.data.tools import hmmsearch, hhsearch
from alphafold.data import templates as tpl

class TemplateProvider(ABC):
    """Abstract base for template hit search + featurization."""

    @abstractmethod
    def run(
        self,
        sequence: str
    ) -> TemplateFeatures:
        """
        Search and featurize templates for given sequence.
        Returns a TemplateFeatures instance.
        """
        pass

class HMMERTemplateProvider(TemplateProvider):
    """HMMER-based template search and featurization"""
    def __init__(
        self,
        hmmsearch_binary: str,
        hmmbuild_binary: str,
        pdb_seqres_db: str,
        mmcif_dir: str,
        max_template_date: str,
        max_hits: int,
        kalign_binary: str,
        obsolete_pdbs_path: str
    ):
        self.searcher = hmmsearch.Hmmsearch(
            binary_path=hmmsearch_binary,
            hmmbuild_binary_path=hmmbuild_binary,
            database_path=pdb_seqres_db
        )
        self.featurizer = tpl.HmmsearchHitFeaturizer(
            mmcif_dir=mmcif_dir,
            max_template_date=max_template_date,
            max_hits=max_hits,
            kalign_binary_path=kalign_binary,
            obsolete_pdbs_path=obsolete_pdbs_path
        )

    def run(self, sequence: str) -> TemplateFeatures:
        # run search
        hits = self.searcher.query(sequence)
        # featurize
        feat_dict = self.featurizer.get_templates(hits)
        # extract arrays
        aatype = feat_dict['template_aatype']
        pos = feat_dict['template_all_atom_positions']
        mask = feat_dict.get('template_all_atom_mask', feat_dict.get('template_all_atom_masks'))
        names = [n.decode() if isinstance(n, bytes) else n for n in feat_dict['template_domain_names']]
        conf = feat_dict.get('template_confidence_scores')
        dates = feat_dict.get('template_release_date')
        return TemplateFeatures(
            aatype=aatype,
            all_atom_positions=pos,
            all_atom_mask=mask,
            template_domain_names=names,
            confidence_scores=conf,
            release_date=dates
        )

class HHsearchTemplateProvider(TemplateProvider):
    """HHsearch-based template search and featurization"""
    def __init__(
        self,
        hhsearch_binary: str,
        pdb70_db: str,
        mmcif_dir: str,
        max_template_date: str,
        max_hits: int,
        kalign_binary: str,
        obsolete_pdbs_path: str
    ):
        self.searcher = hhsearch.HHSearch(
            hhsearch_binary,
            [pdb70_db]
        )
        self.featurizer = tpl.HhsearchHitFeaturizer(
            mmcif_dir=mmcif_dir,
            max_template_date=max_template_date,
            max_hits=max_hits,
            kalign_binary_path=kalign_binary,
            obsolete_pdbs_path=obsolete_pdbs_path
        )

    def run(self, sequence: str) -> TemplateFeatures:
        hits = self.searcher.query(sequence)
        feat_dict = self.featurizer.get_templates(hits)
        aatype = feat_dict['template_aatype']
        pos = feat_dict['template_all_atom_positions']
        mask = feat_dict.get('template_all_atom_mask', feat_dict.get('template_all_atom_masks'))
        names = [n.decode() if isinstance(n, bytes) else n for n in feat_dict['template_domain_names']]
        conf = feat_dict.get('template_confidence_scores')
        dates = feat_dict.get('template_release_date')
        return TemplateFeatures(
            aatype=aatype,
            all_atom_positions=pos,
            all_atom_mask=mask,
            template_domain_names=names,
            confidence_scores=conf,
            release_date=dates
        )
