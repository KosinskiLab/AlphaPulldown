#!/usr/bin/env python3

"""Take the output of the AlphaPulldown pipeline and turn it into a ModelCIF
file with a lot of metadata in place."""

from typing import Tuple, List, Any, Dict
import datetime
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
import glob
import ast

from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.Structure import Structure as BioStructure
from absl import app, flags, logging
import numpy as np

import ihm.citations
import modelcif
import modelcif.associated
import modelcif.dumper
import modelcif.model
import modelcif.protocol

from alphapulldown.utils.file_handling import iter_seqs

# ToDo: Software versions can not have a white space, e.g. ColabFold (drop time)
# ToDo: DISCUSS Get options properly, best get the same names as used in
#       existing scripts
# ToDo: Monomers work separately - features may come from different set of
#       software, databases... so target sequences may be connected to different
#       versions of the same sequence database, may use different versions of
#       software... can this still go into single protocol steps or would this
#       prevent identifying which MSAs were produced with which software? E.g.
#       can there still be a single "sequence search" step? (This is
#       definitively for later, not the first working version of the converter
#       script)
# ToDo: sort non-ModelCIF items in the main JSON object into '__meta__'
# ToDo: protocol step software parameters
# ToDo: Example 1 from the GitHub repo mentions MMseqs2
# ToDo: Discuss input of protocol steps, feature creation has baits, sequences
#       does modelling depend on mode?
# ToDo: deal with `--max_template_date`, beta-barrel project has it as software
#       parameter
flags.DEFINE_string(
    "ap_output", None, "AlphaPulldown pipeline output directory"
)
flags.DEFINE_integer(
    "model_selected",
    None,
    "model to be converted into ModelCIF, omit to convert all models found in "
    + "'--af2_output'",
)
flags.DEFINE_bool(
    "add_associated",
    False,
    "Add models not marked by "
    + "'--model_selected' to the archive for associated files",
)
flags.DEFINE_bool("compress", False, "compress the ModelCIF file(s) using Gzip")
flags.mark_flags_as_required(["ap_output"])

FLAGS = flags.FLAGS

# ToDo: implement a flags.register_validator() checking that files/ directories
#       exist as expected.
# ToDo: implement a flags.register_validator() to make sure that
#       --add_associated is only activated if --model_selected is used, too


# pylint: disable=too-few-public-methods
class _GlobalPLDDT(modelcif.qa_metric.Global, modelcif.qa_metric.PLDDT):
    """Predicted accuracy according to the CA-only lDDT in [0,100]"""

    name = "pLDDT"
    software = None


class _GlobalPTM(modelcif.qa_metric.Global, modelcif.qa_metric.PTM):
    """Predicted accuracy according to the TM-score score in [0,1]"""

    name = "pTM"
    software = None


class _GlobalIPTM(modelcif.qa_metric.Global, modelcif.qa_metric.IpTM):
    # `python-modelcif` reads the first line of class-doc has description of a
    # score, so we need to allow long lines, here.
    # pylint: disable=line-too-long
    """Predicted protein-protein interface score, based on the TM-score score in [0,1]."""

    name = "ipTM"
    software = None


class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
    """Predicted accuracy according to the CA-only lDDT in [0,100]"""

    name = "pLDDT"
    software = None


class _PAE(modelcif.qa_metric.MetricType):
    """Predicted aligned error (in Angstroms)"""

    type = "PAE"
    other_details = None


class _LocalPairwisePAE(modelcif.qa_metric.LocalPairwise, _PAE):
    """Predicted aligned error (in Angstroms)"""

    name = "PAE"
    software = None


# pylint: enable=too-few-public-methods


class _Biopython2ModelCIF(modelcif.model.AbInitioModel):
    """Map Biopython `PDB.Structure()` object to `ihm.model()`."""

    def __init__(self, *args, **kwargs):
        """Initialise a model"""
        self.structure = kwargs.pop("bio_pdb_structure")
        self.asym = kwargs.pop("asym")

        super().__init__(*args, **kwargs)

    def get_atoms(self):
        for atm in self.structure.get_atoms():
            yield modelcif.model.Atom(
                asym_unit=self.asym[atm.parent.parent.id],
                seq_id=atm.parent.id[1],
                atom_id=atm.name,
                type_symbol=atm.element,
                x=atm.coord[0],
                y=atm.coord[1],
                z=atm.coord[2],
                het=atm.parent.id[0] != " ",
                biso=atm.bfactor,
                occupancy=atm.occupancy,
            )

    def add_scores(self, scores_json, entry_id, file_prefix, sw_dct, add_files):
        """Add QA metrics"""
        _GlobalPLDDT.software = sw_dct["AlphaFold"]
        _GlobalPTM.software = sw_dct["AlphaFold"]
        _GlobalIPTM.software = sw_dct["AlphaFold"]
        _LocalPLDDT.software = sw_dct["AlphaFold"]
        _LocalPairwisePAE.software = sw_dct["AlphaFold"]
        # global scores
        if "iptm+ptm" in scores_json:
            conf = scores_json["iptm+ptm"]
            iptm = scores_json["iptm"]
            ptm = (conf - 0.8*iptm)/0.2
        elif "ptm" in scores_json:
            iptm = 0
            ptm = scores_json["ptm"]
        self.qa_metrics.extend(
            (
                _GlobalPLDDT(np.mean(scores_json["plddt"])),
                _GlobalPTM(ptm),
                _GlobalIPTM(iptm),
            )
        )

        # local scores
        # iterate polypeptide chains
        # local PLDDT
        i = 0
        lpae = []
        # aa_only=False includes non-canonical amino acids but seems to skip
        # non-peptide-linking residues like ions
        # make C-N radius huge to make it work on unrelaxed AF models
        polypeptides = PPBuilder(radius=999999999).build_peptides(self.structure, aa_only=False)
        for chn_i in polypeptides:
            for res_i in chn_i:
                # local pLDDT
                # Assertion assumes that pLDDT values are also stored in the
                # B-factor column.
                assert (
                    round(scores_json["plddt"][i], 2)
                    == next(res_i.get_atoms()).bfactor
                )
                self.qa_metrics.append(
                    _LocalPLDDT(
                        self.asym[res_i.parent.id].residue(res_i.id[1]),
                        round(scores_json["plddt"][i], 2),
                    )
                )

                # pairwise alignment error
                j = 0
                # We do a 2nd iteration over the structure instead of doing
                # index magic because it keeps the code cleaner and should not
                # be noticeably slower than iterating the array directly.
                # Majority of time goes into writing files, anyway.
                for chn_j in polypeptides:
                    for res_j in chn_j:
                        lpae.append(
                            _LocalPairwisePAE(
                                self.asym[res_i.parent.id].residue(res_i.id[1]),
                                self.asym[res_j.parent.id].residue(res_j.id[1]),
                                scores_json["pae"][i][j],
                            )
                        )
                        j += 1

                i += 1
        self.qa_metrics.extend(lpae)

        # outsource PAE to associated file
        arc_files = [
            modelcif.associated.QAMetricsFile(
                f"{file_prefix}_local_pairwise_qa.cif",
                categories=["_ma_qa_metric_local_pairwise"],
                copy_categories=["_ma_qa_metric"],
                entry_id=entry_id,
                entry_details="This file is an associated file consisting "
                + "of local pairwise QA metrics. This is a partial mmCIF "
                + "file and can be validated by merging with the main "
                + "mmCIF file containing the model coordinates and other "
                + "associated data.",
                details="Predicted aligned error",
            )
        ]

        if add_files:
            arc_files.extend([x[1] for x in add_files.values()])

        return modelcif.associated.Repository(
            "",
            [
                modelcif.associated.ZipFile(
                    f"{file_prefix}.zip", files=arc_files
                )
            ],
        )


def _get_modelcif_entities(target_ents, asym_units, system):
    """Create ModelCIF entities and asymmetric units."""
    for cif_ent in target_ents:
        mdlcif_ent = modelcif.Entity(
            # 'pdb_sequence' can be used here, since AF2 always has the
            # complete sequence.
            cif_ent["pdb_sequence"],
            description=cif_ent["description"],
        )
        for pdb_chain_id in cif_ent["pdb_chain_id"]:
            asym_units[pdb_chain_id] = modelcif.AsymUnit(mdlcif_ent)
        system.entities.append(mdlcif_ent)


def _get_step_output_method_type(method_type, protocol_steps):
    """Get the output of a protocol step of a certain type."""
    for step in protocol_steps:
        if step.method_type == method_type:
            # `modelcif.data.DataGroup()` is some kind of list
            if isinstance(step.output_data, list):
                return step.output_data
            return modelcif.data.DataGroup(step.output_data)

    raise RuntimeError(f"Step with 'method_type' '{method_type}' not found.")


def _get_modelcif_protocol_input(
    input_data_group, target_entities, ref_dbs, protocol_steps
):
    """Assemble input data for a ModelCIF protocol step."""
    input_data = modelcif.data.DataGroup()
    for inpt in input_data_group:
        if inpt == "target_sequences":
            input_data.extend(target_entities)
        elif inpt == "reference_dbs":
            input_data.extend(ref_dbs)
        elif inpt.startswith("STEPTYPE$"):
            input_data.extend(
                _get_step_output_method_type(
                    inpt[len("STEPTYPE$") :], protocol_steps
                )
            )
        else:
            raise RuntimeError(f"Unknown protocol input: '{inpt}'")

    return input_data


def _get_modelcif_protocol_output(output_data_group, model):
    """Assemble output data for a ModelCIF protocol step."""
    if output_data_group == "model":
        output_data = model
    elif output_data_group == "monomer_pickle_files":
        output_data = modelcif.data.DataGroup(
            [
                modelcif.data.Data(
                    "Pickle files", details="Monomer feature/ MSA pickle files"
                )
            ]
        )
    else:
        raise RuntimeError(f"Unknown protocol output: '{output_data_group}'")
    return output_data


def _get_modelcif_protocol(
    protocol_steps, target_entities, model, sw_dict, ref_dbs
):
    """Create the protocol for the ModelCIF file."""
    protocol = modelcif.protocol.Protocol()
    for js_step in protocol_steps:
        # assemble input & output data
        input_data = _get_modelcif_protocol_input(
            js_step["input_data_group"],
            target_entities,
            ref_dbs,
            protocol.steps,
        )
        output_data = _get_modelcif_protocol_output(
            js_step["output_data_group"], model
        )
        # loop over software group and assemble software group from that
        sw_grp = modelcif.SoftwareGroup()
        for (
            pss,
            psp,
        ) in zip(  # protocol step software & protocol step parameters
            js_step["software_group"], js_step["parameter_group"]
        ):
            plst = []
            for arg, val in psp.items():
                plst.append(modelcif.SoftwareParameter(arg, val))
            if len(plst) > 0:
                # add software with individual parameters
                sw_grp.append(
                    modelcif.SoftwareWithParameters(sw_dict[pss], plst)
                )
            else:
                # add software w/o individual parameters
                sw_grp.append(sw_dict[pss])
        # ToDo: make sure AlphaPulldown is first in the SoftwareGroup() list,
        #       AlphaFold second; that influences citation order in the ModelCIF
        #       file.
        # mix everything together in a protocol step
        protocol.steps.append(
            modelcif.protocol.Step(
                input_data=input_data,
                output_data=output_data,
                name=js_step["step_name"],
                details=js_step["details"],
                software=sw_grp,
            )
        )
        protocol.steps[-1].method_type = js_step["method_type"]

    return protocol


def _cast_release_date(release_date):
    """Type cast a date into `datetime.date`"""
    # "AF2" has a special meaning, those DBs did not change since the first
    # release of AF2. This information is needed in the model-producing
    # pipeline.
    if release_date is None or release_date == "AF2":
        return None

    try:
        return datetime.datetime.strptime(release_date, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        logging.warning(
            f"Unsupported release date format found: {release_date}"
        )
        raise


def _cmp_ref_dbs(db_dct, db_objs):
    """Compare a reference DB dict to a list of ReferenceDatabase objects.
    Note: does not check the DB name!"""
    for obj in db_objs:
        if db_dct["release_date"] != obj.release_date:
            continue
        if db_dct["version"] != obj.version:
            continue
        for url in db_dct["location_url"]:
            if url == obj.url:
                return True

    return False


def _get_modelcif_ref_dbs(meta_json):
    """Get sequence databases used for monomer features."""
    # vendor formatting for DB names/ URLs, extend on KeyError
    sdb_lst = {}  # 'sequence database list' starts as dict since we need to
    # compare DBs between the different monomers.
    i = 0
    for data in meta_json.values():
        i += 1
        for db_name, vdct in data["databases"].items():
            vdct["release_date"] = _cast_release_date(vdct["release_date"])
            if db_name in sdb_lst:
                if _cmp_ref_dbs(vdct, sdb_lst[db_name]):
                    continue
            else:
                sdb_lst[db_name] = []
            for url in vdct["location_url"]:
                sdb_lst[db_name].append(
                    modelcif.ReferenceDatabase(
                        db_name,
                        url,
                        version=vdct["version"],
                        release_date=vdct["release_date"],
                    )
                )

    return [x for sublist in sdb_lst.values() for x in sublist]


def _store_as_modelcif(
    data_json: dict,
    structure: BioStructure,
    mdl_file: str,
    out_dir: str,
    compress: bool = False,
    add_files: list = None,
    # file_prfx
) -> None:
    """Create the actual ModelCIF file."""
    system = modelcif.System(
        title=data_json["_struct.title"],
        id=data_json["data_"].upper(),
        model_details=data_json["_struct.pdbx_model_details"],
    )

    # create target entities, references, source, asymmetric units & assembly
    # create an asymmetric unit and an entity per target sequence
    asym_units = {}
    _get_modelcif_entities(
        data_json["target_entities"],
        asym_units,
        system,
    )

    # ToDo: get modelling-experiment authors
    # audit_authors
    # system.authors.extend(data_json["audit_authors"])

    # set up the model to produce coordinates
    model = _Biopython2ModelCIF(
        assembly=modelcif.Assembly(asym_units.values()),
        asym=asym_units,
        bio_pdb_structure=structure,
        name=data_json["_ma_model_list.model_name"],
    )

    # create software list from feature metadata
    # ToDo: store_as_modelcif should not use __meta__
    sw_dct = _get_software_data(data_json["__meta__"])

    # process scores
    mdl_file = os.path.splitext(os.path.basename(mdl_file))[0]
    system.repositories.append(
        model.add_scores(data_json, system.id, mdl_file, sw_dct, add_files)
    )

    system.model_groups.append(modelcif.model.ModelGroup([model]))

    # ToDo: get protocol steps together
    #       - 'coevolution MSA' (create_individual_features.py) step
    #       - 'template search' is usually omitted for AF2 as that term more
    #         relates to homology modelling, AF2 uses templates in a different
    #         way and a template search usually results in a list of templates
    #         which would need to be included, here, in theory
    #      - for MSA/ template search, how to represent the outcome? Adding the
    #        pickle files quickly exceeds reasonable storage use
    #      - 'modeling' (run_multimer_jobs.py), are the four modes reflected by
    #        the JSON data/ does the JSON data look different for each mode?
    #      - are the scores only calculated by `alpha-analysis.sif` or do they
    #        come out of run_multimer_jobs.py? Does this go into its own step?
    #      - what about including the tabular summary?
    #      - model selection: only if just a certain model is translated to
    #        ModelCIF, or mix it with scoring step?
    #
    #      - model selection like for Tara
    system.protocols.append(
        _get_modelcif_protocol(
            data_json["ma_protocol_step"],
            system.entities,
            model,
            sw_dct,
            # ToDo: _store_as_modelcif should not use __meta__, __meta__ is
            #       tool specific
            _get_modelcif_ref_dbs(data_json["__meta__"]),
        )
    )

    # write `modelcif.System()` to file
    # NOTE: this will dump PAE on path provided in add_scores
    # -> hence we cheat by changing path and back while being exception-safe...
    oldpwd = os.getcwd()
    os.chdir(out_dir)
    created_files = {}
    try:
        mdl_file = f"{mdl_file}.cif"
        with open(
            mdl_file,
            "w",
            encoding="ascii",
        ) as mmcif_fh:
            modelcif.dumper.write(mmcif_fh, [system])
        if compress:
            mdl_file = _compress_cif_file(mdl_file)
        created_files[mdl_file] = (
            os.path.join(out_dir, mdl_file),
            _get_assoc_mdl_file(mdl_file, data_json),
        )
        # Create associated archive
        for archive in system.repositories[0].files:
            with zipfile.ZipFile(
                archive.path, "w", zipfile.ZIP_BZIP2
            ) as cif_zip:
                for zfile in archive.files:
                    try:
                        # Regardless off error, fall back to `zfile.path`, the
                        # other path is only needed as a special case.
                        # pylint: disable=bare-except
                        sys_path = add_files[zfile.path][0]
                    except:
                        sys_path = zfile.path
                    cif_zip.write(sys_path, arcname=zfile.path)
                    os.remove(sys_path)
            created_files[archive.path] = (
                os.path.join(out_dir, archive.path),
                _get_assoc_zip_file(archive.path, data_json),
            )
    finally:
        os.chdir(oldpwd)

    return created_files


def _get_assoc_mdl_file(fle_path, data_json):
    """Generate a `modelcif.associated.File` object that looks like a CIF
    file."""
    cfile = modelcif.associated.File(
        fle_path,
        details=data_json["_ma_model_list.model_name"],
    )
    cfile.file_format = "cif"
    return cfile


def _get_assoc_zip_file(fle_path, data_json):
    """Create a `modelcif.associated.File` object that looks like a ZIP file.
    This is NOT the archive ZIP file for the PAEs but to store that in the
    ZIP archive of the selected model."""
    zfile = modelcif.associated.File(
        fle_path,
        details="archive with multiple files for "
        + data_json["_ma_model_list.model_name"],
    )
    zfile.file_format = "other"
    return zfile


def _compress_cif_file(cif_file):
    """Compress CIF file and delete original."""
    cif_gz_file = cif_file + ".gz"
    with open(cif_file, "rb") as f_in:
        with gzip.open(cif_gz_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(cif_file)
    return cif_gz_file


def _get_model_details(cmplx_name: str, data_json: dict) -> str:
    """Get the model description."""
    ap_versions = []
    af2_version = None
    for mnmr in data_json["__meta__"]:  # `mnmr = monomer`
        if (
            data_json["__meta__"][mnmr]["software"]["AlphaPulldown"]["version"]
            not in ap_versions
        ):
            ap_versions.append(
                data_json["__meta__"][mnmr]["software"]["AlphaPulldown"][
                    "version"
                ]
            )
        # AlphaFold-Multimer builds the model we are looking at, can only be a
        # single version.
        if af2_version is None:
            af2_version = data_json["__meta__"][mnmr]["software"]["AlphaFold"][
                "version"
            ]
        else:
            if (
                data_json["__meta__"][mnmr]["software"]["AlphaFold"]["version"]
                != af2_version
            ):
                # pylint: disable=line-too-long
                raise RuntimeError(
                    "Different versions of AlphaFold-Multimer found: "
                    + f"'{data_json['__meta__'][mnmr]['software']['alphafold']['version']}'"
                    + f" vs. '{af2_version}'"
                )

    return (
        f"Model generated for {cmplx_name}, produced "
        + f"using AlphaFold-Multimer ({af2_version}) as implemented by "
        + f"AlphaPulldown ({', '.join(ap_versions)})."
    )


def _file_exists_or_exit(path, msg):
    """Check if a file exists, otherwise exit."""
    if not os.path.isfile(path):
        logging.info(msg)
        sys.exit()


def _cast_param(val):
    """Cast a string input val to its actual data type."""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    if val == "True":
        return True
    if val == "False":
        return False

    return val


def _get_software_with_parameters(sw_dict, other_dict):
    """Get software with versions and parameters."""

    # ToDo: deal with `use_mmseqs=True`
    def _assemble_params(key, known_args, swwp):
        for mthd in known_args[key]["method_type"]:
            for tool in known_args[key]["sw"]:
                if mthd not in swwp[tool]:
                    swwp[tool][mthd] = {}
                swwp[tool][mthd][f"--{key}"] = _cast_param(val)

    known_args = {
        "db_preset": {"sw": ["AlphaFold"], "method_type": ["coevolution MSA"]},
        "max_template_date": {
            "sw": ["AlphaFold"],
            "method_type": ["coevolution MSA"],
        },
        "model_preset": {
            "sw": ["AlphaFold"],
            "method_type": ["coevolution MSA", "modeling"],
        },
        "num_multimer_predictions_per_model": {
            "sw": ["AlphaFold"],
            "method_type": ["modeling"],
        },
        "plddt_threshold": {
            "sw": ["AlphaPulldown"],
            "method_type": ["coevolution MSA", "modeling"],
        },
        "hb_allowance": {
            "sw": ["AlphaPulldown"],
            "method_type": ["coevolution MSA", "modeling"],
        },
        "threshold_clashes": {
            "sw": ["AlphaPulldown"],
            "method_type": ["coevolution MSA", "modeling"],
        },
        "job_index": {
            "sw": ["AlphaPulldown"],
            "method_type": ["coevolution MSA", "modeling"],
        },
        "use_mmseqs2": {
            "sw": ["AlphaPulldown"],
            "method_type": ["coevolution MSA"],
        },
        "use_hhsearch": {
            "sw": ["AlphaPulldown"],
            "method_type": ["coevolution MSA"],
        },
        "skip_existing": {
            "sw": ["AlphaPulldown"],
            "method_type": ["coevolution MSA", "modeling"],
        },
        "save_msa_files": {
            "sw": ["AlphaPulldown"],
            "method_type": ["coevolution MSA", "modeling"],
        },
        "num_predictions_per_model": {
            "sw": ["AlphaPulldown"],
            "method_type": ["modeling"],
        },
        "benchmark": {
            "sw": ["AlphaFold", "AlphaPulldown"],
            "method_type": ["coevolution MSA", "modeling"],
        },
        "use_precomputed_msas": {
            "sw": ["AlphaFold", "AlphaPulldown"],
            "method_type": ["coevolution MSA", "modeling"],
        },
        "models_to_relax": {
            "sw": ["AlphaFold", "AlphaPulldown"],
            "method_type": ["modeling"],
        },
    }
    trans_args = {
        "num_multimer_predictions_per_model": "num_predictions_per_model"
    }
    ignored_args = [
        "?",
        "alsologtostderr",
        "bfd_database_path",
        "data_dir",
        "delta_threshold",
        "description_file",
        "hbm_oom_exit",
        "hhblits_binary_path",
        "hhsearch_binary_path",
        "hmmbuild_binary_path",
        "hmmsearch_binary_path",
        "jackhmmer_binary_path",
        "kalign_binary_path",
        "log_dir",
        "logger_levels",
        "logtostderr",
        "mgnify_database_path",
        "obsolete_pdbs_path",
        "only_check_args",
        "op_conversion_fallback_to_while_loop",
        "output_dir",
        "fasta_paths",
        "path_to_mmt",
        "pdb",
        "pdb70_database_path",
        "pdb_post_mortem",
        "pdb_seqres_database_path",
        "run_with_pdb",
        "run_with_profiling",
        "runtime_oom_exit",
        "showprefixforinfo",
        "small_bfd_database_path",
        "stderrthreshold",
        "template_mmcif_dir",
        "tt_check_filter",
        "tt_single_core_summaries",
        "uniprot_database_path",
        "uniref30_database_path",
        "uniref90_database_path",
        "use_small_bfd",
        "v",
        "verbosity",
        "xml_output_file",
        "multiple_mmts",
        "protein",
        "compress_features",
    ]
    re_args = re.compile(
        r"(?:fasta_paths|multimeric_chains|multimeric_templates|protein)_\d+"
    )
    swwp = sw_dict  # Software With Parameters
    for key, val in other_dict.items():
        if key in known_args:
            _assemble_params(key, known_args, swwp)
            if key in trans_args:
                key = trans_args[key]
                _assemble_params(key, known_args, swwp)
        else:
            if key not in ignored_args and re.match(re_args, key) is None:
                logging.info(f"Found unknown key in 'other': {key}")
                #sys.exit()

    return swwp


def _get_feature_metadata(
    modelcif_json: dict,
    cmplx_name: str,
    out_dir: list,
) -> Tuple[List[str], List[str]]:
    """Read metadata from a feature JSON file."""
    if "__meta__" not in modelcif_json:
        modelcif_json["__meta__"] = {}
    fasta_dicts = []
    feature_json_files = glob.glob(os.path.join(out_dir, f"*_feature_metadata_*.json"))
    if feature_json_files:
        for feature_json in feature_json_files:
            _file_exists_or_exit(
                feature_json, f"No feature metadata file '{feature_json}' found."
            )
            # ToDo: make sure that its always ASCII
            with open(feature_json, "r", encoding="ascii") as jfh:
                jdata = json.load(jfh)
                #mnmr = jdata["protein"] # For backwards compatibility parse from filename
                mnmr = os.path.basename(feature_json).split("_feature_metadata_")[0]
                modelcif_json["__meta__"][mnmr] = {}
                modelcif_json["__meta__"][mnmr]["databases"] = jdata["databases"]
                modelcif_json["__meta__"][mnmr][
                    "software"
                ] = _get_software_with_parameters(jdata["software"], jdata["other"])
                fp = jdata["other"]["fasta_paths"]
                fp = ast.literal_eval(fp)
                for curr_seq, curr_desc in iter_seqs(fp):
                    new_entry = {'description': curr_desc, 'sequence': curr_seq}
                    if new_entry not in fasta_dicts:
                        fasta_dicts.append(new_entry)

    return cmplx_name, fasta_dicts


def _get_model_info(
    cif_json: dict,
    cmplx_name: str,
    mdl_id: str,
    mdl_rank: int,
) -> None:
    """Get 'data_' block ID and data for categories '_struct' and '_entry'."""
    cif_json["data_"] = cmplx_name
    cif_json["_struct.title"] = f"Prediction for {cmplx_name}"
    cif_json["_struct.pdbx_model_details"] = _get_model_details(
        cmplx_name, cif_json
    )
    cif_json[
        "_ma_model_list.model_name"
    ] = f"Model {mdl_id} (ranked #{mdl_rank})"


def _get_entities(
    cif_json: dict, pdb_file: str, cmplx_name: str, fasta_dicts: List[dict]
) -> BioStructure:
    """Gather data for the mmCIF (target) entities."""
    sequences = {}
    # Process the sequences from the list of dictionaries
    for fasta_dict in fasta_dicts:
        seq = fasta_dict['sequence']
        description = fasta_dict['description']
        # Using MD5 sums for comparing sequences
        sequences[hashlib.md5(seq.encode()).hexdigest()] = description

    # gather molecular entities from PDB file
    structure = PDBParser().get_structure(cmplx_name, pdb_file)
    cif_json["target_entities"] = []
    already_seen = []
    for seq in PPBuilder(radius=999999999).build_peptides(structure, aa_only=False):
        chn_id = seq[0].parent.id
        seq = str(seq.get_sequence())
        seq_md5 = hashlib.md5(seq.encode()).hexdigest()
        cif_ent = {}
        try:
            e_idx = already_seen.index(seq_md5)
        except ValueError:
            pass
        else:
            cif_json["target_entities"][e_idx]["pdb_chain_id"].append(chn_id)
            continue
        cif_ent["pdb_sequence"] = seq
        cif_ent["pdb_chain_id"] = [chn_id]
        if seq_md5 in sequences:
            cif_ent["description"] = sequences[seq_md5]
        else:
            cif_ent["description"] = "Unknown"
        cif_json["target_entities"].append(cif_ent)
        already_seen.append(seq_md5)

    return structure


def _get_scores(cif_json: dict, scr_file: str) -> None:
    """Add scores to JSON data."""
    # Read from jsons instead
    mdl_name = scr_file.split('result_')[1].split('.pkl')[0]
    output_dir = os.path.dirname(scr_file)
    with open(os.path.join(output_dir, f"confidence_{mdl_name}.json"), 'r') as f:
        plddt = json.load(f)["confidenceScore"]
        cif_json["plddt"] = plddt
    with open(os.path.join(output_dir, "ranking_debug.json"), 'r') as f:
        ranking = json.load(f)
        # Multimer
        if "iptm+ptm" in ranking:
            iptm_ptm = ranking["iptm+ptm"][mdl_name]
            cif_json["iptm+ptm"] = iptm_ptm
            iptm = ranking["iptm"][mdl_name]
            cif_json["iptm"] = iptm
        # Monomer
        elif "ptm" in ranking:
            ptm = ranking["ptm"][mdl_name]
            cif_json["ptm"] = ptm
        else:
            raise RuntimeError("No PTM scores found in ranking_debug.json")
    with open(os.path.join(output_dir, f"pae_{mdl_name}.json"), 'r') as f:
        pae = json.load(f)[0]["predicted_aligned_error"]
        cif_json["pae"] = pae
    #with open(scr_file, "rb") as sfh:
     #   scr_dict = pickle.load(sfh)
    # Get pLDDT as a list, the global pLDDT is the average, calculated on the
    # spot.
    #cif_json["plddt"] = scr_dict["plddt"]
    #cif_json["ptm"] = float(scr_dict["ptm"])
    #cif_json["iptm"] = float(scr_dict["iptm"])
    #cif_json["pae"] = scr_dict["predicted_aligned_error"]


def _get_software_data(meta_json: dict) -> list:
    """Turn meta data about software into `modelcif.Software()` objects."""
    cite_hhsuite = ihm.Citation(
        pmid="31521110",
        title="HH-suite3 for fast remote homology detection and deep "
        + "protein annotation.",
        journal="BMC Bioinformatics",
        volume=20,
        page_range=None,
        year=2019,
        authors=[
            "Steinegger, M.",
            "Meier, M.",
            "Mirdita, M.",
            "Voehringer, H.",
            "Haunsberger, S.J.",
            "Soeding, J.",
        ],
        doi="10.1186/s12859-019-3019-7",
    )

    # pylint: disable=too-few-public-methods
    class _HHsuiteSW(modelcif.Software):
        """Prefilled software object for HH-suite tools."""

        # We keep the parameter names from the parent class here, so let Pylint
        # ignore redefining the 'type' builtin.
        # pylint: disable=redefined-builtin

        def __init__(
            self,
            name,
            classification="data collection",
            description="Iterative protein sequence searching by HMM-HMM "
            + "alignment",
            location="https://github.com/soedinglab/hh-suite",
            type="program",
            version=None,
            citation=cite_hhsuite,
        ):
            """Initialise a model"""
            super().__init__(
                name,
                classification,
                description,
                location,
                type,
                version,
                citation,
            )

    class _HmmerSW(modelcif.Software):
        """Prefilled software object for HMMER tools."""

        # We keep the parameter names from the parent class here, so let Pylint
        # ignore redefining the 'type' builtin.
        # pylint: disable=redefined-builtin

        def __init__(
            self,
            name,
            classification="data collection",
            description="Building HMM search profiles",
            location="http://hmmer.org/",
            type="program",
            version=None,
            citation=None,
        ):
            """Initialise a model"""
            super().__init__(
                name,
                classification,
                description,
                location,
                type,
                version,
                citation,
            )

    # pylint: enable=too-few-public-methods

    # {key from JSON: dict needed to produce software entry plus internal key}
    sw_data = {
        "AlphaFold": modelcif.Software(
            "AlphaFold-Multimer",
            "model building",
            "Structure prediction",
            "https://github.com/deepmind/alphafold",
            "package",
            None,
            ihm.Citation(
                pmid=None,
                title="Protein complex prediction with AlphaFold-Multimer.",
                journal="bioRxiv",
                volume=None,
                page_range=None,
                year=2021,
                authors=[
                    "Evans, R.",
                    "O'Neill, M.",
                    "Pritzel, A.",
                    "Antropova, N.",
                    "Senior, A.",
                    "Green, T.",
                    "Zidek, A.",
                    "Bates, R.",
                    "Blackwell, S.",
                    "Yim, J.",
                    "Ronneberger, O.",
                    "Bodenstein, S.",
                    "Zielinski, M.",
                    "Bridgland, A.",
                    "Potapenko, A.",
                    "Cowie, A.",
                    "Tunyasuvunakool, K.",
                    "Jain, R.",
                    "Clancy, E.",
                    "Kohli, P.",
                    "Jumper, J.",
                    "Hassabis, D.",
                ],
                doi="10.1101/2021.10.04.463034",
            ),
        ),
        "AlphaPulldown": modelcif.Software(
            "AlphaPulldown",
            "model building",
            "Structure prediction",
            "https://github.com/KosinskiLab/AlphaPulldown",
            "package",
            None,
            ihm.Citation(
                pmid="36413069",
                title="AlphaPulldown-a python package for protein-protein "
                + "interaction screens using AlphaFold-Multimer.",
                journal="Bioinformatics",
                volume=39,
                page_range=None,
                year=2023,
                authors=[
                    "Yu, D.",
                    "Chojnowski, G.",
                    "Rosenthal, M.",
                    "Kosinski, J.",
                ],
                doi="10.1093/bioinformatics/btac749",
            ),
        ),
        "hhblits": _HHsuiteSW("HHblits"),
        "hhsearch": _HHsuiteSW(
            "HHsearch",
            description="Protein sequence searching by HMM-HMM comparison",
        ),
        "hmmbuild": _HmmerSW("hmmbuild"),
        "hmmsearch": _HmmerSW(
            "hmmsearch",
            description="Search profile(s) against a sequence database",
        ),
        "jackhmmer": _HmmerSW(
            "jackhmmer",
            description="Iteratively search sequence(s) against a sequence "
            + "database",
        ),
        "kalign": modelcif.Software(
            "kalign",
            "data collection",
            "Kalign is a fast multiple sequence alignment program for "
            + "biological sequences",
            "https://github.com/timolassmann/kalign",
            "program",
            None,
            ihm.Citation(
                pmid="31665271",
                title="Kalign 3: multiple sequence alignment of large data "
                + "sets",
                journal="Bioinformatics",
                volume=36,
                page_range=(1928, 1929),
                year=2019,
                authors=["Lassmann, T."],
                doi="10.1093/bioinformatics/btz795",
            ),
        ),
    }
    # ToDo: refactor to only those SW objects created/ added that are actually
    #       in the dictionary. That is, instead of a pre-build dictionary,
    #       instantiate on the fly, there is anyway a hard-coded-tool-name.
    for data in meta_json.values():
        for sftwr, version in data["software"].items():
            if sftwr not in sw_data:
                raise RuntimeError(
                    "Unknown software found in meta data: " + f"'{sftwr}'"
                )
            version = version["version"]
            # ToDo: software should not be None, remove in final version
            if sw_data[sftwr] is not None:
                if sw_data[sftwr].version is not None:
                    if sw_data[sftwr].version != version:
                        raise RuntimeError(
                            "Software versions differ for "
                            + f"'{sftwr}': '{sw_data[sftwr].version}' vs. "
                            + f"'{version}'"
                        )
                sw_data[sftwr].version = version

    return sw_data


def _get_protocol_steps(modelcif_json):
    """Create the list of protocol steps with software and parameters used."""
    # ToDo: Get software_group from external input, right now the protocol steps
    #       are hard-coded here with the software per step. The JSON input does
    #       not list steps, only software.
    protocol = []
    # MSA/ monomer feature generation step
    # ToDo: Discuss input, manual has baits & sequences
    step = {
        "method_type": "coevolution MSA",
        "step_name": "MSA generation",
        "details": "Create sequence features for corresponding monomers.",
        "input_data_group": ["target_sequences", "reference_dbs"],
        "output_data_group": "monomer_pickle_files",
        "software_group": [],
        "parameter_group": [],
    }
    for sftwr in modelcif_json["__meta__"].values():
        sftwr = sftwr["software"]
        for tool in sftwr:
            if tool not in step["software_group"]:
                step["software_group"].append(tool)
                if step["method_type"] in sftwr[tool]:
                    step["parameter_group"].append(
                        sftwr[tool][step["method_type"]]
                    )
                else:
                    step["parameter_group"].append({})
            # else:
            #     pos = step["software_group"].index(tool)
            #     if step["method_type"] in sftwr[tool]:
            #         params = sftwr[tool][step["method_type"]]
            #     else:
            #         params = {}
            #     # always raises an error due to different --job_index!
            #     #if step["parameter_group"][pos] != params:
            #     #    raise RuntimeError(
            #     #        f"Different parameters/ values for {tool}."
            #     #    )

    protocol.append(step)

    # modelling step
    # ToDo: Discuss input, seem to depend on mode
    # ToDo: what about step details? Would it be nice to add the AlphaPulldown
    #       mode here?
    m_type = "modeling"
    step = {
        "method_type": m_type,
        "step_name": None,
        "details": None,
        "input_data_group": ["target_sequences", "STEPTYPE$coevolution MSA"],
        "output_data_group": "model",
        "software_group": ["AlphaPulldown", "AlphaFold"],
        "parameter_group": [],
    }
    for sftwr in modelcif_json["__meta__"].values():
        sftwr = sftwr["software"]
        for i, tool in enumerate(["AlphaPulldown", "AlphaFold"]):
            if i >= len(step["parameter_group"]):
                step["parameter_group"].append(sftwr[tool][m_type])
            # always raises an error due to different --job_index!
            #else:
            #    if step["parameter_group"][i] != sftwr[tool][m_type]:
            #        raise RuntimeError(
            #            f"Different parameters/ values for {tool}."
            #        )

    protocol.append(step)

    # model selection step <- ask if there is automated selection, if only
    # manual, skip this step here?

    # ToDo: Example 1 in the GitHub repo has a 3rd step: "Evaluation and
    #       visualisation"

    return protocol


# def _collect_monomer_dictionary(monomer_objects_dir):
#     """
#     a function to gather all monomers across different monomer_objects_dir
#
#     args
#     monomer_objects_dir: a list of directories where monomer objects are stored, given by FLAGS.monomer_objects_dir
#     """
#     output_dict = dict()
#     for dir in monomer_objects_dir:
#         monomers = glob.glob(f"{dir}/*.pkl")
#         for m in monomers:
#             output_dict[m] = dir
#     return output_dict


def alphapulldown_model_to_modelcif(
    cmplx_name: str,
    mdl: tuple,
    out_dir: str,
    compress: bool = False,
    additional_assoc_files: list = None,
) -> None:
    """Convert an AlphaPulldown model into a ModelCIF formatted mmCIF file.

    Metadata for the ModelCIF categories will be fetched from AlphaPulldown
    output as far as possible. This expects modelling projects to exists in
    AlphaPulldown's output directory structure."""
    # ToDo: ENABLE logging.info(f"Processing '{mdl[0]}'...")
    modelcif_json = {}
    # fetch metadata
    cmplx_name, fasta_dicts = _get_feature_metadata(
        modelcif_json, cmplx_name, out_dir
    )
    # fetch/ assemble more data about the modelling experiment
    _get_model_info(
        modelcif_json,
        cmplx_name,
        mdl[2],
        mdl[3],
    )
    # gather target entities (sequences that have been modeled) info
    structure = _get_entities(modelcif_json, mdl[0], cmplx_name, fasta_dicts)

    # read quality scores from pickle file
    _get_scores(modelcif_json, mdl[1])

    modelcif_json["ma_protocol_step"] = _get_protocol_steps(modelcif_json)
    cfs = _store_as_modelcif(
        modelcif_json,
        structure,
        mdl[0],
        out_dir,
        compress,
        additional_assoc_files,
    )
    # ToDo: ENABLE logging.info(f"... done with '{mdl[0]}'")
    return cfs


def _add_mdl_to_list(mdl, model_list, mdl_path, score_files):
    """Fetch info from file name to add to list"""
    rank = re.match(r"ranked_(\d+)\.pdb", mdl)
    if rank is not None:
        rank = int(rank.group(1))
        model_list.append(
            (
                os.path.join(mdl_path, mdl),
                score_files[rank][0],
                score_files[rank][1],  # model ID
                score_files[rank][2],  # model rank
            )
        )


def _get_model_list(
    ap_dir: str, model_selected: str, get_non_selected: bool
) -> List[Dict[str, List[Any] | Any]]:
    """Get the list of models to be converted.

    If `model_selected` is none, all models will be marked for conversion."""
    if 'ranking_debug.json' in os.listdir(ap_dir): #One complex was given, not the root directory
        cmplx = [os.path.basename(os.path.normpath(ap_dir))]
        mdl_all_paths = [ap_dir]
    else:
        cmplx = [d for d in os.listdir(ap_dir) if os.path.isdir(os.path.join(ap_dir, d))]
        mdl_all_paths = [os.path.join(ap_dir, c) for c in cmplx]
    result = []

    for c, specific_mdl_path in zip(cmplx, mdl_all_paths):
        models = []
        # We are going for models with name "rank_?.pdb" as it does not depend on
        # relaxation. But this means we have to match the pickle files via
        # ranking_debug.json.
        ranking_dbg = os.path.join(specific_mdl_path, "ranking_debug.json")
        _file_exists_or_exit(
            ranking_dbg,
            f"Ranking file '{ranking_dbg} does not exist or is no regular file.",
        )
        with open(ranking_dbg, "r", encoding="utf8") as jfh:
            ranking_dbg = json.load(jfh)
        score_files = {}
        for i, fle in enumerate(ranking_dbg["order"]):
            if not fle.startswith("model_"):
                raise RuntimeError(
                    "Filename does not start with 'model_', can "
                    + f"not determine model ID: '{fle}'"
                )
            score_files[i] = (
                os.path.join(specific_mdl_path, f"result_{fle}.pkl"),
                fle.split("_")[1],
                i,
            )

        not_selected_models = []
        if model_selected is not None:
            if model_selected not in score_files:
                logging.info(f"Model of rank {model_selected} not found.")
                sys.exit()

            _add_mdl_to_list(
                f"ranked_{model_selected}.pdb", models, specific_mdl_path, score_files
            )
            if get_non_selected:
                for mdl in os.listdir(specific_mdl_path):
                    if mdl == f"ranked_{model_selected}.pdb":
                        continue
                    _add_mdl_to_list(
                        mdl, not_selected_models, specific_mdl_path, score_files
                    )
        else:
            for mdl in os.listdir(specific_mdl_path):
                _add_mdl_to_list(mdl, models, specific_mdl_path, score_files)

        # check that files actually exist
        for mdl, scrs, *_ in models:
            _file_exists_or_exit(
                mdl, f"Model file '{mdl}' does not exist or is not a regular file."
            )
            _file_exists_or_exit(
                scrs,
                f"Scores file '{scrs}' does not exist or is not a regular file.",
            )
        result.append(
            {'complex': c, 'path': specific_mdl_path, 'models': models, 'not_selected': not_selected_models}
        )

    return result


def main(argv):
    """Run as script."""
    # pylint: disable=pointless-string-statement
    """
    Here, the metadata json files for each feature are in features_monomers/
    directory. The models are in models/ directory, and usually there are many
    complexes modelled using different permutations of the monomeric features.
    For the sake of size, I send you the models of only one dimer
    cage_B_and_cage_C/ that was generated using features_monomers/cage_B.pkl
    and features_monomers/cage_C.pkl accordingly.
    Please note that typically all the cage_?_feature_metadata.json files are
    identical in terms of used databases and software versions and generated
    in one go.
    However, theoretically they could be generated using different binaries/DBs
    versions, so maybe it makes sense to compare them and store both/all
    versions if they are different. This merging can be done on our
    AlphaPulldown side and may be added now or later on. Let me know if it is
    critical for you now.
    """
    # pylint: enable=pointless-string-statement
    del argv  # Unused.

    # get list of selected models and assemble ModelCIF files + associated data
    models = _get_model_list(
        FLAGS.ap_output,
        FLAGS.model_selected,
        FLAGS.add_associated,
    )
    for d in models:
        complex_name = d['complex']
        model_list = d['models']
        not_selected = d['not_selected']
        model_dir = d['path']
        add_assoc_files = {}
        try:
            if len(not_selected) > 0:
                # pylint: disable=consider-using-with
                ns_tmpdir = tempfile.TemporaryDirectory(suffix="_modelcif")
                for mdl in not_selected:
                    add_assoc_files.update(
                        alphapulldown_model_to_modelcif(
                            complex_name,
                            mdl,
                            ns_tmpdir.name,
                            FLAGS.compress,
                        )
                    )
            for mdl in model_list:
                alphapulldown_model_to_modelcif(
                    complex_name,
                    mdl,
                    model_dir,
                    FLAGS.compress,
                    add_assoc_files,
                )
        except Exception as exc:
            logging.error(
                f"Error while processing model '{mdl[0]}' of complex "
                + f"'{complex_name}': {exc}"
            )


if __name__ == "__main__":
    app.run(main)

# ToDo: Things to look at: '_struct.title', '_struct.pdbx_model_details',
#       'data_', '_entry', maybe have a user-defined JSON document with things
#       like that, including author names?
# ToDo: where to store which model was chosen? Should be in Tara's models.
# ToDo: make sure all functions come with types

# From former discussions:
# - including Jupyter notebooks would require adding the pickle files to the
#   associated files (too much storage needed for that)
