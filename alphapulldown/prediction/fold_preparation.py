"""Turn the interactors of one fold into the object to model, and its output directory.

Both the single-fold command and the resident batch command need this, and they used to
carry near-identical copies that had drifted apart: the copy in the single-fold path
reassigned its metadata glob inside the feature-directory loop and tested it outside, so
only the last feature directory could contribute. The same fold produced different output
depending on which command ran it. One implementation, one meaning.
"""

from __future__ import annotations

import glob
import lzma
import os
import pickle
import shutil
from typing import Any, List, Tuple, Union

from absl import logging

from alphapulldown.objects import ChoppedObject, MonomericObject, MultimericObject


# Most filesystems reject paths longer than this.
_MAX_PATH_LENGTH = 4096


def _interactor_description(interactor: Any) -> str | None:
    """Name under which this interactor's feature metadata was written, if any."""
    if isinstance(interactor, ChoppedObject):
        return interactor.monomeric_description
    if isinstance(interactor, MonomericObject):
        return interactor.description
    return None


def _output_directory_for(description: str, output_dir: str) -> str:
    """AlphaPulldown-style output directory, collapsing repeated chains to `_homo_Ner`."""
    oligomers = description.split("_and_")
    if len(oligomers) == len(set(oligomers)):
        return os.path.join(output_dir, description)
    fragments = []
    for oligomer in dict.fromkeys(oligomers):
        count = oligomers.count(oligomer)
        fragments.append(oligomer if count == 1 else f"{oligomer}_homo_{count}er")
    return os.path.join(output_dir, "_and_".join(fragments))


def _copy_feature_metadata(interactors: List[Any], flags: Any, output_dir: str) -> None:
    """Copy each interactor's most recent feature metadata, decompressing `.json.xz`.

    Every configured feature directory is considered, not just the last one, and the
    newest match across all of them wins.
    """
    for interactor in interactors:
        description = _interactor_description(interactor)
        if description is None:
            continue
        metadata_files: list[str] = []
        for feature_dir in flags.features_directory:
            metadata_files.extend(
                glob.glob(
                    os.path.join(
                        feature_dir, f"{description}_feature_metadata_*.json*"
                    )
                )
            )
        if not metadata_files:
            logging.warning(
                "No feature metadata found for %s in %s",
                description,
                ", ".join(map(str, flags.features_directory)),
            )
            continue
        latest = max(metadata_files, key=os.path.getmtime)
        destination = os.path.join(output_dir, os.path.basename(latest))
        if latest.endswith(".json.xz"):
            destination = destination[: -len(".xz")]
            logging.info("Decompressing %s to %s", latest, destination)
            with lzma.open(latest, "rb") as source, open(destination, "wb") as target:
                target.write(source.read())
        else:
            logging.info("Copying %s to %s", latest, output_dir)
            shutil.copyfile(latest, destination)


def prepare_fold(
    interactors: List[Union[MonomericObject, ChoppedObject]],
    output_dir: str,
    flags: Any,
) -> Tuple[Union[MultimericObject, MonomericObject, ChoppedObject], str]:
    """Build the object to model for one fold and create its output directory.

    ``flags`` is the parsed invocation (absl ``FLAGS`` or an equivalent), read for
    ``pair_msa``, the multimeric-template settings, ``save_features_for_multimeric_object``,
    ``use_ap_style`` and ``features_directory``.
    """
    if (
        len(interactors) > 1
        and flags.pair_msa
        and any(getattr(interactor, "skip_msa", False) for interactor in interactors)
    ):
        raise ValueError(
            "--skip_msa generates query-only MSAs and cannot be combined with "
            "--pair_msa=True. Re-run structure prediction with --pair_msa=False."
        )

    if len(interactors) > 1:
        object_to_model = MultimericObject(
            interactors=interactors,
            pair_msa=flags.pair_msa,
            multimeric_template=flags.multimeric_template,
            multimeric_template_meta_data=flags.description_file,
            multimeric_template_dir=flags.path_to_mmt,
            threshold_clashes=flags.threshold_clashes,
            hb_allowance=flags.hb_allowance,
            plddt_threshold=flags.plddt_threshold,
        )
        if flags.save_features_for_multimeric_object:
            with open(
                os.path.join(output_dir, "multimeric_object_features.pkl"), "wb"
            ) as handle:
                pickle.dump(MultimericObject.feature_dict, handle)
    else:
        object_to_model = interactors[0]
        object_to_model.input_seqs = [object_to_model.sequence]

    if flags.use_ap_style:
        output_dir = _output_directory_for(object_to_model.description, output_dir)

    if len(output_dir) > _MAX_PATH_LENGTH:
        logging.warning(
            "Output directory path is too long: %s. "
            "Please use a shorter path with --output_directory.",
            output_dir,
        )
    os.makedirs(output_dir, exist_ok=True)

    _copy_feature_metadata(interactors, flags, output_dir)
    return object_to_model, output_dir
