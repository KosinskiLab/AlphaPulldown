#
# Author: Grzegorz Chojnowski @ EMBL-Hamburg
#

import os, sys, re
import pathlib
import glob
import time
import json
import pickle as pkl

from multiprocessing import Pool, Process
import multiprocessing
import matplotlib.pyplot as plt

import py3Dmol
import numpy as np

sys.path.append(os.path.dirname(__file__))
from cctbx_imps import *
from scitbx.math import superpose
from scitbx import matrix
from iotbx.bioinformatics import any_sequence_format

from IPython.display import Image, FileLink, FileLinks
from IPython.core.display import Image, display

import pickle
import subprocess

SLURM_MAIL = "--mail-user $USER@embl-hamburg.de --mail-type=FAIL --mail-type=END"

# --amber
# SLURM_TEMPLATE_COLABFOLD="""sbatch -p gpu -p gpu-long --qos=gpu-long --gres=gpu:1 -n 8 --mem 40gb --job-name=%(job_name)s --output %(output_dir)s/output.log %(mail_str)s /cryo_em/AlphaFold/scripts/run_af2_colabfold.sh %(relax_string)s --num-models=%(num_models)i %(fasta_fn)s %(output_dir)s"""

WARNING = """

*******************************************************************************
*****************************  ERRROR  ****************************************
*******************************************************************************

ERROR: your job didn't start. Contact IT at cg@embl-hamburg.de to check if
        you have enough permissions to use GPU nodes on HYDE cluster

*******************************************************************************

"""

SLURM_TEMPLATE_COLABFOLD = """sbatch -p gpu -p gpu-long --qos=gpu-long --gres=gpu:1 -n 8 --mem %(mem)igb --job-name=%(job_name)s --output %(output_dir)s/output.log %(mail_str)s /cryo_em/AlphaFold/scripts/run_af2_colabfold_devel.sh %(relax_string)s --model-type=AlphaFold2-ptm --num-models=%(num_models)i %(fasta_fn)s %(output_dir)s"""

# --amber
SLURM_TEMPLATE_AF2MULTI = """sbatch -p gpu -p gpu-long --qos=gpu-long --gres=gpu:1 -n 8 --mem %(mem)igb --job-name=%(job_name)s --output %(output_dir)s/output.log %(mail_str)s /cryo_em/AlphaFold/scripts/run_af2_multimer.sh %(relax_string)s --fasta_paths=%(fasta_fn)s --output_dir=%(output_dir)s"""

SLURM_TEMPLATE_AF2MONO = """sbatch -p gpu -p gpu-long --qos=gpu-long --gres=gpu:1 -n 8 --mem %(mem)igb --job-name=%(job_name)s --output %(output_dir)s/output.log %(mail_str)s /cryo_em/AlphaFold/scripts/run_af2_monomer.sh %(relax_string)s --fasta_paths=%(fasta_fn)s --output_dir=%(output_dir)s"""
# ------------------------------------------------------


def AF2_run_and_parse(
    sequence,
    output,
    colabfold=False,
    msa_file=None,
    amber=False,
    color=None,
    models=3,
    force_multimer=False,
    num_models=5,
    mail=False,
    job_name=None,
    mem=30,
):
    """
    gchojnowski@embl-hamburg.de

    last update: 6.12.2021
    Arguments:
        * sequence   = FASTA string; single chain for a monomer,
                                   multiple entries for a complex e.g.
                                      >1
                                      ABC
                                      >2
                                      DEF
        * output     = directory (will be ignored if already exists)
        * colabfold  = [False|True] run colabfold instead of AF2-multimer
                                    (uses much faster MMseqs2 instead og HMMER)
        * msa_file   = input alignment in a3m format (with .a3m extension!)
                       if given in colabfold mode, will be used instead insted
                       input sequence (ignored otherwise!)
        * amber      = [False|True] enable output model regularization
        * color      = 'lDDT'[default] or 'chain'
                       in chain mode top ranked complex color-coded
                       by lDDT and chain ids is shown
                       for monomers top 3 predictions are shown side-by-side
        * num_modles = default 5, can be reduced for ColabFold only!
        * mail       = [False|True] send an email when job completes
        * job_name   = unique job name for slurm [default af2cfold or af2mmer]
        * mem        = [default 30] requested memory in GB. can be increased up to 120

    """

    print("*** Checking output directory: %s\n" % output)
    jobid = slurm_job_running(output)
    if jobid > 0:
        print("Job %i is running" % jobid)
        print("\n")
        print("*** Log file:\n")
        print(get_log(output))

        return

    mem_adj = min(120, mem)
    if mem_adj < mem:
        print(
            "\n\n*** WARNING reduced requested memory to %(mem_adj)iGB\n\n" % locals()
        )

    if colabfold:
        output_dir = run_colabfold(
            sequence=sequence,
            msa_file=msa_file,
            output=output,
            job_name=job_name if job_name else "af2cfold",
            amber=amber,
            num_models=num_models,
            mail=mail,
            mem=mem_adj,
        )
    else:
        output_dir = run_af2multimer(
            sequence=sequence,
            output=output,
            job_name=job_name if job_name else "af2mmer",
            amber=amber,
            force_multimer=force_multimer,
            mail=mail,
            mem=mem_adj,
        )

    ret = parse_results(output, color=color, models=models)

    if ret and not jobid:
        print("*** Log file:\n")
        log = get_log(output)
        print("\n".join(log.splitlines()[-10:]))


# ------------------------------------------------------


def make_output_director(output):
    output_dir = os.path.expanduser(output)

    try:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=False)
    except:
        print(
            "*** Cannot start a job, output directory already exists: %s\n" % output_dir
        )
        return None

    print("Created output directory: ", output_dir)
    return output_dir


# ------------------------------------------------------


def get_log(output):

    output_dir = os.path.expanduser(output)

    try:
        with open(os.path.join(output_dir, "output.log"), "r") as ifile:
            return ifile.read()
    except:
        return ""


# ------------------------------------------------------


def slurm_job_running(output, jobid=None):

    output_dir = os.path.expanduser(output)

    if jobid:
        slurm_jobid = jobid
    else:
        try:
            with open(os.path.join(output_dir, "SLURM"), "r") as ifile:
                slurm_jobid = int(ifile.read())
        except:
            return 0

    ppipe = subprocess.Popen(
        "squeue --job %i| grep -v JOBID" % slurm_jobid,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    sto = []
    for stdout_line in iter(ppipe.stdout.readline, ""):
        print(stdout_line)
        sto.append(stdout_line)

    retcode = subprocess.Popen.wait(ppipe)

    if sto:
        return slurm_jobid

    return 0


# ------------------------------------------------------


def submit2slurm(shell_cmd, output_dir):

    template = r"Submitted batch job (?P<slurm_jobid>\d+)$"
    ppipe = subprocess.Popen(
        shell_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    slurm_jobid = None
    for stdout_line in iter(ppipe.stdout.readline, ""):
        m = re.match(template, stdout_line)
        if m:
            slurm_jobid = int(m.group("slurm_jobid"))

    if slurm_jobid is None:
        return -999

    print("Submitted job with id %i" % slurm_jobid)
    with open(os.path.join(output_dir, "SLURM"), "w") as of:
        of.write(str(slurm_jobid))

    return subprocess.Popen.wait(ppipe)


# ------------------------------------------------------


def run_colabfold(
    sequence,
    output,
    msa_file=None,
    job_name="af2",
    amber=False,
    num_models=5,
    mail=False,
    mem=30,
):

    output_dir = make_output_director(output)
    if not output_dir:
        return

    fasta_fn = os.path.join(output_dir, "input.fasta")

    _seq, err = any_sequence_format(file_name="weird.fasta", data=sequence)

    if msa_file is None:
        colabfold_fasta_name = "_".join(["X" if not _.name else _.name for _ in _seq])
        colabfold_fasta_seq = ":".join([_.sequence for _ in _seq])

        with open(fasta_fn, "w") as ofile:
            ofile.write(">%s\n%s" % (colabfold_fasta_name, colabfold_fasta_seq))

    else:
        print("Using input MSA: ", msa_file)
        fasta_fn = msa_file

    relax_string = "--amber" if amber else ""

    mail_str = ""
    if mail:
        mail_str = SLURM_MAIL

    shell_cmd = SLURM_TEMPLATE_COLABFOLD % locals()
    print(shell_cmd)

    retcode = submit2slurm(shell_cmd, output_dir)
    if retcode < 0:
        print(WARNING)

    return output_dir


# ------------------------------------------------------


def run_af2multimer(
    sequence,
    output,
    job_name="af2",
    amber=False,
    force_multimer=False,
    mail=False,
    mem=30,
):

    output_dir = make_output_director(output)
    if not output_dir:
        return

    fasta_fn = os.path.join(output_dir, "input.fasta")

    _seq, err = any_sequence_format(file_name="weird.fasta", data=sequence)

    monomer = True
    if len(_seq) > 1:
        monomer = False
    with open(fasta_fn, "w") as ofile:
        for entry in _seq:
            ofile.write(
                ">%s\n%s\n" % ("X" if not entry.name else entry.name, entry.sequence)
            )

    relax_string = "--amber" if amber else ""
    mail_str = ""
    if mail:
        mail_str = SLURM_MAIL

    if monomer and not force_multimer:
        print(" *** Starting AF2-MONOMER")
        shell_cmd = SLURM_TEMPLATE_AF2MONO % locals()
    else:
        print(" *** Starting AF2-MULTIMER")
        shell_cmd = SLURM_TEMPLATE_AF2MULTI % locals()
    print(shell_cmd)

    retcode = submit2slurm(shell_cmd, output_dir)
    if retcode < 0:
        print(WARNING)

    return output_dir


# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------


def plot_msa_coverage(datadir, msa_file=None):
    def parse_msa(msa):
        query_sequence = msa[0]
        seq_len_list = [len(msa[0])]
        prev_pos = 0
        lines = []
        Ln = np.cumsum(np.append(0, seq_len_list))

        for l in seq_len_list:
            chain_seq = np.array(list(query_sequence[prev_pos : prev_pos + l]))
            chain_msa = np.array([_[prev_pos : prev_pos + l] for _ in msa])
            seqid = np.array(
                [
                    np.count_nonzero(
                        np.array(chain_seq) == msa_line[prev_pos : prev_pos + l]
                    )
                    / len(chain_seq)
                    for msa_line in msa
                ]
            )
            non_gaps = (chain_msa != "-").astype(float)

            non_gaps[non_gaps == 0] = np.nan

            # order by decreasing sequence identity
            new_order = np.argsort(seqid, axis=0)[::-1]
            lines.append((non_gaps[:] * seqid[:, None])[new_order])
            prev_pos += l

        lines = np.concatenate(lines, 1)
        return lines, Ln

    # --------------------------------------------------

    if msa_file:
        a3m_filenames = [msa_file]
    else:
        a3m_filenames = glob.glob("%s/msas/*/*.a3m" % datadir)
        # try for monomers
        if not a3m_filenames:
            a3m_filenames = glob.glob("%s/msas/*.a3m" % datadir)

        if not a3m_filenames:
            print("No a3m files found!")
            return

    chain_descr_dict = None
    try:
        with open("%s/msas/chain_id_map.json" % datadir, "r") as ifile:
            chain_descr_dict = json.loads(ifile.read())
    except:
        pass

    msas = []
    for fn in a3m_filenames:
        chid = os.path.dirname(fn).split("/")[-1]
        print()
        with open(fn, "r") as ifile:
            _msa, err = any_sequence_format(file_name="weird.fasta", data=ifile.read())
        # print("# sequences in MSA %s " % os.path.basename(fn),len(_msa)  )
        if chain_descr_dict:
            msas.append(
                (
                    chain_descr_dict[chid].get("description", "UNK"),
                    np.array([list(_.sequence.strip()) for _ in _msa], dtype=object),
                )
            )
        else:
            msas.append(
                (
                    "UNK",
                    np.array([list(_.sequence.strip()) for _ in _msa], dtype=object),
                )
            )

    # plotting stuff
    plt.figure(figsize=(6 * len(msas), 4), dpi=100)

    for idx, (name, msa) in enumerate(msas):

        plt.subplot(1, len(msas), idx + 1)

        lines, Ln = parse_msa(msa)

        # plt.figure(figsize=(5, 3), dpi=100)
        plt.title("Sequence coverage\n%s" % name)
        plt.imshow(
            lines[::-1],
            interpolation="nearest",
            aspect="auto",
            cmap="rainbow_r",
            vmin=0,
            vmax=1,
            origin="lower",
            extent=(0, lines.shape[1], 0, lines.shape[0]),
        )
        for i in Ln[1:-1]:
            plt.plot([i, i], [0, lines.shape[0]], color="black")

        plt.plot((np.isnan(lines) == False).sum(0), color="black")
        plt.xlim(0, lines.shape[1])
        plt.ylim(0, lines.shape[0])

        plt.xlabel("Positions")
        plt.ylabel("Sequences")

    plt.colorbar(label="Sequence identity to query")

    plt.show()


# ------------------------------------------------------


def parse_colabfold(datadir, color="lDDT", models=3):
    png_fnames = glob.glob("%s/*.png" % datadir)

    for fn in png_fnames:
        display(Image(filename=fn))
        # display(FileLink(os.path.join('/view/jupter_farm/', '/../../', fn)))
        # print('/view/jupter_farm/../../'+fn)

    relaxed = sorted(
        glob.glob("%s/*_relaxed_*.pdb" % datadir),
        reverse=False,
        key=lambda x: int(x[:-4].split("_")[-1]),
    )
    if relaxed:
        return relaxed
    return sorted(
        glob.glob("%s/*_unrelaxed_*.pdb" % datadir),
        reverse=False,
        key=lambda x: int(x[:-4].split("_")[-1]),
    )


# ------------------------------------------------------


def plot_predicted_alignment_error(datadir):
    seqs = []
    if os.path.isfile(os.path.join(datadir, "concatenated.fasta")):

        with open(datadir + "/concatenated.fasta", "r") as infile:
            lines = list(infile.readlines())
            for i in range(len(lines)):
                if ">" not in lines[i]:
                    seqs.append(lines[i].rstrip())

    with open(datadir + "/ranking_debug.json", "r") as infile:
        data = json.load(infile)

    order = data["order"]
    outs = dict()
    for i in order:
        prediction_result = pkl.load(open(datadir + "/result_{}.pkl".format(i), "rb"))
        outs[i] = prediction_result

    # %%
    if len(seqs) > 0:
        xticks = []
        initial_tick = 0
        for s in seqs:
            initial_tick = initial_tick + len(s)
            xticks.append(initial_tick)

        xticks_labels = []
        for i, t in enumerate(xticks):
            xticks_labels.append(str(i + 1))

        plt.figure(figsize=(3 * 5, 2), dpi=100)
        for i in range(len(order)):
            check = outs[order[i]]["predicted_aligned_error"]
            plt.subplot(1, 5, i + 1)
            plt.title("ranked_{}:{}".format(i, order[i]))
            plt.imshow(check, cmap="bwr", vmin=0, vmax=30)
            plt.xticks(xticks, labels=[1, 2])
            plt.yticks(xticks)
            for t in xticks:
                plt.axhline(t, color="black", linewidth=3.5)
                plt.axvline(t, color="black", linewidth=3.5)
        plt.show()
    else:
        plt.figure(figsize=(3 * 5, 2), dpi=100)
        for i in range(len(order)):
            check = outs[order[i]]["predicted_aligned_error"]
            plt.subplot(1, 5, i + 1)
            plt.title("ranked_{}:{}".format(i, order[i]))
        plt.show()


# ------------------------------------------------------


def plot_plddts(datadict, Ls=None, dpi=100, fig=True):
    plt.figure(figsize=(6, 4), dpi=100)
    plt.title("Predicted lDDT per position")
    for idx, k in enumerate(sorted(datadict, key=lambda x: datadict[x]["rank"])):
        plt.plot(
            datadict[k]["plddt"],
            label="#%i: model_%i" % (datadict[k]["rank"], datadict[k]["idx"]),
        )

    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel("Predicted lDDT")
    plt.xlabel("Positions")
    plt.show()


# ------------------------------------------------------


def prepare_MR_model(
    output, pae_power=1, pae_cutoff=5, graph_resolution=1, output_filename=None
):
    """

    Based on Tristan's Croll code from https://github.com/tristanic/pae_to_domains
    modified by gchojnowski@embl-hamburg.de (write me rather than Tristan if you face any issues!)

    Takes a predicted aligned error (PAE) matrix representing the predicted error in distances between each
    pair of residues in a model, and uses a graph-based community clustering algorithm to partition the model
    into approximately rigid groups.
    last update: 6.12.2021

    Arguments:
        * pae_power (optional, default=1): each edge in the graph will be weighted proportional to (1/pae**pae_power)
        * pae_cutoff (optional, default=5): graph edges will only be created for residue pairs with pae<pae_cutoff
        * graph_resolution (optional, default=1): regulates how aggressively the clustering algorithm is. Smaller values
                           lead to larger clusters. Value should be larger than zero, and values larger than 5 are unlikely to be useful.
        * output_filename: output filename for writing segmentded model (the one you see on display) with REVERSED B-factors (100-plddt)

        Returns: a series of lists, where each list contains the indices of residues belonging to one cluster.



    """

    def __ranges(i):

        for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
            b = list(b)
            yield b[0][1], b[-1][1]

    from pae_to_domains import pae_to_domains
    import igraph
    import itertools

    datadir = os.path.expanduser(output)
    datadict = None
    for subdir in [
        os.path.join(datadir, name)
        for name in os.listdir(datadir)
        if os.path.isdir(os.path.join(datadir, name))
    ]:
        try:
            print("\n*** Processing %s\n" % subdir)
            # pdb_fnames = parse_af2multi(subdir)
            datadict = parse_model_pickles(subdir)
            break
        except:
            pass
    if not datadict:
        print("*** Results not available, cannot prepare MR moldel\n")
        return

    topmodel_fn = sorted(datadict, key=lambda x: datadict[x]["rank"], reverse=True)[-1]

    domains = []
    for x in pae_to_domains.domains_from_pae_matrix_igraph(
        datadict[topmodel_fn]["pae"]
    ):
        domains.append(list(__ranges(x)))

    _ph, _symm = read_ph(os.path.join(datadict[topmodel_fn]["datadir"], "ranked_0.pdb"))

    ph_cut = iotbx.pdb.hierarchy.root()
    ph_cut.append_model(iotbx.pdb.hierarchy.model(id="0"))

    sel_cache = _ph.atom_selection_cache()
    isel = sel_cache.iselection
    for domain in domains:
        group = []
        for _r in domain:
            if (_r[1] - _r[0]) < 1:
                continue
            group.append("resi %i:%i" % _r)
        selstr = " or ".join(group)
        _ph_sel = _ph.select(isel(selstr))

        # there is an issue with detaching broken chains
        for ch in _ph_sel.chains():
            _ch = ch.detached_copy()
            _ch.id = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[
                len(ph_cut.models()[0].chains())
            ]
            print("Selecting a rigid group %s as chain %s" % (selstr, _ch.id))
            ph_cut.models()[0].append_chain(_ch)

    view = py3Dmol.view(
        js="https://3dmol.org/build/3Dmol.js", width=900, height=600, viewergrid=(1, 2)
    )

    viewer = (0, 0)
    view.addModel(ph_cut.as_pdb_string(), "pdb", viewer=viewer)
    view.zoomTo(viewer=viewer)
    set_3dmol_styles(
        view, viewer, chain_ids=[_.id for _ in ph_cut.chains()], color="chain"
    )

    viewer = (0, 1)
    view.addModel(ph_cut.as_pdb_string(), "pdb", viewer=viewer)
    view.zoomTo(viewer=viewer)
    set_3dmol_styles(
        view, viewer, chain_ids=[_.id for _ in ph_cut.chains()], color="lDDT"
    )

    view.show()

    if output_filename:
        for atm in ph_cut.atoms():
            atm.b = 100.0 - atm.b
        fn_path = os.path.join(datadir, output_filename)
        with open(fn_path, "w") as ofile:
            ofile.write(ph_cut.as_pdb_string())

        print("Wrote segmented top-scored model to: ", fn_path)


# ------------------------------------------------------


def parse_model_pickles(datadir):

    datadict = {}

    for fn in glob.glob("%s/result*.pkl" % datadir):
        # m=re.match(r".*result\_model\_(?P<idx>\d+)\_multimer\.pkl", fn)
        m = re.match(r".*result\_model\_(?P<idx>\d+)(\_\w+)?\.pkl", fn)

        with open(fn, "rb") as ifile:
            data = pickle.load(ifile)
            # ptm - ranking_confidence
            datadict[fn] = {
                "datadir": datadir,
                "fn": fn,
                "idx": int(m.group("idx")),
                "ptm": float(data["ptm"])
                if "ptm" in data
                else float(data["ranking_confidence"]),
                "pae": data["predicted_aligned_error"]
                if "predicted_aligned_error" in data
                else None,
                "plddt": data["plddt"],
            }
    assert datadict
    for rank, k in enumerate(
        sorted(datadict, key=lambda x: datadict[x]["ptm"], reverse=True)
    ):
        datadict[k]["rank"] = rank + 1
        print("rank_%i %s pLDDT=%.2f" % (rank, os.path.basename(k), datadict[k]["ptm"]))

    return datadict


# ------------------------------------------------------


def parse_af2multi(datadir, color="lDDT", models=3):

    datadict = parse_model_pickles(datadir)

    if list(datadict.values())[0]["pae"] is not None:
        plot_predicted_alignment_error(datadict, datadir)
    else:
        print(" *** WARNING: No PAE data found")
    plot_plddts(datadict)

    plot_msa_coverage(datadir)

    return sorted(
        glob.glob("%s/ranked_*.pdb" % datadir),
        reverse=False,
        key=lambda x: int(x[:-4].split("_")[-1]),
    )


# ------------------------------------------------------


def parse_results(output, color=None, models=5, multimer=False):

    if color is None:
        color = ["lDDT", "rainbow", "chain"][0]

    datadir = os.path.expanduser(output)

    pdb_fnames = sorted(glob.glob("%s/ranked*.pdb" % datadir))

    ph_array = []
    for idx, fn in enumerate(pdb_fnames[:models]):
        _ph, _symm = read_ph(fn)
        if len(ph_array) > 0:
            _s = superpose.least_squares_fit(
                ph_array[0].atoms().extract_xyz(),
                _ph.atoms().extract_xyz(),
                method=["kearsley", "kabsch"][0],
            )
            rtmx = matrix.rt((_s.r, _s.t))
            _ph.atoms().set_xyz(new_xyz=rtmx * _ph.atoms().extract_xyz())

        ph_array.append(_ph)

    chain_ids = [_.id for _ in _ph.chains()]

    if len(chain_ids) > 1 and color == "chain":
        view = py3Dmol.view(
            js="https://3dmol.org/build/3Dmol.js",
            width=900,
            height=600,
            viewergrid=(1, 2),
        )

        viewer = (0, 0)
        view.addModel(ph_array[0].as_pdb_string(), "pdb", viewer=viewer)
        view.zoomTo(viewer=viewer)
        set_3dmol_styles(
            view, viewer, chain_ids=[_.id for _ in _ph.chains()], color="chain"
        )

        viewer = (0, 1)
        view.addModel(ph_array[0].as_pdb_string(), "pdb", viewer=viewer)
        view.zoomTo(viewer=viewer)
        set_3dmol_styles(
            view, viewer, chain_ids=[_.id for _ in _ph.chains()], color="lDDT"
        )

    else:
        frames = min(models, len(ph_array))
        view = py3Dmol.view(
            js="https://3dmol.org/build/3Dmol.js",
            width=400 * frames,
            height=400,
            viewergrid=(1, frames),
        )

        for idx, _ph in enumerate(ph_array):
            viewer = (0, idx)
            view.addModel(_ph.as_pdb_string(), "pdb", viewer=viewer)
            view.zoomTo(viewer=viewer)

            set_3dmol_styles(view, viewer, chain_ids=chain_ids, color=color)

    view.show()


def parse_results_colour_chains(output, color=None, models=5, multimer=False):

    if color is None:
        color = ["chain"][0]

    datadir = os.path.expanduser(output)

    pdb_fnames = sorted(glob.glob("%s/ranked*.pdb" % datadir))
    # pdb_fnames = []

    # if png_fnames:
    #     pdb_fnames = parse_colabfold(datadir)
    # else:
    #     for subdir in [ os.path.join(datadir, name) for name in os.listdir(datadir)
    #                    if os.path.isdir(os.path.join(datadir, name)) ]:
    #         try:
    #             print("\n*** Processing %s\n" % subdir)
    #             pdb_fnames = parse_af2multi(subdir)
    #             break
    #         except:
    #             pass

    # if not pdb_fnames:
    #     print("\n*** No results found in %s\n" % datadir )
    #     return 1

    ph_array = []
    for idx, fn in enumerate(pdb_fnames[:models]):
        _ph, _symm = read_ph(fn)
        if len(ph_array) > 0:
            _s = superpose.least_squares_fit(
                ph_array[0].atoms().extract_xyz(),
                _ph.atoms().extract_xyz(),
                method=["kearsley", "kabsch"][0],
            )
            rtmx = matrix.rt((_s.r, _s.t))
            _ph.atoms().set_xyz(new_xyz=rtmx * _ph.atoms().extract_xyz())

        ph_array.append(_ph)

    chain_ids = [_.id for _ in _ph.chains()]

    if len(chain_ids) > 1 and color == None:
        view = py3Dmol.view(
            js="https://3dmol.org/build/3Dmol.js",
            width=900,
            height=600,
            viewergrid=(1, 2),
        )

        viewer = (0, 0)
        view.addModel(ph_array[0].as_pdb_string(), "pdb", viewer=viewer)
        view.zoomTo(viewer=viewer)
        set_3dmol_styles(
            view, viewer, chain_ids=[_.id for _ in _ph.chains()], color="chain"
        )

        viewer = (0, 1)
        view.addModel(ph_array[0].as_pdb_string(), "pdb", viewer=viewer)
        view.zoomTo(viewer=viewer)
        set_3dmol_styles(
            view, viewer, chain_ids=[_.id for _ in _ph.chains()], color="chain"
        )

    else:
        frames = min(models, len(ph_array))
        view = py3Dmol.view(
            js="https://3dmol.org/build/3Dmol.js",
            width=400 * frames,
            height=400,
            viewergrid=(1, frames),
        )

        for idx, _ph in enumerate(ph_array):
            viewer = (0, idx)
            view.addModel(_ph.as_pdb_string(), "pdb", viewer=viewer)
            view.zoomTo(viewer=viewer)

            set_3dmol_styles(view, viewer, chain_ids=chain_ids, color=color)

    view.show()


# ------------------------------------------------------


def set_3dmol_styles(
    view,
    viewer,
    chain_ids=1,
    color=["lDDT", "rainbow", "chain"][0],
    show_sidechains=False,
    show_mainchains=False,
):

    """
    borrowed from colabfolds notebook at
    https://github.com/sokrypton/ColabFold/blob/main/colabfold/pdb.py
    """

    if color == "lDDT":
        view.setStyle(
            {
                "cartoon": {
                    "colorscheme": {
                        "prop": "b",
                        "gradient": "roygb",
                        "min": 0,
                        "max": 100,
                    }
                }
            },
            viewer=viewer,
        )
    elif color == "rainbow":
        view.setStyle({"cartoon": {"color": "spectrum"}}, viewer=viewer)

    elif color == "chain":
        for cid, color in zip(
            chain_ids,
            [
                "lime",
                "cyan",
                "magenta",
                "yellow",
                "salmon",
                "white",
                "blue",
                "orange",
                "black",
                "green",
                "gray",
            ]
            * 2,
        ):
            view.setStyle({"chain": cid}, {"cartoon": {"color": color}}, viewer=viewer)
    if show_sidechains:
        BB = ["C", "O", "N"]
        view.addStyle(
            {
                "and": [
                    {"resn": ["GLY", "PRO"], "invert": True},
                    {"atom": BB, "invert": True},
                ]
            },
            {"stick": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
        )
        view.addStyle(
            {"and": [{"resn": "GLY"}, {"atom": "CA"}]},
            {"sphere": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
        )
        view.addStyle(
            {"and": [{"resn": "PRO"}, {"atom": ["C", "O"], "invert": True}]},
            {"stick": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
        )
    if show_mainchains:
        BB = ["C", "O", "N", "CA"]
        view.addStyle(
            {"atom": BB},
            {"stick": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
            viewer=viewer,
        )
