"""Model flags must follow the object, not whichever fold came last.

`run_structure_prediction.main()` decided the flag set inside its loop over folds
and read it after the loop, so every object was predicted with the LAST fold's
flags. Queue a monomer behind a multimer and the monomer was predicted with
model_name "multimer". It did not raise -- it silently predicted the wrong thing,
which is the same failure `fold_preparation.py` was extracted to fix.
"""

from __future__ import annotations

from alphapulldown.inference_flags import group_by_model_flags


def _job(name):
    return {"object": name, "output_dir": f"/out/{name}"}


def test_a_monomer_behind_a_multimer_keeps_its_own_flags():
    monomer = {"model_name": "monomer_ptm"}
    multimer = {"model_name": "multimer", "msa_depth": None}

    groups = group_by_model_flags(
        [(_job("multi"), multimer), (_job("mono"), monomer)]
    )

    by_model = {flags["model_name"]: [j["object"] for j in jobs] for flags, jobs in groups}
    assert by_model == {"multimer": ["multi"], "monomer_ptm": ["mono"]}


def test_like_folds_still_share_one_call():
    """Grouping must not split a homogeneous batch into one call per fold."""
    flags = {"model_name": "monomer_ptm"}

    groups = group_by_model_flags([(_job("a"), flags), (_job("b"), flags), (_job("c"), flags)])

    assert len(groups) == 1
    assert [j["object"] for _, jobs in groups for j in jobs] == ["a", "b", "c"]


def test_order_is_preserved_within_and_between_groups():
    a = {"model_name": "monomer_ptm"}
    b = {"model_name": "multimer"}

    groups = group_by_model_flags(
        [(_job("a1"), a), (_job("b1"), b), (_job("a2"), a), (_job("b2"), b)]
    )

    assert [flags["model_name"] for flags, _ in groups] == ["monomer_ptm", "multimer"]
    assert [[j["object"] for j in jobs] for _, jobs in groups] == [["a1", "a2"], ["b1", "b2"]]


def test_unhashable_flag_values_do_not_break_grouping():
    """Flag values include lists and None; grouping must not require hashability."""
    flags = {"model_name": "multimer", "model_names_custom": ["model_2_multimer_v3"]}

    groups = group_by_model_flags([(_job("a"), flags), (_job("b"), dict(flags))])

    assert len(groups) == 1


def test_no_jobs_means_no_calls():
    assert group_by_model_flags([]) == []
