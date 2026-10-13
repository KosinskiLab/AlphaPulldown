"""The fold record: what a resumed run needs to know about an earlier one."""


def test_fold_record_is_incomplete_until_finished(tmp_path):
    from alphapulldown.prediction import resume

    settings = {"backend": "alphafold3", "seeds": (5, 6), "num_recycles": 10}
    assert resume.read_fold_record(tmp_path) == {}

    resume.start_fold_record(tmp_path, settings)
    record = resume.read_fold_record(tmp_path)
    assert record["complete"] is False
    assert record["settings"]["seeds"] == [5, 6]
    assert not resume.finished_fold(record, settings, "fold", tmp_path)

    resume.finish_fold_record(tmp_path)
    assert resume.finished_fold(resume.read_fold_record(tmp_path), settings, "fold", tmp_path)


def test_fold_record_names_the_settings_that_changed(tmp_path):
    from alphapulldown.prediction import resume

    resume.record_fold_complete(tmp_path, {"seeds": [5], "storage_mode": "vanilla"})
    record = resume.read_fold_record(tmp_path)

    assert resume.changed_settings(record, {"seeds": [5], "storage_mode": "slim"}) == [
        "storage_mode"
    ]
    assert resume.changed_settings(record, {"seeds": [5]}) == ["storage_mode"]
    assert not resume.finished_fold(
        record, {"seeds": [5], "storage_mode": "slim"}, "fold", tmp_path
    )


def test_finishing_a_fold_without_a_record_writes_none(tmp_path):
    from alphapulldown.prediction import resume

    resume.finish_fold_record(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_an_unreadable_fold_record_counts_as_none(tmp_path):
    from alphapulldown.prediction import resume

    (tmp_path / resume.FOLD_RECORD_NAME).write_text('{"complete": tr', encoding="utf-8")

    assert resume.read_fold_record(tmp_path) == {}
