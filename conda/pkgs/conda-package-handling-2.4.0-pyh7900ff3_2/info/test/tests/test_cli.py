import os
from pathlib import Path

import pytest

import conda_package_handling.cli as cli

from .test_api import data_dir, test_package_name


def test_cli(tmpdir, mocker):
    """
    Code coverage for the cli.
    """
    for command in [
        [
            "x",
            str(Path(data_dir, test_package_name + ".tar.bz2")),
            f"--prefix={tmpdir}",
        ],
        [
            "x",
            str(Path(data_dir, test_package_name + ".conda")),
            "--info",
            f"--prefix={tmpdir}",
        ],
        [
            "c",
            str(Path(tmpdir, test_package_name)),
            ".tar.bz2",
            f"--out-folder={tmpdir}",
        ],
    ]:
        cli.main(args=command)

    # XXX difficult to get to this error handling code through the actual CLI;
    # for example, a .tar.bz2 that can't be extracted raises OSError instead of
    # returning errors. Designed for .tar.bz2 -> .conda conversions that somehow
    # omit files?
    mocker.patch(
        "conda_package_handling.api.transmute",
        return_value=set("that is why you fail".split()),
    )
    with pytest.raises(SystemExit):
        command = [
            "t",
            str(Path(data_dir, test_package_name + ".tar.bz2")),
            ".conda",
            f"--out-folder={tmpdir}",
        ]
        cli.main(args=command)


def test_import_main():
    """
    e.g. python -m conda_package_handling
    """
    with pytest.raises(SystemExit):
        import conda_package_handling.__main__  # noqa


@pytest.mark.parametrize(
    "artifact,n_files",
    [("mock-2.0.0-py37_1000.conda", 43), ("mock-2.0.0-py37_1000.tar.bz2", 43)],
)
def test_list(artifact, n_files, capsys):
    "Integration test to ensure `cph list` works correctly."
    cli.main(["list", os.path.relpath(os.path.join(data_dir, artifact), os.getcwd())])
    stdout, stderr = capsys.readouterr()
    assert n_files == sum(bool(line.strip()) for line in stdout.splitlines())

    # test verbose flag
    cli.main(
        [
            "list",
            "--verbose",
            os.path.join(data_dir, artifact),
        ]
    )
    stdout, stderr = capsys.readouterr()
    assert n_files == sum(bool(line.strip()) for line in stdout.splitlines())

    with pytest.raises(ValueError):
        cli.main(["list", "setup.py"])

    cli.main(["list", os.path.join(data_dir, artifact), "--components=pkg"])
    stdout, stderr = capsys.readouterr()
    listed_files = sum(bool(line.strip()) for line in stdout.splitlines())
    if artifact.endswith(".conda"):
        assert listed_files < n_files  # info folder filtered out
    else:
        assert listed_files == n_files  # no info filtering in tar.bz2


@pytest.mark.parametrize(
    "fn,n_files",
    [
        ("mock-2.0.0-py37_1000.conda", 43),
        ("mock-2.0.0-py37_1000.tar.bz2", -1),
    ],
)
def test_list_remote(capsys, localserver, fn, n_files):
    "Integration test to ensure `cph list <URL>` works correctly."
    url = "/".join([localserver, fn])
    if url.endswith(".tar.bz2"):
        # This is not supported in streaming mode
        with pytest.raises(ValueError):
            cli.main(["list", url])
        return

    cli.main(["list", url])
    stdout, _ = capsys.readouterr()
    assert n_files == sum(bool(line.strip()) for line in stdout.splitlines())

    # Local list should be the same as a 'remote' list
    cli.main(["list", str(Path(__file__).parent / "data" / fn)])
    stdout_local, _ = capsys.readouterr()
    assert list(map(str.strip, stdout_local.splitlines())) == list(
        map(str.strip, stdout.splitlines())
    )

    cli.main(["list", url, "-v"])
    stdout_verbose, _ = capsys.readouterr()
    assert n_files == sum(bool(line.strip()) for line in stdout_verbose.splitlines())
