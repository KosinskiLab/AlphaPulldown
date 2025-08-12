from __future__ import annotations

import os
import shutil
import threading
import typing
import wsgiref.simple_server
from pathlib import Path
from typing import Any

import bottle
import pytest

if typing.TYPE_CHECKING:
    import wsgiref.types  # added in 3.11


@pytest.fixture(scope="function")
def testing_workdir(tmpdir, request):
    """Create a workdir in a safe temporary folder; cd into dir above before test, cd out after

    :param tmpdir: py.test fixture, will be injected
    :param request: py.test fixture-related, will be injected (see pytest docs)
    """

    saved_path = os.getcwd()

    tmpdir.chdir()
    # temporary folder for profiling output, if any
    tmpdir.mkdir("prof")

    def return_to_saved_path():
        if os.path.isdir(os.path.join(saved_path, "prof")):
            profdir = tmpdir.join("prof")
            files = profdir.listdir("*.prof") if profdir.isdir() else []

            for f in files:
                shutil.copy(str(f), os.path.join(saved_path, "prof", f.basename))
        os.chdir(saved_path)

    request.addfinalizer(return_to_saved_path)

    return str(tmpdir)


datadir = Path(__file__).parent / "data"


@bottle.route("/<filename>", "GET")
def serve_file(filename):
    mimetype = "auto"
    # from https://repo.anaconda.com/ behavior:
    if filename.endswith(".tar.bz2"):
        mimetype = "application/x-tar"
    elif filename.endswith(".conda"):
        mimetype = "binary/octet-stream"
    return bottle.static_file(filename, root=datadir.as_posix(), mimetype=mimetype)


class ServerThread(threading.Thread):
    server: Any
    app: Any


def get_server_thread():
    """
    Return test server thread with additional .server, .app properties.

    Call .start() to serve in the background.
    """
    app: wsgiref.types.WSGIApplication = bottle.app()
    server = wsgiref.simple_server.make_server("127.0.0.1", 0, app)
    t = ServerThread(daemon=True, target=server.serve_forever)
    t.app = app
    t.server = server  # server.application == app
    return t


@pytest.fixture(scope="session")
def localserver():
    thread = get_server_thread()
    thread.start()
    yield f"http://{thread.server.server_address[0]}:{thread.server.server_address[1]}"


if __name__ == "__main__":
    bottle.run(port=8080)
