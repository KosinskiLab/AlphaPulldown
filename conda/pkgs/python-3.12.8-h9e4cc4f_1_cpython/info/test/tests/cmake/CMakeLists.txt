# https://martinopilia.com/posts/2018/09/15/building-python-extension.html
# FindPythonInterp and FindPythonLibs are deprecated since cmake 3.12
# These can be replaced by find_package(Python ${PY_VER} REQUIRED)
# But these are still used by other packages, so we keep them.

cmake_minimum_required(VERSION 3.10)
enable_language(C)
project(mymath)

option(PY_VER, "Python version to use")

find_package(PythonInterp ${PY_VER} REQUIRED)
#            PATHS $ENV{CONDA_PREFIX}

# This goes after, since it uses PythonInterp as hint
find_package(PythonLibs ${PY_VER} REQUIRED)
