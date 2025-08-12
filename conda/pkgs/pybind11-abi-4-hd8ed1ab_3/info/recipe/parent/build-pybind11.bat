:: There are several required flags, like --no-deps, but conda-build nicely sets them for us
:: https://github.com/conda/conda-build/blob/d75ef9c66dbcc832d8d96f33a2aecfe893266e8c/conda_build/build.py#L2517-L2527
%PYTHON% -m pip install . -vv --no-build-isolation
if errorlevel 1 exit 1
