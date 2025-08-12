



conda create -n test --dry-run scipy --solver=libmamba
IF %ERRORLEVEL% NEQ 0 exit /B 1
CONDA_SOLVER=libmamba conda create -n test --dry-run scipy
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
