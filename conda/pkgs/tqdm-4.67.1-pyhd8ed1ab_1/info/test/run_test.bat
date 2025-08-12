



pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
tqdm --help
IF %ERRORLEVEL% NEQ 0 exit /B 1
tqdm -v
IF %ERRORLEVEL% NEQ 0 exit /B 1
pytest -k "not tests_perf and not tests_tk"
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
