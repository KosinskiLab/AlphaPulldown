



pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
distro --help
IF %ERRORLEVEL% NEQ 0 exit /B 1
pytest -vvv --capture=tee-sys tests
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
