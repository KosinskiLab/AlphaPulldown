



pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
mypy -p platformdirs --ignore-missing-imports
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
