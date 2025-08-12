



pytest --doctest-modules boltons tests
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
