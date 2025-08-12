IF not "x%PKG_NAME:static=%" == "x%PKG_NAME%" (
    set BUILD_DIR="build-static"
) ELSE (
    set BUILD_DIR="build-dyn"
)
IF not "x%PKG_NAME:reproc-cpp=%" == "x%PKG_NAME%" (
    set COMPONENT="reproc++"
) ELSE (
    set COMPONENT="reproc"
)

cmake --install "%BUILD_DIR%" --component %COMPONENT%-development
if errorlevel 1 exit 1
cmake --install "%BUILD_DIR%" --component %COMPONENT%-runtime
if errorlevel 1 exit 1
