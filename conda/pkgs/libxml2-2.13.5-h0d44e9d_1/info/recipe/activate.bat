@echo off

if defined XML_CATALOG_FILES (
    set xml_catalog_files_libxml2=%XML_CATALOG_FILES%
    set XML_CATALOG_FILES=%XML_CATALOG_FILES% 
) else (
    set xml_catalog_files_libxml2=
)

setlocal EnableDelayedExpansion
set conda_catalog_files=file:///!CONDA_PREFIX: =%%20!/etc/xml/catalog
endlocal & set "XML_CATALOG_FILES=%XML_CATALOG_FILES%%conda_catalog_files:\=/%"
