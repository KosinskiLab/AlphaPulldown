@echo off

if defined xml_catalog_files_libxml2 (
    set XML_CATALOG_FILES=%xml_catalog_files_libxml2%
) else (
    set XML_CATALOG_FILES=
)
set xml_catalog_files_libxml2=
