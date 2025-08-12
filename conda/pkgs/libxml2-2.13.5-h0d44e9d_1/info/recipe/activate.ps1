if ($Env:XML_CATALOG_FILES) {
    $Env:xml_catalog_files_libxml2 = "$Env:XML_CATALOG_FILES"
    $Env:XML_CATALOG_FILES += " "
} else {
    $Env:xml_catalog_files_libxml2 = ""
    $Env:XML_CATALOG_FILES = ""
}

$conda_catalog_files += "file:///" + $Env:CONDA_PREFIX.replace(" ", "%20").replace("\", "/") + "/etc/xml/catalog"
$Env:XML_CATALOG_FILES += "$conda_catalog_files"
