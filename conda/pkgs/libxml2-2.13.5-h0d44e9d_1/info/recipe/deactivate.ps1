if ($Env:xml_catalog_files_libxml2) {
   $Env:XML_CATALOG_FILES = "$Env:xml_catalog_files_libxml2"
} else {
   $Env:XML_CATALOG_FILES = ''
}
$Env:xml_catalog_files_libxml2 = ''
