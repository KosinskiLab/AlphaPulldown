

set -ex



echo on
xmllint test.xml
mkdir -p "${PREFIX}/etc/xml"
cp   test_catalog.xml "${PREFIX}/etc/xml/catalog"
xmlcatalog "" "http://www.w3.org/2001/xml.xsd" | grep -x -F -e "file://test-uri-override"
rm  "${PREFIX}/etc/xml/catalog"
xmlcatalog "" "test-id" | grep -x -F -e "No entry for URI test-id"
xmlcatalog "test_catalog.xml" "test-id" | grep -x -F -e "file://test-uri"
export XML_CATALOG_FILES="file://$(pwd)/test_catalog.xml"
xmlcatalog "" "test-id" | grep -x -F -e "file://test-uri"
xmlcatalog "" "http://www.w3.org/2009/01/xml.xsd" | grep -x -F -e "No entry for URI http://www.w3.org/2009/01/xml.xsd"
exit 0
