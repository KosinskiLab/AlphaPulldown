

set -ex



test ! -f ${PREFIX}/bin/iconv
test_man_files=$(jq '.files[] | select( . | startswith("share/man"))' $CONDA_PREFIX/conda-meta/libiconv-1.17-${PKG_BUILD_STRING}.json)
if [[ ${test_man_files} ]]; then echo "found GPL licensed files being packaged ${test_man_files}"; exit 1; fi
test_doc_files=$(jq '.files[] | select( . | startswith("share/doc"))' $CONDA_PREFIX/conda-meta/libiconv-1.17-${PKG_BUILD_STRING}.json)
if [[ ${test_doc_files} ]]; then echo "found GPL licensed files being packaged ${test_doc_files}"; exit 1; fi
test -f $CONDA_PREFIX/lib/libiconv${SHLIB_EXT}
test -f $CONDA_PREFIX/lib/libcharset${SHLIB_EXT}
exit 0
