make install

# TODO :: Only provide a static iconv executable for GNU/Linux.
# TODO :: glibc has iconv built-in. I am only providing it here
# TODO :: for legacy packages (and through gritted teeth).
if [[ ${HOST} =~ .*linux.* ]]; then
  chmod 755 ${PREFIX}/lib/libiconv.so.2.6.1
  chmod 755 ${PREFIX}/lib/libcharset.so.1.0.0
  if [ -f ${PREFIX}/lib/preloadable_libiconv.so ]; then
    chmod 755 ${PREFIX}/lib/preloadable_libiconv.so
  fi
fi

# remove libtool files
find $PREFIX -name '*.la' -delete

if [[ "${PKG_NAME}" == "libiconv" ]]; then
  # remove iconv executable   
  rm $PREFIX/bin/iconv
  rm -rf $PREFIX/share/man
  rm -rf $PREFIX/share/doc
else
  # relying on conda-build to deduplicate files
  echo "Keeping all files, conda-build will deduplicate files"
fi