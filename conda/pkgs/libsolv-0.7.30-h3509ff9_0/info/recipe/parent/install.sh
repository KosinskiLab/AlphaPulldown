cd build
make install

if [[ "$PKG_NAME" == *static ]]
then
	# relying on conda to dedup package
	echo "Doing nothing"
else
	rm -rf ${PREFIX}/lib/*a
fi