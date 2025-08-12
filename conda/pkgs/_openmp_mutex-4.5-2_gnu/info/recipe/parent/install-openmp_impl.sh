set -e -x

export CHOST="${gcc_machine}-${gcc_vendor}-linux-gnu"

mkdir -p ${PREFIX}/lib

pushd ${PREFIX}/lib/
ln -s libgomp.so.${libgomp_ver} libgomp.so.${libgomp_ver:0:1}
popd
