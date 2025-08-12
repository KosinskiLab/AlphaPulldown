#! /bin/sh
# Get an updated config.sub and config.guess
cp $BUILD_PREFIX/share/gnuconfig/config.* .

# Create configure script in separate environment to workaround the fact
# that libnsl cannot depend on itself
conda create -y -p $(pwd)/autogen gettext autoconf automake pkg-config
conda run -p $(pwd)/autogen ./autogen.sh

./configure --prefix=${PREFIX} --disable-static
make
if [[ "${CONDA_BUILD_CROSS_COMPILATION:-}" != "1" ]]; then
make check
fi
make install
