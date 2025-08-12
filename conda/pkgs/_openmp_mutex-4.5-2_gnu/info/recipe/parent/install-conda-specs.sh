set -ex
export CHOST="${gcc_machine}-${gcc_vendor}-linux-gnu"
specdir=$PREFIX/lib/gcc/$CHOST/${gcc_version}
if [[ "$cross_target_platform" == "$target_platform" ]]; then
    install -Dm644 -T ${SRC_DIR}/build/gcc/specs $specdir/conda.specs

    # Add specs when we're not cross compiling so that the toolchain works more like a system
    # toolchain (i.e. conda installed libs can be #include <>'d and linked without adding any
    # cmdline args or FLAGS and likewise the assumptions we have about rpath are built in)
    #
    # THIS IS INTENDED as a safety net for casual users who just want the native toolchain to work.
    # It is not to be relied on by conda-forge package recipes and best practice is still to set the
    # appropriate FLAGS vars (either via compiler activation scripts or explicitly in the recipe)
    #
    # We use double quotes here because we want $PREFIX and $CHOST to be expanded at build time
    #   and recorded in the specs file.  It will undergo a prefix replacement when our compiler
    #   package is installed.
    sed -i -e "/\*link_command:/,+1 s+%.*+& %{\!static:-rpath ${PREFIX}/lib -rpath-link ${PREFIX}/lib} -L ${PREFIX}/lib+" $specdir/conda.specs
    # put -disable-new-dtags at the front of the cmdline so that user provided -enable-new-dtags (in %l) can  override it
    sed -i -e "/\*link_command:/,+1 s+%(linker)+& -disable-new-dtags +" $specdir/conda.specs
    # use -idirafter to put the conda "system" includes where /usr/local/include would typically go
    # in a system-packaged non-cross compiler
    sed -i -e "/\*cpp_options:/,+1 s+%.*+& -idirafter ${PREFIX}/include+" $specdir/conda.specs
    # cc1_options also get used for cc1plus... at least in 11.2.0
    sed -i -e "/\*cc1_options:/,+1 s+%.*+& -idirafter ${PREFIX}/include+" $specdir/conda.specs

else
    # does it even make sense to do anything here?  Could do something with %:getenv(BUILD_PREFIX  /include) 
    # but in the case that we aren't inside conda-build, it will cause gcc to fatal
    # because it won't be set.  Just explicitly making this fail for now so that the meta.yaml
    # is consitent with when it creates the conda-gcc-specs package
    false
fi
