# When host=target, gcc installs un-prefixed tools together with prefixed-tools
# In the case of cpp, only un-prefixed cpp. Let's copy the un-prefixed tool
# to prefix-tool and delete the un-prefixed one to get back ct-ng behaviour
for tool in gcc g++ gfortran cpp gcc-ar gcc-nm gcc-ranlib c++; do
  if [ -f ${PREFIX}/bin/${tool} ]; then
    if [ ! -f ${PREFIX}/bin/${gcc_machine}-${gcc_vendor}-linux-gnu-${tool} ]; then
      cp ${PREFIX}/bin/${tool} ${PREFIX}/bin/${gcc_machine}-${gcc_vendor}-linux-gnu-${tool}
    fi
    rm ${PREFIX}/bin/${tool}
  fi
done

# Make a symlink from new gcc vendor to the old one
for exe in `ls ${PREFIX}/bin/*-${gcc_vendor}-linux-gnu-*`; do
  nm=`basename ${exe}`
  new_nm=${nm/"-${gcc_vendor}-"/"-${old_gcc_vendor}-"}
  if [ ! -f ${PREFIX}/bin/${new_nm} ]; then
    ln -s ${exe} ${PREFIX}/bin/${new_nm}
  fi
done
