get_cpu_arch() {
  local CPU_ARCH
  if [[ "$1" == *"-64" ]]; then
    CPU_ARCH="x86_64"
  elif [[ "$1" == *"-ppc64le" ]]; then
    CPU_ARCH="powerpc64le"
  elif [[ "$1" == *"-aarch64" ]]; then
    CPU_ARCH="aarch64"
  elif [[ "$1" == *"-s390x" ]]; then
    CPU_ARCH="s390x"
  else
    echo "Unknown architecture"
    exit 1
  fi
  echo $CPU_ARCH
}

get_triplet() {
  if [[ "$1" == linux-* ]]; then
    echo "$(get_cpu_arch $1)-conda-linux-gnu"
  elif [[ "$1" == osx-64 ]]; then
    echo "x86_64-apple-darwin13.4.0"
  elif [[ "$1" == osx-arm64 ]]; then
    echo "arm64-apple-darwin20.0.0"
  elif [[ "$1" == win-64 ]]; then
    echo "x86_64-w64-mingw32"
  else
    echo "unknown platform"
    exit 1
  fi
}

export BUILD="$(get_triplet $build_platform)"
export HOST="$(get_triplet $target_platform)"
export TARGET="$(get_triplet $cross_target_platform)"
