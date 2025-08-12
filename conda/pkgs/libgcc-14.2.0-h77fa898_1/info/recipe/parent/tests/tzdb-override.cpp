// We are using a weak symbol in linux and windows using a patch
// while upstream only has the definition and no implementation.
// Check that overriding works.

// adapted from upstream tests
// https://github.com/gcc-mirror/gcc/blob/releases/gcc-14.1.0/libstdc%2B%2B-v3/testsuite/std/time/tzdb/1.cc
// https://github.com/gcc-mirror/gcc/blob/releases/gcc-14.1.0/libstdc%2B%2B-v3/testsuite/std/time/tzdb/leap_seconds.cc

#include <chrono>
#include <stdio.h>
#include <stdlib.h>

static bool override_used = false;

namespace __gnu_cxx
{
  const char* zoneinfo_dir_override() {
    override_used = true;
    return "./";
  }
}

using namespace std::chrono;

int main()
{
  auto &a = get_tzdb();
  return !override_used;
}
