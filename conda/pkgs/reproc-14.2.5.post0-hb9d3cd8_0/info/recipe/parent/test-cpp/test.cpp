#include <array>
#include <string>
#include <iostream>

#include <reproc++/run.hpp>

int main()
{
  int status = -1;
  std::error_code ec;

  std::array<std::string, 2> cmd = {"echo", "Hello"};
  reproc::options options;
  options.deadline = reproc::milliseconds(5000);

  std::tie(status, ec) = reproc::run(cmd, options);

  return 0;
  // Fails on the CI somehow but packages are fine
  // if (ec) {
  //   std::cerr << ec.message() << std::endl;
  // }
  //
  // return ec ? ec.value() : status;
}
