//===-- exe_path.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "exe_path.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using std::string;
namespace path = llvm::sys::path;

namespace {
string exePath;
}

void exe_path::initialize(const char *arg0) {
  assert(exePath.empty());

  // Some platforms can't implement LLVM's getMainExecutable
  // without being given the address of a function in the main executable.
  // Thus getMainExecutable needs the address of a function;
  // any function address in the main executable will do.
  exePath = llvm::sys::fs::getMainExecutable(
      arg0, reinterpret_cast<void *>(&exe_path::initialize));
}

const string &exe_path::getExePath() {
  assert(!exePath.empty());
  return exePath;
}

string exe_path::getBinDir() {
  assert(!exePath.empty());
  return string(path::parent_path(exePath));
}

string exe_path::getBaseDir() {
  string binDir = getBinDir();
  assert(!binDir.empty());
  return string(path::parent_path(binDir));
}

string exe_path::getLibDir() {
  llvm::SmallString<128> r(getBaseDir());
  path::append(r, "lib" LDC_LIBDIR_SUFFIX);
  return {r.data(), r.size()};
}

string exe_path::prependBinDir(const llvm::Twine &suffix) {
  llvm::SmallString<128> r(getBinDir());
  path::append(r, suffix);
  return {r.data(), r.size()};
}

string exe_path::prependLibDir(const llvm::Twine &suffix) {
  llvm::SmallString<128> r(getLibDir());
  path::append(r, suffix);
  return {r.data(), r.size()};
}
