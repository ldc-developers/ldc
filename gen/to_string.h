//===-- gen/to_string.h - std::to_string replacement ------------*- C++ -*-===//
//
//                         LDC - the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Provide to_string because std::to_string is not available on some systems,
// notably Android. See https://github.com/android-ndk/ndk/issues/82.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sstream>
#include <string>

namespace ldc {

template <class T> const std::string to_string(const T &val) {
  std::ostringstream os;
  os << val;
  return os.str();
}

} // namespace ldc
