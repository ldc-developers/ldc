//===-- cl_helpers.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/cl_helpers.h"

#include "dmd/errors.h"
#include "dmd/ldcbindings.h"
#include "dmd/root/rmem.h"
#include <algorithm>
#include <cctype> // isupper, tolower
#include <stdarg.h>
#include <utility>

namespace opts {

DString dupPathString(llvm::StringRef src) {
  const auto length = src.size();
  char *r = static_cast<char *>(mem.xmalloc(length + 1));
  memcpy(r, src.data(), length);
#if _WIN32
  std::replace(r, r + length, '/', '\\');
#endif
  r[length] = '\0';
  return {length, r};
}

DString fromPathString(const cl::opt<std::string> &src) {
  if (src.getNumOccurrences() != 0) {
    if (src.empty()) {
      error(Loc(), "Expected argument to '-%s'", src.ArgStr.str().c_str());
    }
    return dupPathString(src);
  }
  return {0, nullptr};
}

MultiSetter::MultiSetter(bool invert, CHECKENABLE *p, ...) {
  this->invert = invert;
  if (p) {
    locations.push_back(p);
    va_list va;
    va_start(va, p);
    while ((p = va_arg(va, CHECKENABLE *))) {
      locations.push_back(p);
    }
    va_end(va);
  }
}

void MultiSetter::operator=(bool val) {
  for (auto &l : locations) {
    *l = (val != invert) ? CHECKENABLEon : CHECKENABLEoff;
  }
}

void StringsAdapter::push_back(const char *cstr) {
  if (!cstr || !*cstr) {
    error(Loc(), "Expected argument to '-%s'", name);
  }

  arrp->push(mem.xstrdup(cstr));
}

} // namespace opts
