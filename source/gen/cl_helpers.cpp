//===-- cl_helpers.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/cl_helpers.h"
#include "mars.h"
#include "rmem.h"
#include "root.h"
#include <algorithm>
#include <cctype> // isupper, tolower
#include <stdarg.h>
#include <utility>

namespace opts {

MultiSetter::MultiSetter(bool invert, bool *p, ...) {
  this->invert = invert;
  if (p) {
    locations.push_back(p);
    va_list va;
    va_start(va, p);
    while ((p = va_arg(va, bool *))) {
      locations.push_back(p);
    }
    va_end(va);
  }
}

void MultiSetter::operator=(bool val) {
  for (auto &l : locations) {
    *l = (val != invert);
  }
}

void StringsAdapter::push_back(const char *cstr) {
  if (!cstr || !*cstr) {
    error(Loc(), "Expected argument to '-%s'", name);
  }

  if (!*arrp) {
    *arrp = new Strings;
  }
  (*arrp)->push(mem.xstrdup(cstr));
}

} // namespace opts
