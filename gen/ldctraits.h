//===-- gen/ldctraits.h - LDC-specific __traits handling --------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_LDCTRAITS_H
#define LDC_GEN_LDCTRAITS_H

struct Dstring
{
    size_t length = 0;
    const char *ptr = nullptr;

    Dstring(size_t length, const char *ptr) : length(length), ptr(ptr) {}
};

Dstring traitsGetTargetCPU();
bool traitsTargetHasFeature(Dstring feature);

template <int N> bool traitsTargetHasFeature(const char (&feature)[N]) {
  return traitsTargetHasFeature(Dstring(N - 1, feature));
}

#endif
