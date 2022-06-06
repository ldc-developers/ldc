//===-- gen/ldctraits.h - LDC-specific __traits handling --------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

struct Dstring
{
    size_t length;
    const char *ptr;
};

Dstring traitsGetTargetCPU();
bool traitsTargetHasFeature(Dstring feature);

template <int N> bool traitsTargetHasFeature(const char (&feature)[N]) {
  return traitsTargetHasFeature({N - 1, feature});
}
