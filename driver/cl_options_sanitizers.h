//=== driver/cl_options_sanitizers.h - LDC command line options -*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Deals with -fsanitize=...
//
//===----------------------------------------------------------------------===//

#pragma once

#include "gen/cl_helpers.h"
#include "llvm/Transforms/Instrumentation.h"

class FuncDeclaration;
namespace llvm {
class raw_ostream;
}

namespace opts {
namespace cl = llvm::cl;

typedef unsigned SanitizerBits;
enum SanitizerCheck : SanitizerBits {
  NoneSanitizer = 0,
  AddressSanitizer = 1 << 0,
  FuzzSanitizer = 1 << 1,
  MemorySanitizer = 1 << 2,
  ThreadSanitizer = 1 << 3,
  CoverageSanitizer = 1 << 4
};
extern SanitizerBits enabledSanitizers;

inline bool isAnySanitizerEnabled() { return enabledSanitizers; }
inline bool isSanitizerEnabled(SanitizerCheck san) {
  return enabledSanitizers & san;
}

void initializeSanitizerOptionsFromCmdline();

llvm::SanitizerCoverageOptions getSanitizerCoverageOptions();

void outputSanitizerSettings(llvm::raw_ostream &hash_os);

bool functionIsInSanitizerBlacklist(FuncDeclaration *funcDecl);

} // namespace opts
