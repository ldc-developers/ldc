//===-- gen/abi-x86-64.h - x86_64 ABI description ---------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The ABI implementation used for 64 bit x86 (i.e. x86_64/AMD64/x64) targets.
//
//===----------------------------------------------------------------------===//

#ifndef __LDC_GEN_ABI_X86_64_H__
#define __LDC_GEN_ABI_X86_64_H__

#include "gen/abi.h"


TargetABI* getX86_64TargetABI();


#endif
