//===-- gen/abi-win64.h - Windows x86_64 ABI description --------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The ABI implementation used for 64 bit x86 (i.e. x86_64/AMD64/x64) targets
// on Windows.
//
//===----------------------------------------------------------------------===//

#ifndef __LDC_GEN_ABI_WIN64_H__
#define __LDC_GEN_ABI_WIN64_H__

#include "gen/abi.h"


TargetABI* getWin64TargetABI();


#endif
