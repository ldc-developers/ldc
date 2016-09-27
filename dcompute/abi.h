//===-- dcompute/abi.h - LDC ------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DCOMPUTE_ABI_H
#define LDC_DCOMPUTE_ABI_H

#include "gen/abi.h"

TargetABI *createCudaABI();
TargetABI *createOCLABI();
#endif
