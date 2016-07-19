//===-- dcompute/abi.h - LDC ------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef __ldc__abi_cuda__
#define __ldc__abi_cuda__

#include "gen/abi.h"

TargetABI *createCudaABI();
TargetABI *createOCLABI();
#endif /* defined(__ldc__abi_cuda__) */
