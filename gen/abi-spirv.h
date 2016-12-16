//===-- gen/abi-spirv.h - SPIRV ABI description ----------------*- C++ -*-===//
//
//                         LDC - the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The ABI implementation used by dcompute for the SPIR-V targets, only OpenCL
// at the moment.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_ABI_SPIRV_H
#define LDC_GEN_ABI_SPIRV_H

struct TargetABI;

TargetABI *createSPIRVABI();

#endif
