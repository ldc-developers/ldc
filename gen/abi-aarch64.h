//===-- gen/abi-ppc-64.h - PPC64 ABI description ----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The ABI implementation used for AArch64 targets.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_ABI_AARCH64_H
#define LDC_GEN_ABI_AARCH64_H

struct TargetABI;

TargetABI *getAArch64TargetABI();

#endif
