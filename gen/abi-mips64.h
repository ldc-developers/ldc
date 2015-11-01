//===-- gen/abi-mips64.h - MIPS64 ABI description --------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The ABI implementation used for 64 bit MIPS targets.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_ABI_MIPS64_H
#define LDC_GEN_ABI_MIPS64_H

struct TargetABI;

TargetABI *getMIPS64TargetABI(bool Is64Bit);

#endif
