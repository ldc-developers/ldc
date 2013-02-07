//===-- gen/abi-x86.h - x86 ABI description ---------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The ABI implementation used for 32 bit x86 targets.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_ABI_X86_H
#define LDC_GEN_ABI_X86_H

struct TargetABI;
TargetABI* getX86TargetABI();

#endif
