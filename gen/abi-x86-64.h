//===-- gen/abi-x86-64.h - x86_64 ABI description ---------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The System V AMD64 ABI implementation used on all x86-64 platforms except
// for Windows.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_ABI_X86_64_H
#define LDC_GEN_ABI_X86_64_H

struct TargetABI;
namespace llvm {
class Type;
}

TargetABI *getX86_64TargetABI();

#endif
