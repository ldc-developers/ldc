//===-- gen/abi-ppc-64.h - PPC64 ABI description ----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The ABI implementation used for 64 bit little-endian PowerPC targets.
//
//===----------------------------------------------------------------------===//

#pragma once

struct TargetABI;

TargetABI *getPPC64LETargetABI();
