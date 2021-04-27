//===-- gen/abi-aarch-64.h - AArch64 ABI description ------------*- C++ -*-===//
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

#pragma once

struct TargetABI;

TargetABI *getAArch64TargetABI();
