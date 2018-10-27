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

#pragma once

struct TargetABI;

TargetABI *getX86_64TargetABI();
