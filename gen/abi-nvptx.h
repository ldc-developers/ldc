//===-- gen/abi-nvptx.h - NVPTX ABI description ----------------*- C++ -*-===//
//
//                         LDC - the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The ABI implementation used by dcompute for targetting NVPTX for cuda
//
//===----------------------------------------------------------------------===//

#pragma once

struct TargetABI;

TargetABI *createNVPTXABI();
