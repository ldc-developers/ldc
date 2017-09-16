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

#ifndef LDC_GEN_ABI_NVPTX_H
#define LDC_GEN_ABI_NVPTX_H

struct TargetABI;

TargetABI *createNVPTXABI();

#endif
