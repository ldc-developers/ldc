//===-- gen/abi-targets.h - ABI description for targets ---------*- C++ -*-===//
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

TargetABI *getArmTargetABI();

TargetABI *getMIPS64TargetABI(bool Is64Bit);

TargetABI *createNVPTXABI();

TargetABI *getPPCTargetABI(bool Is64Bit);

TargetABI *getPPC64LETargetABI();

TargetABI *getRISCV64TargetABI();

TargetABI *createSPIRVABI();

TargetABI *getWin64TargetABI();

TargetABI *getX86_64TargetABI();

TargetABI *getX86TargetABI();

TargetABI *getLoongArch64TargetABI();
