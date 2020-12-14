//===-- gen/coverage.h - Code Coverage Analysis -----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to generate code for code coverage analysis.
// The coverage analysis is enabled by the "-cov" commandline switch.
//
//===----------------------------------------------------------------------===//

#pragma once

struct Loc;

void emitCoverageLinecountInc(const Loc &loc);
