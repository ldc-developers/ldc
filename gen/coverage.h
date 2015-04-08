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

#ifndef LDC_GEN_COVERAGE_H
#define LDC_GEN_COVERAGE_H

struct Loc;

void emitCoverageLinecountInc(Loc &loc);

#endif
