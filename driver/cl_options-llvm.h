//===-- driver/cl_options-llvm.h - LLVM command line options ----*- C++ -*-===//
//
//                         LDC � the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DRIVER_CL_OPTIONS_LLVM_H
#define LDC_DRIVER_CL_OPTIONS_LLVM_H

#include "llvm/ADT/Optional.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetOptions.h"

namespace opts {

std::string getArchStr();
#if LDC_LLVM_VER >= 309
llvm::Optional<llvm::Reloc::Model> getRelocModel();
#else
llvm::Reloc::Model getRelocModel();
#endif
llvm::CodeModel::Model getCodeModel();
llvm::cl::boolOrDefault disableFPElim();
bool disableRedZone();
bool printTargetFeaturesHelp();

llvm::TargetOptions InitTargetOptionsFromCodeGenFlags();
std::string getCPUStr();
std::string getFeaturesStr();
}

#endif
