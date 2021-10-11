//===-- driver/cl_options-llvm.h - LLVM command line options ----*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/Optional.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetOptions.h"

namespace llvm {
class Function;
class Triple;
}

namespace opts {

std::string getArchStr();
llvm::Optional<llvm::Reloc::Model> getRelocModel();
llvm::Optional<llvm::CodeModel::Model> getCodeModel();
#if LDC_LLVM_VER >= 1300
llvm::Optional<llvm::FramePointerKind> framePointerUsage();
#elif LDC_LLVM_VER >= 800
llvm::Optional<llvm::FramePointer::FP> framePointerUsage();
#else
llvm::cl::boolOrDefault disableFPElim();
#endif

bool disableRedZone();
bool printTargetFeaturesHelp();

llvm::TargetOptions
InitTargetOptionsFromCodeGenFlags(const llvm::Triple &triple);

std::string getCPUStr();
std::string getFeaturesStr();
#if LDC_LLVM_VER >= 1000
void setFunctionAttributes(llvm::StringRef cpu, llvm::StringRef features,
                           llvm::Function &function);
#endif
}
