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
std::optional<llvm::Reloc::Model> getRelocModel();
std::optional<llvm::CodeModel::Model> getCodeModel();
#if LDC_LLVM_VER >= 1300
std::optional<llvm::FramePointerKind> framePointerUsage();
#else
std::optional<llvm::FramePointer::FP> framePointerUsage();
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
