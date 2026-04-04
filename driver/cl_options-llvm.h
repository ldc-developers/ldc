//===-- driver/cl_options-llvm.h - LLVM command line options ----*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetOptions.h"
#include <optional>

namespace llvm {
class Function;
class Triple;
}

namespace opts {

std::string getArchStr();
std::optional<llvm::Reloc::Model> getRelocModel();
std::optional<llvm::CodeModel::Model> getCodeModel();
std::optional<llvm::FramePointerKind> framePointerUsage();

bool disableRedZone();
bool printTargetFeaturesHelp();

llvm::TargetOptions
InitTargetOptionsFromCodeGenFlags(const llvm::Triple &triple);

std::string getCPUStr();
std::string getFeaturesStr();
void setFunctionAttributes(llvm::StringRef cpu, llvm::StringRef features,
                           llvm::Function &function);
}
