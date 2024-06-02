//===-- driver/cl_options-llvm.h - LLVM command line options ----*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#if LDC_LLVM_VER < 1700
#include "llvm/ADT/Optional.h"
#else
#include <optional>
namespace llvm {
template <typename T> using Optional = std::optional<T>;
}
#endif
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
llvm::Optional<llvm::FramePointerKind> framePointerUsage();

bool disableRedZone();
bool printTargetFeaturesHelp();

llvm::TargetOptions
InitTargetOptionsFromCodeGenFlags(const llvm::Triple &triple);

std::string getCPUStr();
std::string getFeaturesStr();
void setFunctionAttributes(llvm::StringRef cpu, llvm::StringRef features,
                           llvm::Function &function);
}
