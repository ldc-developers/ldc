//===-- driver/targetmachine.h - LLVM target setup --------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Handles setting up an LLVM TargetMachine from the ugiven command-line
// arguments.
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CodeGen.h"
#include <string>
#include <vector>
namespace llvm {
#if LDC_LLVM_VER < 1800
using CodeGenOptLevel = llvm::CodeGenOpt::Level;
#endif
}

namespace ExplicitBitness {
enum Type { None, M32, M64 };
}

namespace FloatABI {
enum Type { Default, Soft, SoftFP, Hard };
}

namespace MipsABI {
enum Type { Unknown, O32, N32, N64, EABI };
}

namespace llvm {
class Triple;
class Target;
class TargetMachine;
class Module;
}

namespace ComputeBackend {
enum Type { None, SPIRV, NVPTX };
}

ComputeBackend::Type getComputeTargetType(llvm::Module*);

/**
 * Creates an LLVM TargetMachine suitable for the given (usually command-line)
 * parameters and the host platform defaults.
 * Also finalizes floatABI if it's set to FloatABI::Default.
 *
 * Does not depend on any global state.
*/
llvm::TargetMachine *
createTargetMachine(std::string targetTriple, std::string arch, std::string cpu,
                    std::string featuresString, ExplicitBitness::Type bitness,
                    FloatABI::Type &floatABI,
                    llvm::Optional<llvm::Reloc::Model> relocModel,
                    llvm::Optional<llvm::CodeModel::Model> codeModel,
                    llvm::CodeGenOptLevel codeGenOptLevel,
                    bool noLinkerStripDead);

/**
 * Returns the Mips ABI which is used for code generation.
 *
 * Function may only be called after the target machine is created.
 * Returns MipsABI::Unknown in case the ABI is not known (e.g. not compiling
 * for Mips).
 */
MipsABI::Type getMipsABI();

// Looks up a target based on an arch name and a target triple.
const llvm::Target *lookupTarget(const std::string &arch, llvm::Triple &triple,
                                 std::string &errorMsg);

const char *getABI(const llvm::Triple &triple,
                   const llvm::SmallVectorImpl<llvm::StringRef> &features);

bool isFeatureEnabled(const llvm::SmallVectorImpl<llvm::StringRef> &features,
                      llvm::StringRef feature);
