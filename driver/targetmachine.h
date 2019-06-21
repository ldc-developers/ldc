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

#ifndef LDC_DRIVER_TARGET_H
#define LDC_DRIVER_TARGET_H

#if LDC_LLVM_VER >= 309
#include "llvm/ADT/Optional.h"
#endif
#include "llvm/Support/CodeGen.h"
#include <string>
#include <vector>

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
}

/**
 * Creates an LLVM TargetMachine suitable for the given (usually command-line)
 * parameters and the host platform defaults.
 *
 * Does not depend on any global state.
 */
llvm::TargetMachine *createTargetMachine(
    std::string targetTriple, std::string arch, std::string cpu,
    std::string featuresStr, ExplicitBitness::Type bitness,
    FloatABI::Type floatABI,
#if LDC_LLVM_VER >= 309
    llvm::Optional<llvm::Reloc::Model> relocModel,
#else
    llvm::Reloc::Model relocModel,
#endif
#if LDC_LLVM_VER >= 600
    llvm::Optional<llvm::CodeModel::Model> codeModel,
#else
    llvm::CodeModel::Model codeModel,
#endif
    llvm::CodeGenOpt::Level codeGenOptLevel, bool noLinkerStripDead);

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

#endif // LDC_DRIVER_TARGET_H
