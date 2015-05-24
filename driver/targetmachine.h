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

#include "llvm/Support/CodeGen.h"
#include <string>
#include <vector>

namespace ExplicitBitness {
    enum Type {
        None,
        M32,
        M64
    };
}

namespace FloatABI {
    enum Type {
        Default,
        Soft,
        SoftFP,
        Hard
    };
}

namespace MipsABI {
    enum Type {
        Unknown,
        O32,
        N32,
        N64,
        EABI
    };
}

namespace llvm { class TargetMachine; }

/**
 * Creates an LLVM TargetMachine suitable for the given (usually command-line)
 * parameters and the host platform defaults.
 *
 * Does not depend on any global state.
 */
llvm::TargetMachine* createTargetMachine(
    std::string targetTriple,
    std::string arch,
    std::string cpu,
    std::vector<std::string> attrs,
    ExplicitBitness::Type bitness,
    FloatABI::Type floatABI,
    llvm::Reloc::Model relocModel,
    llvm::CodeModel::Model codeModel,
    llvm::CodeGenOpt::Level codeGenOptLevel,
    bool noFramePointerElim,
    bool noLinkerStripDead
    );

/**
 * Returns the Mips ABI which is used for code generation.
 *
 * Function may only be called after the target machine is created.
 * Returns MipsABI::Unknown in case the ABI is not known (e.g. not compiling
 * for Mips).
 */
MipsABI::Type getMipsABI();

#endif // LDC_DRIVER_TARGET_H
