//===-- driver/toobj.h - Object file emission -------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Handles emission of "finished" LLVM modules to on-disk object files.
//
//===----------------------------------------------------------------------===//

#pragma once
#if LDC_MLIR_ENABLED
#include "mlir/IR/MLIRContext.h"
#include "gen/irstate.h"
#include "dmd/module.h"
#endif

namespace llvm {
class Module;
}

void writeModule(llvm::Module *m, const char *filename);
