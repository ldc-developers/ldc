//===-- driver/tomlirfile.h - MLIR file emission ----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Handles emission of "finished" MLIR modules to on-disk object files.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "gen/irstate.h"
#include "dmd/module.h"
#include "mlir/IR/MLIRContext.h"

void writeMLIRModule(Module *m, mlir::MLIRContext &mlirContext,
    const char *filename, IRState *irs);
#endif // LDC_MLIR_ENABLED

