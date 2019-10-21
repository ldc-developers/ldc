//===-- MLIR/MLIRGen.h - D module CodeGen to MLIR ---------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains ldc::mlirGen, which is the main entry point for emitting code
// for one or more D modules to MLIR. This class executes like a visitor.
//
//===----------------------------------------------------------------------===//
#if LDC_MLIR_ENABLED
#define LDC_MLIRGEN_H

#include "mlir/IR/MLIRContext.h"
#include "dmd/module.h"

#include <memory>

namespace mlir{
class MLIRContext;
class OwningModuleRef;
}

namespace ldc_mlir{

/// Emit IR for the given D module, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &mlirContext, Module *m,
    IRState *irs);
} //Namespace ldc_mlir


#endif // LDC_MLIR_ENABLED
