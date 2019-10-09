//
// Created by Roberto Rosmaninho on 07/10/19.
//

#ifndef LDC_MLIRGEN_H
#define LDC_MLIRGEN_H

#include "mlir/IR/MLIRContext.h"
#include "dmd/module.h"

#include <memory>

namespace mlir{
class MLIRContext;
class OwningModuleRef;
}

namespace ldc_mlir{
//class Module;
//class IRState;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &mlirContext, Module *m);
} //Namespace ldc_mlir


#endif // LDC_MLIRGEN_H
