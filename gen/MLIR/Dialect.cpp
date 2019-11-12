//===-- MLIRStatments.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/MLIR/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::D;

//===----------------------------------------------------------------------===//
// DDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
DDialect::DDialect(mlir::MLIRContext *context) : mlir::Dialect("D",
        context) {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
  >();
}


//===----------------------------------------------------------------------===//
// D Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AddOp

void AddOp::build(mlir::Builder *b, mlir::OperationState &state,
        mlir::Value *lhs, mlir::Value *rhs) {
    state.addTypes(UnrankedTensorType::get(b->getIntegerType(16)));
    state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Ops.cpp.inc"