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

void AddI8Op::build(mlir::Builder *b, mlir::OperationState &state,
        mlir::Value *lhs, mlir::Value *rhs) {
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
      state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void AddI16Op::build(mlir::Builder *b, mlir::OperationState &state,
                    mlir::Value *lhs, mlir::Value *rhs) {
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void AddI32Op::build(mlir::Builder *b, mlir::OperationState &state,
                    mlir::Value *lhs, mlir::Value *rhs) {
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void AddI64Op::build(mlir::Builder *b, mlir::OperationState &state,
                    mlir::Value *lhs, mlir::Value *rhs) {
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

/*void AddI128Op::build(mlir::Builder *b, mlir::OperationState &state,
                    mlir::Value *lhs, mlir::Value *rhs) {
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}*/

void AddF16Op::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs) {
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void AddF32Op::build(mlir::Builder *b, mlir::OperationState &state,
                    mlir::Value *lhs, mlir::Value *rhs) {
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void AddF64Op::build(mlir::Builder *b, mlir::OperationState &state,
                    mlir::Value *lhs, mlir::Value *rhs) {
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Ops.cpp.inc"