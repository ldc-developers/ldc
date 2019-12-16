//===-- MLIRStatments.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
#if LDC_MLIR_ENABLED

#include "gen/MLIR/Dialect.h"
#include "gen/logger.h"

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
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
      state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}


void AddFOp::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs) {
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void SubOp::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs) {
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void SubFOp::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs) {
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void MulOp::build(mlir::Builder *b, mlir::OperationState &state,
                  mlir::Value *lhs, mlir::Value *rhs){
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void MulFOp::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs){
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void DivFOp::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs){
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void DivSOp::build(mlir::Builder *b, mlir::OperationState &state,
                  mlir::Value *lhs, mlir::Value *rhs){
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void DivUOp::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs){
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void ModSOp::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs){
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void ModUOp::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs){
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void ModFOp::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs){
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void AndOp::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs){
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void OrOp::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs){
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void XorOp::build(mlir::Builder *b, mlir::OperationState &state,
                   mlir::Value *lhs, mlir::Value *rhs){
  if(lhs->getType() == rhs->getType())
    state.addTypes(lhs->getType());
  else
    state.addTypes(mlir::NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void CallOp::build(mlir::Builder *b, mlir::OperationState &state,
                   llvm::StringRef callee, llvm::ArrayRef<mlir::Type> types,
                   llvm::ArrayRef<mlir::Value*> arguments) {

  state.addTypes(types);
  state.addOperands(arguments);
  state.addAttribute("callee", b->getSymbolRefAttr(callee));
}

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void IntegerOp::build(mlir::Builder *builder, mlir::OperationState &state,
                      mlir::Type type, int value, int size = 0) {
  if(type.isInteger(size)){
    auto dataType =builder->getIntegerType(size);
    auto dataAttribute = builder->getIntegerAttr(dataType, value);
    IntegerOp::build(builder, state, dataType, dataAttribute);
  }else{
    IF_LOG Logger::println("Unable to get the Attribute for %d", value);
  }
}

void FloatOp::build(mlir::Builder *builder, mlir::OperationState &state,
                    mlir::Type type, float value) {
  if(type.isF16()){
    auto dataType = builder->getF16Type();
    auto dataAttribute = builder->getF16FloatAttr(value);
    FloatOp::build(builder, state, dataType, dataAttribute);
  }else if(type.isF32()){
    auto dataType = builder->getF32Type();
    auto dataAttribute = builder->getF32FloatAttr(value);
    FloatOp::build(builder, state, dataType, dataAttribute);
  }else{
    IF_LOG Logger::println("Unable to get the Attribute for %f", value);
  }
}

void DoubleOp::build(mlir::Builder *builder, mlir::OperationState &state,
                     mlir::Type type, double value) {
  if(type.isF64()){
    auto dataType = builder->getF64Type();
    auto dataAttribute = builder ->getF64FloatAttr(value);
    DoubleOp::build(builder, state, dataType, dataAttribute);
  }else{
    IF_LOG Logger::println("Unable to get the Attribute for %f", value);
  }
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

#endif //LDC_MLIR_ENABLED