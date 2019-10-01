//==- driver/mlircodegenerator.h - D module codegen entry point ---*- C++-*-==//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains ldc::CodeGenerator, which is the main entry point for emitting code
// for one or more D modules to MLIR output.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "gen/mlirstate.h"
#include "mlir/MLIRContext.h"

namespace ldc {

class MLIRCodeGenerator {
public:
  MLIRCodeGenerator(mlir::MLIRContext &context, bool singleObj);
  ~MLIRCodeGenerator();
  void emit(Module *m);

private:
  void prepareMLIRModule(Module *m);
  void finishMLIRModule(Module *m);
  void writeAndFreeMLIRModule(const char *filename);

  mlir::MLIRontext &context_;
  int moduleCount_;
  bool const singleObj_;
  MLIRState *mlir_;
};
}
