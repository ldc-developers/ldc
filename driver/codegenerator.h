//===-- driver/codegenerator.h - D module codegen entry point ---*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains ldc::CodeGenerator, which is the main entry point for emitting code
// for one or more D modules to LLVM IR and subsequently to whatever output
// format has been chosen globally.
//
// Currently reads parts of the configuration from global.params, as the code
// has been extracted straight out of main(). This should be cleaned up in the
// future.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "gen/irstate.h"

#if LDC_MLIR_ENABLED
namespace mlir {
class MLIRContext;
class OwningModuleRef;
}
#endif

namespace ldc {

class CodeGenerator {
public:
  CodeGenerator(llvm::LLVMContext &context,
#if LDC_MLIR_ENABLED
                mlir::MLIRContext &mlirContext,
#endif
                bool singleObj);

  ~CodeGenerator();
  void emit(Module *m);

#if LDC_MLIR_ENABLED
  void emitMLIR(Module *m);
#endif

private:
  void prepareLLModule(Module *m);
  void finishLLModule(Module *m);
  void writeAndFreeLLModule(const char *filename);
#if LDC_MLIR_ENABLED
  void writeMLIRModule(mlir::OwningModuleRef *module, const char *filename);
#endif

  llvm::LLVMContext &context_;
#if LDC_MLIR_ENABLED
  mlir::MLIRContext &mlirContext_;
#endif
  int moduleCount_;
  bool const singleObj_;
  IRState *ir_;
};
}
