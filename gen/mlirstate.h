//===-- gen/mlirstate.h - Global codegen MLIR state -------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the global state used and modified when generating the
// code (i.e. MLIR) for a given D module.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/MLIRContext.h"
#include "dmd/globals.h"
#include "gen/dibuilder.h"

namespace mlir {
  class MLIRContext; /*Get the context of the program - similar to LLVMContext*/
  class OwningModuleRef; /*returns a newly created MLIR module or nullptr on
                                                                      failure.*/
} //namespace mlir

class TypeFunction; /*unsed*/
class TypeStruct; /*unsed*/
class ClassDeclaration; /*unsed*/
class FuncDeclaration; /*unsed*/
class Module; /*unsed*/
class TypeStruct; /*unsed*/
struct BaseClass; /*unsed*/
class AnonDeclaration; /*unsed*/
class StructLiteralExp; /*unsed*/

struct IrFunction; /*unsed*/
struct IrModule; /*unsed*/

struct MLIRState ;
extern MLIRState *gMLIR;

namespace Ddialect {
class ModuleAST; //TODO: Create in definitions.cpp a ModuleAST closer to MLIR

/// Emit IR for the given D moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);
} // namespace Ddialect

// represents the MLIR module (object file)
struct MLIRState{
public:
  MLIRState(const char *name, mlir::MLIRContext &context);
  ~MLIRState();

  MLIRState(MLIRState const &) = delete;
  MLIRState &operator=(MLIRState const &) = delete;

  mlir::ModuleOp module;
  mlir::MLIRContext *ontext() const { return module.getContext(); }

  Module *dmodule = nullptr;
  
  /*StructType doesn't has any sintax translation to MLIR.
    TODO: Find a semantic synonym*/
  LLStructType *moduleRefType = nullptr; 

  /*ObjCState objc;  -> Do not worry with ObjC right now!*/

  
  // debug info helper
  ldc::DIBuilder DBuilder;

  // Sets the initializer for a global LL variable.
  // If the types don't match, this entails creating a new helper global
  // matching the initializer type and replacing all existing uses of globalVar
  // by a bitcast pointer to the helper global's payload.
  // Returns either the specified globalVar if the types match, or the bitcast
  // pointer replacing globalVar (and resets globalVar to the new helper
  // global).
  /*Think about the use of this function and if it's important to tranlate
  D into MLIR how to replace this to an mlir::Constant whi the same purpose! */
  llvm::Constant *setGlobalVarInitializer(llvm::GlobalVariable *&globalVar,
                                          llvm::Constant *initializer);

  // Target for dcompute. If not nullptr, it owns this.
  DComputeTarget *dcomputetarget = nullptr;
}
