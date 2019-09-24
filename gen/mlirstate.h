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
  class MLIRContext;
}

struct MLIRState *gMLIR;


struct IRScope {
  /// In MLIR (like in LLVM) a "context" object holds the memory allocation and
  /// ownership of many internal structures of the IR and provides a level of
  /// "uniquing" across multiple modules (types for instance).
  mlir::MLIRContext &context;

  /// This "module" matches the D source file: containing a list of functions.
  mlir::ModuleOp;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuider builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value *> symbolTable;


  
};

/// Helper conversion for the DMD AST location to an MLIR location.
mlir::Location loc(Location loc){
  llvm::StringRef filename(loc.filename);
  return builder.getFileLineColLoc(builder.getIdentifier(filename), loc.linnum,
                                   loc.charnum);
}/*Is this a duplicate from dmd/globals.h:404 ? */

// Declare a variable in the current scope, return success if the variable
/// wasn't declared yet.
mlir::LogicalResult declare(llvm::StringRef var, mlir::Value *value) {
  if (symbolTable.count(var))
    return mlir::failure();
  symbolTable.insert(var, value);
  return mlir::success();
} /*Is this really necessary?*/

struct MLIRState{
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
