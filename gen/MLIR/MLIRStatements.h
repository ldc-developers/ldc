//===-- MLIR/MLIRStatements.h - Generate Statements MLIR code ---*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Generates MLIR code for one or more D Statements and return nullptr if it
// wasn't able to identify a given statement.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#define LDC_MLIRSTATMENTS_H

#include "dmd/statement.h"
#include "dmd/statement.h"
#include "dmd/expression.h"

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/modules.h"
#include "gen/MLIR/MLIRGen.h"
#include "gen/MLIR/MLIRDeclaration.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/ScopedHashTable.h"

using llvm::StringRef;
using llvm::ScopedHashTableScope;

class MLIRStatements{
private:
  IRState *irState;
  Module *module;

  /// In MLIR (like in LLVM) a "context" object holds the memory allocation and
  /// ownership of many internal structures of the IR and provides a level of
  /// "uniquing" across multiple modules (types for instance).
  mlir::MLIRContext &context;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value *> &symbolTable;

  ///Class to deal with all declarations.
  MLIRDeclaration *declaration = nullptr;

public:
  MLIRStatements(IRState *irs, Module *m, mlir::MLIRContext &context,
      mlir::OpBuilder builder_, llvm::ScopedHashTable<StringRef, mlir::Value
      *> &symbolTable);
  ~MLIRStatements();
  mlir::Value* mlirGen(Statement *statement);
  mlir::Value* mlirGen(ExpStatement *expStatement);
  mlir::LogicalResult mlirGen(ReturnStatement *returnStatement);
  void mlirGen(CompoundStatement *compoundStatement);
 // mlir::Value *mlirGen(Expression *exp);
  mlir::LogicalResult genStatements(FuncDeclaration *funcDeclaration);
  mlir::Value* mliGen(IfStatement *ifStatement);
  mlir::ArrayRef<mlir::Value*> mlirGen(ScopeStatement *scopeStatement);

  mlir::Location loc(Loc loc){
    return builder.getFileLineColLoc(builder.getIdentifier(
        StringRef(loc.filename)),loc.linnum, loc.charnum);
  }

/// Declare a variable in the current scope, return success if the variable
/// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value *value) {
    if(symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }
};

#endif //LDC_MLIR_ENABLED 

