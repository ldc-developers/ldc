//==-- MLIR/MLIRDeclaration.h - Generate Declarations MLIR code --*- C++ -*-==//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Generates MLIR code for one or more D Declarations and return nullptr if it
// wasn't able to identify a given declaration.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#define LDC_MLIRDECLARATION_H

#include "dmd/statement.h"
#include "dmd/declaration.h"
#include "dmd/expression.h"
#include "dmd/init.h"
#include "dmd/template.h"
#include "dmd/visitor.h"

#include "gen/logger.h"
#include "gen/modules.h"
#include "gen/irstate.h"
#include "gen/MLIR/Dialect.h"
#include "gen/MLIR/MLIRGen.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/ScopedHashTable.h"

using llvm::StringRef;
using llvm::ScopedHashTableScope;

class MLIRDeclaration {
private:
  IRState *irState;
  Module *module;
  mlir::Value *stmt = nullptr;

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

  /// Temporary flags to mesure the total amount of hits and misses on our
  /// translation through MLIR
  unsigned &_total, &_miss;

public:
  MLIRDeclaration(IRState *irs, Module *m, mlir::MLIRContext &context,
      mlir::OpBuilder builder_,
      llvm::ScopedHashTable<StringRef, mlir::Value*> &symbolTable, unsigned
      &total, unsigned &miss);
  ~MLIRDeclaration();

  mlir::Value* mlirGen(VarDeclaration* varDeclaration);
  mlir::Value* mlirGen(Declaration* declaration);
  mlir::Value* DtoAssignMLIR(mlir::Location Loc, mlir::Value* lhs,
      mlir::Value* rhs, int op, bool canSkipPostblitm, Type* t1, Type* t2);

  //Expression
  mlir::Value* mlirGen(AddExp *addExp = nullptr, AddAssignExp *addAssignExp = nullptr);
  mlir::Value* mlirGen(DeclarationExp* declarationExp);
  mlir::Value* mlirGen(Expression *expression, mlir::Block *block = nullptr);
  mlir::Value* mlirGen(AssignExp *assignExp); //Not perfet yet
  mlir::Value* mlirGen(ConstructExp *constructExp);
  mlir::Value* mlirGen(IntegerExp *integerExp);
  mlir::Value* mlirGen(RealExp *realExp);
  mlir::Value* mlirGen(VarExp *varExp);
  mlir::Value* mlirGen(CallExp *callExp);
  mlir::Value* mlirGen(ArrayLiteralExp *arrayLiteralExp);
  mlir::Value* mlirGen(AndAssignExp *addAssignExp);
  mlir::Value* mlirGen(Expression *expression, int func);
  void mlirGen(TemplateInstance *templateInstance);
  mlir::Value* mlirGen(PostExp *postExp);
  mlir::Type get_MLIRtype(Expression* expression);

  ///Set MLIR Location using D Loc info
  mlir::Location loc(Loc loc){
    return builder.getFileLineColLoc(builder.getIdentifier(
        StringRef(loc.filename)),loc.linnum, loc.charnum);
  }

/// Declare a variable in the current scope, return success if the variable
/// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value *value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }
};

#endif // LDC_MLIR_ENABLED
