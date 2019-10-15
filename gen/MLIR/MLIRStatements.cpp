//===-- MLIRStatments.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "MLIRStatements.h"

namespace llvm{
using llvm::StringRef;
}

MLIRStatements::MLIRStatements(IRState *irs, Module *m,
    mlir::MLIRContext &context, mlir::OpBuilder builder_,
    llvm::ScopedHashTable<StringRef, mlir::Value*> &symbolTable) : irState(irs),
    module(m), context(context), builder(builder_), symbolTable(symbolTable),
    declaration(new MLIRDeclaration(irs, m, context, builder_, symbolTable)) {}
    //Constructor

MLIRStatements::~MLIRStatements() = default; //Default Destructor

mlir::Value* MLIRStatements::mlirGen(ExpStatement *expStmt) {
  IF_LOG Logger::println("MLIRCodeGen: ExpStatement to MLIR: '%s'",
                         expStmt->toChars());
  LOG_SCOPE

  mlir::Value *value = nullptr;

  if (DeclarationExp *decl_exp = expStmt->exp->isDeclarationExp()) {
    value = declaration->mlirGen(decl_exp);
  } else if (Expression *e = expStmt->exp) {
    IF_LOG Logger::println("Passou Aqui");

    if (DeclarationExp *edecl = e->isDeclarationExp()) {
      IF_LOG Logger::println("Declaration");
    }
  } else {
    IF_LOG Logger::println("Unable to recoganize: '%s'",
                           expStmt->exp->toChars());
    return nullptr;
  }
  return value;
}

mlir::LogicalResult MLIRStatements::mlirGen(ReturnStatement *returnStatement){
  //TODO: Accept more types of return
  if(!returnStatement->exp->isIntegerExp() && !returnStatement->exp->isVarExp()){
    IF_LOG Logger::println("Unable to genenerate MLIR code for : '%s'",
        returnStatement->toChars());
    return mlir::failure();
  }

  IF_LOG Logger::println("MLIRCodeGen - Return Stmt: '%s'",
                         returnStatement->toChars());
  LOG_SCOPE

  mlir::OperationState result(loc(returnStatement->loc),"ldc.return");
  if(returnStatement->exp->hasCode()) {
    auto *expr = declaration->mlirGen(returnStatement->exp);
    if(!expr)
      return mlir::failure();
    result.addOperands(expr);
  }
  builder.createOperation(result);
  return mlir::success();
}

void MLIRStatements::mlirGen(CompoundStatement *compoundStatement){
  IF_LOG Logger::println("MLIRCodeGen - CompundStatement: '%s'",
                         compoundStatement->toChars());
  LOG_SCOPE

  for(auto stmt : *compoundStatement->statements){
    if(ExpStatement *expStmt = stmt->isExpStatement()) {
      IF_LOG Logger::println("MLIRCodeGen ExpStatement");
      LOG_SCOPE
      mlirGen(expStmt);
    }else if (CompoundStatement *compoundStatement = stmt->isCompoundStatement()) {
      mlirGen(stmt->isCompoundStatement()); // Try again
    }else if(ReturnStatement *returnStatement = stmt->isReturnStatement()){
      mlirGen(returnStatement);
    }else{
      IF_LOG Logger::println("Unable to recoganize: '%s'",
          stmt->toChars());
    }
  }
}

mlir::Value* MLIRStatements::mlirGen(Statement* stm) {
  IF_LOG Logger::println("Statament doesn't match with any implemented "
                         "function: '%s'",stm->toChars());

  return nullptr;
}

mlir::LogicalResult MLIRStatements::genStatements(FuncDeclaration *funcDeclaration){
  if(CompoundStatement *compoundStatment =
                                  funcDeclaration->fbody->isCompoundStatement()){
    mlirGen(compoundStatment);
    return mlir::success();
  }
  return mlir::failure();
}

