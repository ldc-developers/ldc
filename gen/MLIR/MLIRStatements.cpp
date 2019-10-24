//===-- MLIRStatments.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

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
    value = declaration->mlirGen(e);
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

/*mlir::Value* MLIRStatements::mliGen(ForStatement *forStatement){ //TODO
  IF_LOG Logger::println("MLIRCodeGen: ForStatement to MLIR: '%s'",
                         forStatement->toChars());
  LOG_SCOPE

  mlir::CondBranchOp condBranchOp;
  //condBranchOp->;
  //mlir::Block;

}*/

mlir::Value* MLIRStatements::mliGen(IfStatement *ifStatement){
  IF_LOG Logger::println("MLIRCodeGen: IfStatement to MLIR: '%s'",
                         ifStatement->toChars());
  LOG_SCOPE

  mlir::Location location = loc(ifStatement->loc);

  MLIRDeclaration *mlirDeclaration = new MLIRDeclaration(irState,module,
                                  context, builder, symbolTable);

 /* mlir::Value *cond = mlirDeclaration->mlirGen(ifStatement->condition);
  mlir::Block *ifblock = builder.createBlock(builder.getBlock());
  mlir::Block *endblock = builder.createBlock(ifblock);
  mlir::Block *elseblock =
      ifStatement->elsebody ? builder.createBlock(ifblock) : endblock;

  //TODO: Create an branchOp for D on Dialect
  mlir::OperationState result(location,"ldc.if");

  result.addSuccessor(ifblock,
      mlirGen(ifStatement->ifbody->isScopeStatement()));
  result.addTypes(builder.getNoneType());
//  result.addOperands(cond);
  return builder.createOperation(result)->getResult(0);*/
 return nullptr;
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
    }else if(IfStatement *ifStatement = stmt->isIfStatement()) {
      mliGen(ifStatement);
    }else{
      IF_LOG Logger::println("Unable to recoganize: '%s'",
          stmt->toChars());
    }
  }
}


mlir::ArrayRef<mlir::Value*> MLIRStatements::mlirGen(ScopeStatement
*scopeStatement){
  IF_LOG Logger::println("MLIRCodeGen - ScopeStatement: '%s'",
                         scopeStatement->toChars());
  LOG_SCOPE
  std::vector<mlir::Value*> vec;

  auto stmt = scopeStatement->statement;
  if(auto *compoundStatement = stmt->isCompoundStatement())
    for(auto stmt : *compoundStatement->statements)
      if(auto exp = stmt->isExpStatement())
        vec.push_back(mlirGen(exp));
      if(auto ret = stmt->isReturnStatement())
        mlirGen(ret);
      if(auto If = stmt->isIfStatement())
        vec.push_back(mlirGen(If));
  else if(auto expStatement = stmt->isExpStatement())
    vec.push_back(mlirGen(expStatement));

  mlir::ArrayRef<mlir::Value*> arrayValue(vec);
  return arrayValue;
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

#endif
