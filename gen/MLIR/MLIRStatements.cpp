//===-- MLIRStatments.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "mlir/Dialect/StandardOps/Ops.h"
#include "MLIRStatements.h"

namespace llvm{
using llvm::StringRef;
}

MLIRStatements::MLIRStatements(IRState *irs, Module *m,
    mlir::MLIRContext &context, mlir::OpBuilder builder_,
    llvm::ScopedHashTable<StringRef, mlir::Value*> &symbolTable, unsigned
    &total, unsigned &miss) : irState(irs),
    module(m), context(context), builder(builder_), symbolTable(symbolTable),
    declaration(new MLIRDeclaration(irs, m, context, builder_, symbolTable,
        decl_total, decl_miss)), _total(total), _miss(miss) {}
    //Constructor

MLIRStatements::~MLIRStatements() = default; //Default Destructor

mlir::Value* MLIRStatements::mlirGen(ExpStatement *expStmt) {
  IF_LOG Logger::println("MLIRCodeGen: ExpStatement to MLIR: '%s'",
                         expStmt->toChars());
  LOG_SCOPE

  mlir::Value *value = nullptr;

  if (DeclarationExp *decl_exp = expStmt->exp->isDeclarationExp()) {
    value = declaration->mlirGen(decl_exp, builder.getInsertionBlock());
  } else if (Expression *e = expStmt->exp) {
    value = declaration->mlirGen(e, builder.getInsertionBlock());
    if (DeclarationExp *edecl = e->isDeclarationExp()) {
      IF_LOG Logger::println("Declaration");
    }
  } else {
    _miss++;
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

void MLIRStatements::mliGen(IfStatement *ifStatement){
  IF_LOG Logger::println("MLIRCodeGen: IfStatement to MLIR: '%s'",
                         ifStatement->toChars());
  LOG_SCOPE

  unsigned if_total = 0, if_miss = 0;
  //Builing the object to get the Value for an expression
  MLIRDeclaration *mlirDeclaration = new MLIRDeclaration(irState,module,
                                  context, builder, symbolTable, if_total,
                                  if_miss);



  //Getting Value for Condition
  mlir::Value *cond = mlirDeclaration->mlirGen(ifStatement->condition);

  mlir::Location location = loc(ifStatement->loc);
  mlir::OperationState result(location,"ldc.if");

  //When we create an block mlir automatically change the insert point, but
  // we have to keep it to insert the if operation inside it's own block an
  // then we can write on each successor block.
  mlir::Block *insert = builder.getInsertionBlock();

  //Creating two blocks if, else and end
  mlir::Block *if_then = builder.createBlock(cond->getParentRegion(),
      cond->getParentRegion()->end());
  mlir::Block *if_else = nullptr;
  if(ifStatement->elsebody)
      builder.createBlock(cond->getParentRegion(),
                                             cond->getParentRegion()->end());
  mlir::Block *end_if = builder.createBlock(cond->getParentRegion(),
                                            cond->getParentRegion()->end());

  //Getting back to the old insertion point
  builder.setInsertionPointAfter(&insert->back());

  mlir::CondBranchOp branch; //TODO: Make args to block generic -> phi nodes
  if(ifStatement->elsebody)
    branch.build(&builder, result, cond, if_then, {}/*args to block*/,
                                             if_else, {} /*args to block*/);
  else
    branch.build(&builder, result, cond, if_then, {}/*args to block*/,
                                         end_if,  {}/*args to block*/);

  builder.createOperation(result);

  //After create the branch operation we can fill each block with their
  // operations
  builder.setInsertionPointToStart(if_then);
  auto _result = mlirGen(ifStatement->ifbody->isScopeStatement());


  if(ifStatement->elsebody){
    builder.setInsertionPointToStart(if_else);
    _result = mlirGen(ifStatement->elsebody->isScopeStatement());
  }

  //Writing a branch instruction on each block (if, else) to (end)
  mlir::OperationState jump(location, "ldc.br");
  builder.setInsertionPointToEnd(if_then);
  mlir::BranchOp br_if;
  br_if.build(&builder, jump, end_if, {});
  builder.createOperation(jump);

  if(ifStatement->elsebody) {
    mlir::OperationState jump1(location, "ldc.br");
    builder.setInsertionPointToEnd(if_else);
    mlir::BranchOp br_else;
    br_if.build(&builder, jump1, end_if, {});
    builder.createOperation(jump1);
  }

  //Setting the insertion point to the block before if_then and else
  builder.setInsertionPointToStart(end_if);

  _total += if_total;
  _miss += if_miss;

}


mlir::LogicalResult MLIRStatements::mlirGen(ReturnStatement *returnStatement){
  //TODO: Accept more types of return
  if(!returnStatement->exp->isIntegerExp() && !returnStatement->exp->isVarExp()){
    _miss++;
    IF_LOG Logger::println("Unable to genenerate MLIR code for : '%s'",
        returnStatement->toChars());
    return mlir::failure();
  }

  IF_LOG Logger::println("MLIRCodeGen - Return Stmt: '%s'",
                         returnStatement->toChars());
  LOG_SCOPE

  mlir::OperationState result(loc(returnStatement->loc),"ldc.return");
  if(returnStatement->exp->hasCode()) {
    auto *expr = declaration->mlirGen(returnStatement->exp, builder.getInsertionBlock());
    if(!expr)
      return mlir::failure();
    result.addOperands(expr);
  }
  builder.createOperation(result);
  return mlir::success();
}

std::vector<mlir::Value*> MLIRStatements::mlirGen(CompoundStatement *compoundStatement){
  IF_LOG Logger::println("MLIRCodeGen - CompundStatement: '%s'",
                         compoundStatement->toChars());
  LOG_SCOPE

  std::vector<mlir::Value*> arrayValue;

  for(auto stmt : *compoundStatement->statements){
    _total++;
    if (CompoundStatement *compoundStatement = stmt->isCompoundStatement()) {
      arrayValue = mlirGen(compoundStatement); // Try again
    }else if(ExpStatement *expStmt = stmt->isExpStatement()) {
      arrayValue.push_back(mlirGen(expStmt));
    }else if(ReturnStatement *returnStatement = stmt->isReturnStatement()){
      mlirGen(returnStatement);
    }else if(IfStatement *ifStatement = stmt->isIfStatement()) {
      mliGen(ifStatement);
    }else{
      _miss++;
      IF_LOG Logger::println("Statament doesn't match with any implemented "
                             "CompoundStatement implemented: '%s'",
          stmt->toChars());
    }
  }
  return arrayValue;
}


std::vector<mlir::Value*> MLIRStatements::mlirGen(ScopeStatement *scopeStatement){
  IF_LOG Logger::println("MLIRCodeGen - ScopeStatement: '%s'",
                         scopeStatement->toChars());
  LOG_SCOPE
  std::vector<mlir::Value*> arrayValue;

  if(auto *compoundStatement = scopeStatement->statement->isCompoundStatement()) {
    arrayValue = mlirGen(compoundStatement);
  }

  return arrayValue;
}

mlir::Value* MLIRStatements::mlirGen(Statement* stm) {
  IF_LOG Logger::println("Statament doesn't match with any implemented "
                         "function: '%s'",stm->toChars());
  _total++;
  _miss++;
  return nullptr;
}

mlir::LogicalResult MLIRStatements::genStatements(FuncDeclaration *funcDeclaration){
  _total++;
  if(CompoundStatement *compoundStatment =
                                  funcDeclaration->fbody->isCompoundStatement()){
    mlirGen(compoundStatment);
    _total += decl_total;
    _miss += decl_miss;
    return mlir::success();
  }
  _miss++;
  _total += decl_total;
  _miss += decl_miss;
  return mlir::failure();
}

#endif
