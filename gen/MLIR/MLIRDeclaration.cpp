//===-- MLIRDeclaration.cpp -----------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//


#if LDC_MLIR_ENABLED

#include "MLIRDeclaration.h"

MLIRDeclaration::MLIRDeclaration(IRState *irs, Module *m,
    mlir::MLIRContext &context, mlir::OpBuilder builder_,
    llvm::ScopedHashTable<StringRef,
    mlir::Value*> &symbolTable) : irState(irs), module(m), context(context),
    builder(builder_), symbolTable(symbolTable) {} //Constructor

MLIRDeclaration::~MLIRDeclaration() = default;

mlir::Value *MLIRDeclaration::mlirGen(VarDeclaration *vd){
  IF_LOG Logger::println("MLIRCodeGen - VarDeclaration: '%s'", vd->toChars
        ());
  LOG_SCOPE
  // if aliassym is set, this VarDecl is redone as an alias to another symbol
  // this seems to be done to rewrite Tuple!(...) v;
  // as a TupleDecl that contains a bunch of individual VarDecls
  if (vd->aliassym) {
    IF_LOG Logger::println("MLIRCodeGen -  VarDeclaration: aliassym -> "
                           "APAGAR");
    //return DtoDeclarationExpMLIR(vd->aliassym, mlir_);
  }
  if (vd->isDataseg()) {
    IF_LOG Logger::println("MLIRCodeGen -  VarDeclaration: dataseg -> "
                           "APAGAR");
    //Declaration_MLIRcodegen(vd, mlir_);
  }else {
    if (vd->nestedrefs.dim) {
      IF_LOG Logger::println(
            "has nestedref set (referenced by nested function/delegate)");

      // A variable may not be really nested even if nextedrefs is not empty
      // in case it is referenced by a function inside __traits(compile) or
      // typeof.
      // assert(vd->ir->irLocal && "irLocal is expected to be already set by
      // DtoCreateNestedContext");
    }

    if (vd->_init) {
      if (ExpInitializer *ex = vd->_init->isExpInitializer()) {
        // TODO: Refactor this so that it doesn't look like toElem has no effect.
        Logger::println("MLIRCodeGen - ExpInitializer: '%s'", ex->toChars());
        LOG_SCOPE
        return mlirGen(ex->exp);
      }
    }
    else{
      IF_LOG Logger::println("Unable to recoganize VarDeclaration: '%s'",
                             vd->toChars());
    }
  }
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
// Expressions to be evaluated

mlir::Value *MLIRDeclaration::mlirGen(DeclarationExp *decl_exp){
  IF_LOG Logger::println("MLIRCodeGen - DeclExp: '%s'", decl_exp->toChars());
  LOG_SCOPE
  Dsymbol *dsym = decl_exp->declaration;


  if (VarDeclaration *vd = dsym->isVarDeclaration())
    return mlirGen(vd);

  IF_LOG Logger::println("Unable to recoganize DeclarationExp: '%s'",
                         dsym->toChars());
  return nullptr;
}

mlir::Value *MLIRDeclaration::mlirGen(AssignExp *assignExp){
  IF_LOG Logger::print(
        "AssignExp::toElem: %s | (%s)(%s = %s)\n", assignExp->toChars(),
        assignExp->type->toChars(), assignExp->e1->type->toChars(),
        assignExp->e2->type ? assignExp->e2->type->toChars() : nullptr);
  if (assignExp->memset & referenceInit) {
    IF_LOG Logger::println(" --> Expression op: '%d'", assignExp->e1->op);
    assert(assignExp->op == TOKconstruct || assignExp->op == TOKblit);
    assert(assignExp->e1->op == TOKvar);

    Declaration *d = static_cast<VarExp *>(assignExp->e1)->var;
    if (d->storage_class & (STCref | STCout)) {
      mlir::Value *value = mlirGen(assignExp->e2);

      if (!value)
        return nullptr;

      mlir::OperationState result(loc(assignExp->loc), "assign");
      result.addTypes(builder.getIntegerType(16)); // TODO: type
      result.addOperands(value);
      value = builder.createOperation(result)->getResult(0);

      if (failed(declare(assignExp->e1->toChars(), value)))
        return nullptr;
      return value;
    }
  }
  IF_LOG Logger::println("Unable to translate AssignExp: '%s'",
      assignExp->toChars());
  return nullptr;
}

/// Emit a call expression. It emits specific operations for the `transpose`
/// builtin. Other identifiers are assumed to be user-defined functions.
mlir::Value *MLIRDeclaration::mlirGen(CallExp *callExp){
  IF_LOG Logger::println("MLIRCodeGen - CallExp: '%s'", callExp->toChars
  ());
  LOG_SCOPE

  // Codegen the operands first.
  llvm::SmallVector<mlir::Value *, 4> operands;
  for(auto exp : *callExp->arguments){
    auto *arg = mlirGen(exp);
    if(!arg)
      return nullptr;
    operands.push_back(arg);
  }

  // Otherwise this is a call to a user-defined function. Calls to
  // user-defined functions are mapped to a custom call that takes the callee
  // name as an attribute.
  mlir::OperationState result(loc(callExp->loc), "ldc.call");
  result.addTypes(builder.getIntegerType(32));
  result.operands = std::move(operands);
  result.addAttribute("callee", builder.getSymbolRefAttr(callExp->f->mangleString));
  return builder.createOperation(result)->getResult(0);
}

mlir::Value *MLIRDeclaration::mlirGen(ConstructExp *constructExp){
  IF_LOG Logger::println("MLIRCodeGen - ConstructExp: '%s'", constructExp->toChars());
  LOG_SCOPE
  //mlir::Value *lhs = mlirGen(constructExp->e1);
  mlir::Value *rhs = mlirGen(constructExp->e2);
  mlir::Value *value = nullptr;

  if (failed(declare(constructExp->e1->toChars(), rhs)))
    return nullptr;
  return value;
}

mlir::Value *MLIRDeclaration::mlirGen(IntegerExp *integerExp){
  dinteger_t dint = integerExp->value;
  Logger::println("Integer: '%lu'", dint);
  mlir::OperationState result(loc(integerExp->loc), "ldc.IntegerExp");
  result.addTypes(builder.getIntegerType(16)); // TODO: type
  result.addAttribute("value", builder.getI16IntegerAttr(dint));
  return builder.createOperation(result)->getResult(0);
}

mlir::Value *MLIRDeclaration::mlirGen(VarExp *varExp){
  Logger::println("VarExp: '%s'", varExp->var->toChars());
  auto var = symbolTable.lookup(varExp->var->toChars());
  if (!var) {
    IF_LOG Logger::println("Undeclared VarExp: '%s'", varExp->toChars());
    mlir::OperationState result(loc(varExp->loc), "ldc.IntegerExp");
    result.addTypes(builder.getIntegerType(16)); // TODO: type
    result.addAttribute("value", builder.getI16IntegerAttr(0));
    return builder.createOperation(result)->getResult(0);
    //return ; // val a = 0; Default value for a variable
  }
  return var;
}

mlir::Value *MLIRDeclaration::mlirGen(ArrayLiteralExp *arrayLiteralExp){
  IF_LOG Logger::println("MLIRCodeGen - ArrayLiteralExp: '%s'",
                         arrayLiteralExp->toChars());
  LOG_SCOPE

 // IF_LOG Logger::println("Basis: '%s'",arrayLiteralExp->basis->toChars());
  IF_LOG Logger::println("Elements: '%s'", arrayLiteralExp->elements->toChars
  ());

  std::vector<double> data;
  for(auto e : *arrayLiteralExp->elements){
    data.push_back(e->toInteger());
  }

  //For now lets assume one-dimensional arrays
  std::vector<int64_t> dims;
  dims.push_back(1);
  dims.push_back(data.size());

  // The type of this attribute is tensor of 64-bit floating-point with the
  // shape of the literal.
  mlir::Type elementType = builder.getF64Type();
  auto dataType = builder.getTensorType(dims, elementType);

  // This is the actual attribute that holds the list of values for this
  // tensor literal.
  auto dataAttribute =
      mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));

  // Build the MLIR op `toy.constant`, only boilerplate below.
  mlir::OperationState result(loc(arrayLiteralExp->loc), "ldc.constant");
  result.addTypes(builder.getTensorType(dims, builder.getF64Type()));
  result.addAttribute("value", dataAttribute);
  return builder.createOperation(result)->getResult(0);
}

mlir::Value *MLIRDeclaration::mlirGen(Expression *expression) {
  IF_LOG Logger::println("MLIRCodeGen - Expression: '%s'",
      expression->toChars());
  LOG_SCOPE

  const char *op_name = nullptr;
  auto location = loc(expression->loc);
  mlir::Value *e1 = nullptr;
  mlir::Value *e2 = nullptr;

  if(VarExp *varExp = expression->isVarExp())
    return mlirGen(varExp);
  else if(IntegerExp *integerExp = expression->isIntegerExp())
    return mlirGen(integerExp);
  else if(ConstructExp *constructExp = expression->isConstructExp())
    return mlirGen(constructExp);
  else if(AssignExp *assignExp = expression->isAssignExp())
    return mlirGen(assignExp);
  else if(CallExp *callExp = expression->isCallExp())
    return mlirGen(callExp);
  else if(ArrayLiteralExp *arrayLiteralExp = expression->isArrayLiteralExp())
    return mlirGen(arrayLiteralExp);
  else
  if(AddExp *add = expression->isAddExp()) {
    e1 = mlirGen(add->e1);
    e2 = mlirGen(add->e2);
    op_name = "ldc.add";
  } else if (MinExp *min = expression->isMinExp()) {
    e1 = mlirGen(min->e1);
    e2 = mlirGen(min->e2);
    op_name = "ldc.neg";
  } else if (MulExp *mul = expression->isMulExp()) {
    e1 = mlirGen(mul->e1);
    e2 = mlirGen(mul->e2);
    op_name = "ldc.mul";
  } else if (DivExp *div = expression->isDivExp()) {
    e1 = mlirGen(div->e1);
    e2 = mlirGen(div->e2);
    op_name = "ldc.div";
  }

  if (e1 != nullptr && e2 != nullptr) {
    mlir::OperationState result(location, op_name);
    result.addTypes(
        builder.getIntegerType(16)); // TODO: MODIFY TO ALLOW MORE TYPES
    result.addOperands({e1, e2});
    return builder.createOperation(result)->getResult(0);
  } else {
    IF_LOG Logger::println("Unable to recoganize the Expression: '%s'",
                    expression->toChars());
    return nullptr;
  }
}
#endif //LDC_MLIR_ENABLED
