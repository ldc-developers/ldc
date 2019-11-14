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
    mlir::Value*> &symbolTable, unsigned &total, unsigned &miss) : irState
    (irs), module(m), context(context), builder(builder_), symbolTable
    (symbolTable), _total(total), _miss(miss){}
    //Constructor

MLIRDeclaration::~MLIRDeclaration() = default;

mlir::Value *MLIRDeclaration::mlirGen(Declaration *declaration){
  IF_LOG Logger::println("MLIRCodeGen - Declaration: '%s'",
      declaration->toChars());
  LOG_SCOPE

  if(auto varDeclaration = declaration->isVarDeclaration())
    return mlirGen(varDeclaration);
  else {
    IF_LOG Logger::println("Unable to recoganize Declaration: '%s'",
                           declaration->toChars());
    _miss++;
    return nullptr;
  }
}

mlir::Value *MLIRDeclaration::mlirGen(VarDeclaration *vd){
  IF_LOG Logger::println("MLIRCodeGen - VarDeclaration: '%s'", vd->toChars
        ());
  LOG_SCOPE
  _total++;
  // if aliassym is set, this VarDecl is redone as an alias to another symbol
  // this seems to be done to rewrite Tuple!(...) v;
  // as a TupleDecl that contains a bunch of individual VarDecls
  if (vd->aliassym) {
    IF_LOG Logger::println("MLIRCodeGen -  VarDeclaration: aliassym");
    //return DtoDeclarationExpMLIR(vd->aliassym, mlir_);
  }
  if (vd->isDataseg()) {
    IF_LOG Logger::println("MLIRCodeGen -  VarDeclaration: dataseg");
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
  _miss++;
  return nullptr;
}

mlir::Value* MLIRDeclaration::DtoAssignMLIR(mlir::Location Loc,
             mlir::Value* lhs, mlir::Value* rhs, int op, bool
             canSkipPostblit, Type* t1, Type* t2){
  IF_LOG Logger::println("DtoAssignMLIR()");
  LOG_SCOPE;

  assert(t1->ty != Tvoid && "Cannot assign values of type void.");

  if (t1->ty == Tbool) {
    IF_LOG Logger::println("DtoAssignMLIR == Tbool"); //TODO: DtoStoreZextI8
  }

  IF_LOG Logger::println("Passou aqui! -> '%u'", op);
  return nullptr;
}


////////////////////////////////////////////////////////////////////////////////
// Expressions to be evaluated

mlir::Value *MLIRDeclaration::mlirGen(DeclarationExp *decl_exp){
  IF_LOG Logger::println("MLIRCodeGen - DeclExp: '%s'", decl_exp->toChars());
  LOG_SCOPE
  Dsymbol *dsym = decl_exp->declaration;
  _total++;

  if (VarDeclaration *vd = dsym->isVarDeclaration())
    return mlirGen(vd);

  IF_LOG Logger::println("Unable to recoganize DeclarationExp: '%s'",
                         dsym->toChars());
  _miss++;
  return nullptr;
}

mlir::Value *MLIRDeclaration::mlirGen(AssignExp *assignExp){
  _total++;
  IF_LOG Logger::print(
        "AssignExp::toElem: %s | (%s)(%s = %s)\n", assignExp->toChars(),
        assignExp->type->toChars(), assignExp->e1->type->toChars(),
        assignExp->e2->type ? assignExp->e2->type->toChars() : nullptr);
  if (assignExp->memset & referenceInit) {
    assert(assignExp->op == TOKconstruct || assignExp->op == TOKblit);
    assert(assignExp->e1->op == TOKvar);

    Declaration *d = static_cast<VarExp *>(assignExp->e1)->var;
    if (d->storage_class & (STCref | STCout)) {
      Logger::println("performing ref variable initialization");
      mlir::Value *rhs = mlirGen(assignExp->e2);
      //mlir::Value *lhs = mlirGen(assignExp->e1);
      mlir::Value *value = nullptr;

      if (!rhs) {
        _miss++;
        return nullptr;
      }
      mlir::OperationState result(loc(assignExp->loc), "assign");
      result.addTypes(rhs->getType()); // TODO: type
      result.addOperands(rhs);
      value = builder.createOperation(result)->getResult(0);

      if (failed(declare(assignExp->e1->toChars(), rhs))) {
        _miss++;
        return nullptr;
      }
      return value;
    }
  }

  // This matches the logic in AssignExp::semantic.
  // TODO: Should be cached in the frontend to avoid issues with the code
  // getting out of sync?
  bool lvalueElem = false;
  if ((assignExp->e2->op == TOKslice &&
       static_cast<UnaExp *>(assignExp->e2)->e1->isLvalue()) ||
      (assignExp->e2->op == TOKcast &&
       static_cast<UnaExp *>(assignExp->e2)->e1->isLvalue()) ||
      (assignExp->e2->op != TOKslice && assignExp->e2->isLvalue())) {
    lvalueElem = true;
  }

  Type *t1 = assignExp->e1->type->toBasetype();
  Type *t2 = assignExp->e2->type->toBasetype();

  if(!((assignExp->e1->type->toBasetype()->ty) == Tstruct) &&
     !(assignExp->e2->op == TOKint64) && !(assignExp->op == TOKconstruct ||
                                           assignExp->op == TOKblit) && !
                                           (assignExp->e1->op == TOKslice))
    DtoAssignMLIR(loc(assignExp->loc),  mlirGen(assignExp->e1),
                mlirGen(assignExp->e2), assignExp->op,!lvalueElem, t1, t2);


  //check if it is a declared variable
  mlir::Value* lhs = nullptr;
  lhs = mlirGen(assignExp->e1->isVarExp());

  if(lhs != nullptr) {
    lhs = mlirGen(assignExp->e2);
    return lhs;
  }else{
    _miss++;
    IF_LOG Logger::println("Failed to assign '%s' to '%s'",
        assignExp->e2->toChars(), assignExp->e1->toChars());
    return nullptr;
  }

  IF_LOG Logger::println("Unable to translate AssignExp: '%s'",
      assignExp->toChars());
  _miss++;
  return nullptr;
}

/// Emit a call expression. It emits specific operations for the `transpose`
/// builtin. Other identifiers are assumed to be user-defined functions.
mlir::Value *MLIRDeclaration::mlirGen(CallExp *callExp){
  IF_LOG Logger::println("MLIRCodeGen - CallExp: '%s'", callExp->toChars());
  LOG_SCOPE
  _total++;

  // Codegen the operands first.
  llvm::SmallVector<mlir::Value *, 4> operands;
  for(auto exp : *callExp->arguments){
    auto *arg = mlirGen(exp);
    if(!arg) {
      _miss++;
      return nullptr;
    }
    operands.push_back(arg);
  }
  //Get the return type
  mlir::Value* ret = nullptr;
  if(ReturnStatement *returnStatement = *callExp->f->returns->data)
    ret = mlirGen(returnStatement->exp);

  mlir::Type ret_type = nullptr;
  if(ret)
    ret_type = ret->getType();
  else
    ret_type = mlir::NoneType::get(&context);
  // Otherwise this is a call to a user-defined function. Calls to
  // user-defined functions are mapped to a custom call that takes the callee
  // name as an attribute.
  mlir::OperationState result(loc(callExp->loc), "ldc.call");
  result.addTypes(ret_type);
  result.operands = std::move(operands);
  result.addAttribute("callee", builder.getSymbolRefAttr(callExp->f->mangleString));
  return builder.createOperation(result)->getResult(0);
}

mlir::Value *MLIRDeclaration::mlirGen(ConstructExp *constructExp){
  IF_LOG Logger::println("MLIRCodeGen - ConstructExp: '%s'", constructExp->toChars());
  LOG_SCOPE
  _total++;
  //mlir::Value *lhs = mlirGen(constructExp->e1);
  mlir::Value *rhs = mlirGen(constructExp->e2);

  if (failed(declare(constructExp->e1->toChars(), rhs))){
    _miss++;
    return nullptr;
  }
  return rhs;
}

mlir::Value *MLIRDeclaration::mlirGen(IntegerExp *integerExp){
  _total++;
  dinteger_t dint = integerExp->value;
  Logger::println("Integer: '%lu'", dint);
  auto ret_type = get_MLIRtype(integerExp);
  mlir::Attribute value_type;
  if(ret_type.isInteger(1))
    value_type = builder.getIntegerAttr(builder.getI1Type(), dint);
  else if(ret_type.isInteger(8))
    value_type = builder.getI8IntegerAttr(dint);
  else if(ret_type.isInteger(16))
    value_type = builder.getI16IntegerAttr(dint);
  else if(ret_type.isInteger(32))
    value_type = builder.getI32IntegerAttr(dint);
  else if(ret_type.isInteger(64))
    value_type = builder.getI64IntegerAttr(dint);
  else
    _miss++; //TODO: Create getI128IntegerAttr on DDialect
  llvm::StringRef name;
  if(value_type.getType() == builder.getI1Type())
    name = "ldc.bool";
  else
    name = "ldc.IntegerExp";
  mlir::OperationState result(loc(integerExp->loc), name);
  result.addTypes(ret_type);
  result.addAttribute("value", value_type);
  return builder.createOperation(result)->getResult(0);
}



mlir::Value* MLIRDeclaration::mlirGen(RealExp *realExp){
  _total++;
  real_t dfloat = realExp->value;
  Logger::println("RealExp: '%Lf'", dfloat);
  mlir::OperationState result(loc(realExp->loc), "ldc.Integer");
  auto ret_type = get_MLIRtype(realExp);
  result.addTypes(ret_type);
  mlir::Attribute value_type;
  if(ret_type.isF16())
    value_type = builder.getF16FloatAttr(dfloat);
  else if(ret_type.isF32())
    value_type = builder.getF32FloatAttr(dfloat);
  else if(ret_type.isF64())
    value_type = builder.getF64FloatAttr(dfloat);
  else
    _miss++; //TODO: Create getF80FloatAttr on DDialect
  result.addAttribute("value", value_type);
  return builder.createOperation(result)->getResult(0);
}

mlir::Value *MLIRDeclaration::mlirGen(VarExp *varExp){
  _total++;
  Logger::println("VarExp: '%s'", varExp->var->toChars());
  auto var = symbolTable.lookup(varExp->var->toChars());
  if (!var) {
    IF_LOG Logger::println("Undeclared VarExp: '%s' | '%u'", varExp->toChars(), varExp->op);
    mlir::OperationState result(loc(varExp->loc), "ldc.IntegerExp");
    result.addTypes(get_MLIRtype(varExp));
    result.addAttribute("value", builder.getI16IntegerAttr(0));
    return builder.createOperation(result)->getResult(0);
    // var = 0; Default value for a variable
  }
  return var;
}

mlir::Value *MLIRDeclaration::mlirGen(ArrayLiteralExp *arrayLiteralExp){
  IF_LOG Logger::println("MLIRCodeGen - ArrayLiteralExp: '%s'",
                         arrayLiteralExp->toChars());
  LOG_SCOPE
  _total++;

 // IF_LOG Logger::println("Basis: '%s'",arrayLiteralExp->basis->toChars());
  IF_LOG Logger::println("Elements: '%s'", arrayLiteralExp->elements->toChars());

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
  auto dataType = mlir::RankedTensorType::get(dims, elementType);

  // This is the actual attribute that holds the list of values for this
  // tensor literal.
  auto dataAttribute =
      mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));

  // Build the MLIR op `toy.constant`, only boilerplate below.
  mlir::OperationState result(loc(arrayLiteralExp->loc), "ldc.constant");
  result.addTypes(mlir::RankedTensorType::get(dims, builder.getF64Type()));
  result.addAttribute("value", dataAttribute);
  return builder.createOperation(result)->getResult(0);
}

mlir::Value *MLIRDeclaration::mlirGen(Expression *expression, int func){
  CmpExp *cmpExp = static_cast<CmpExp*>(expression);
  IF_LOG Logger::println("MLIRCodeGen - CmpExp: '%s'", cmpExp->toChars());
  _total++;

  mlir::Location location = loc(cmpExp->loc);
  mlir::Value *e1 = mlirGen(cmpExp->e1);
  mlir::Value *e2 = mlirGen(cmpExp->e2);

  Type *t = cmpExp->e1->type->toBasetype();//?

  StringRef name;

  switch(cmpExp->op){
    case TOKlt:
      name = t->isunsigned() ? "ult" : "slt";
      break;
    case TOKle:
      name = t->isunsigned() ? "ule" : "sle";
      break;
    case TOKgt:
      name = t->isunsigned() ? "ugt" : "sgt";
      break;
    case TOKge:
      name = t->isunsigned() ? "uge" : "sge";
      break;
    case TOKequal:
      name = "eq";
      break;
    case TOKnotequal:
      name = "neq";
      break;
    default:
      _miss++;
      IF_LOG Logger::println("Invalid comparison operation");
      break;
  }

  mlir::OperationState result(location, "icmp");
  result.addTypes(builder.getIntegerType(1));
  auto stringAttr = mlir::StringAttr::get(name, &context);
  result.addAttribute("Type", stringAttr);
  result.addOperands({e1,e2});
  return builder.createOperation(result)->getResult(0);
}

mlir::Value* MLIRDeclaration::mlirGen(PostExp *postExp){
  IF_LOG Logger::print("MLIRGen - PostExp: %s @ %s\n", postExp->toChars(),
                       postExp->type->toChars());
  LOG_SCOPE;

  mlir::Value* e1 = mlirGen(postExp->e1);

  if(e1 == nullptr){
    _miss++;
    IF_LOG Logger::println("Unable to build PostExp '%s'", postExp->toChars());
    return nullptr;
  }


  StringRef opName;
  if(postExp->op == TOKplusplus)
  opName = "ldc.Plusplus";
  else if(postExp->op == TOKminusminus)
    opName = "ldc.MinusMinus";
  else {
    _miss++;
    return nullptr;
  }

  mlir::OperationState result(loc(postExp->loc), opName);
  result.addOperands(e1);
  result.addTypes(e1->getType()); // TODO: type
  result.addAttribute("value", builder.getI16IntegerAttr(1));
  return builder.createOperation(result)->getResult(0);

}

mlir::Value *MLIRDeclaration::mlirGen(Expression *expression, mlir::Block*
block) {
  IF_LOG Logger::println("MLIRCodeGen - Expression: '%s' | '%u'",
      expression->toChars(), expression->op);
  LOG_SCOPE
  this->_total++;

  if(block != nullptr)
    builder.setInsertionPointToEnd(block);

  const char *op_name = nullptr;
  auto location = loc(expression->loc);
  mlir::Value *e1 = nullptr;
  mlir::Value *e2 = nullptr;
  int op = expression->op;

  if(VarExp *varExp = expression->isVarExp())
    return mlirGen(varExp);
  else if(DeclarationExp *declarationExp = expression->isDeclarationExp())
    return mlirGen(declarationExp);
  else if(IntegerExp *integerExp = expression->isIntegerExp())
    return mlirGen(integerExp);
  else if(RealExp *realExp = expression->isRealExp())
    return mlirGen(realExp);
  else if(ConstructExp *constructExp = expression->isConstructExp())
    return mlirGen(constructExp);
  else if(AssignExp *assignExp = expression->isAssignExp())
    return mlirGen(assignExp);
  else if(CallExp *callExp = expression->isCallExp())
    return mlirGen(callExp);
  else if(ArrayLiteralExp *arrayLiteralExp = expression->isArrayLiteralExp())
    return mlirGen(arrayLiteralExp);
  else if(op >= 54 && op < 60)
    return mlirGen(expression, 1);
  else if(PostExp *postExp = expression->isPostExp())
    return mlirGen(postExp);
  else if(StringExp *stringExp = expression->isStringExp())
    return nullptr;//add mlirGen(stringlExp); //needs to implement with blocks
  else if (LogicalExp *logicalExp = expression->isLogicalExp())
    return nullptr; //add mlirGen(logicalExp); //needs to implement with blocks
  else if(expression->isAddExp() || expression->isAddAssignExp()){
    mlir::OperationState addi(location, "ldc.add");
    if(expression->isAddAssignExp()) {
        AddAssignExp *addAssignExp = expression->isAddAssignExp();
        e1 = mlirGen(addAssignExp->e1);
        e2 = mlirGen(addAssignExp->e2);
    }else {
        AddExp *add = expression->isAddExp();
        e1 = mlirGen(add->e1);
        e2 = mlirGen(add->e2);
    }
      mlir::D::AddOp addOp;
      addOp.build(&builder, addi, e1, e2);
      return builder.createOperation(addi)->getResult(0);
  } else if (expression->isMinExp() || expression->isMinAssignExp()) {
    if(expression->isMinAssignExp()){
      MinAssignExp* minAssignExp = expression->isMinAssignExp();
      e1 = mlirGen(minAssignExp->e1);
      e2 = mlirGen(minAssignExp->e2);
    }else{
      MinExp *min = expression->isMinExp();
      e1 = mlirGen(min->e1);
      e2 = mlirGen(min->e2);
    }
    op_name = "ldc.neg";
  } else if (expression->isMulExp() || expression->isMulAssignExp()) {
    if(expression->isMulAssignExp()){
      MulAssignExp* mulAssignExp = expression->isMulAssignExp();
      e1 = mlirGen(mulAssignExp->e1);
      e2 = mlirGen(mulAssignExp->e2);
    }else{
      MulExp *mul = expression->isMulExp();
      e1 = mlirGen(mul->e1);
      e2 = mlirGen(mul->e2);
    }
    op_name = "ldc.mul";
  } else if (expression->isDivExp() || expression->isDivAssignExp()) {
    if(expression->isDivAssignExp()){
      DivAssignExp* divAssignExp = expression->isDivAssignExp();
      e1 = mlirGen(divAssignExp->e1);
      e2 = mlirGen(divAssignExp->e2);
    }else{
      DivExp *div = expression->isDivExp();
      e1 = mlirGen(div->e1);
      e2 = mlirGen(div->e2);
    }
    op_name = "ldc.div";
  } else if (expression->isModExp() || expression->isModAssignExp()){
    if(expression->isModAssignExp()){
      ModAssignExp* modAssignExp = expression->isModAssignExp();
      e1 = mlirGen(modAssignExp->e1);
      e2 = mlirGen(modAssignExp->e2);
    }else{
      ModExp *mod = expression->isModExp();
      e1 = mlirGen(mod->e1);
      e2 = mlirGen(mod->e2);
    }
    op_name = "ldc.mod";
  } else if (expression->isAndExp() || expression->isAndAssignExp()){
    if(expression->isAndAssignExp()){
      AndAssignExp* andAssignExp = expression->isAndAssignExp();
      e1 = mlirGen(andAssignExp->e1);
      e2 = mlirGen(andAssignExp->e2);
    }else {
      AndExp *andExp = expression->isAndExp();
      e1 = mlirGen(andExp->e1);
      e2 = mlirGen(andExp->e2);
    }
    op_name = "ldc.and";
  } else if (expression->isOrExp() || expression->isOrAssignExp()) {
    if(expression->isOrAssignExp()){
      OrAssignExp* orAssignExp = expression->isOrAssignExp();
      e1 = mlirGen(orAssignExp->e1);
      e2 = mlirGen(orAssignExp->e2);
    }else {
      OrExp *orExp = expression->isOrExp();
      e1 = mlirGen(orExp->e1);
      e2 = mlirGen(orExp->e2);
    }
    op_name = "ldc.or";
  } else if (expression->isXorExp() || expression->isXorAssignExp()){
    if(expression->isXorAssignExp()){
      XorAssignExp* xorAssignExp = expression->isXorAssignExp();
      e1 = mlirGen(xorAssignExp->e1);
      e2 = mlirGen(xorAssignExp->e2);
    }else{
      XorExp *xorExp = expression->isXorExp();
      e1 = mlirGen(xorExp->e1);
      e2 = mlirGen(xorExp->e2);
    }
    op_name = "ldc.xor";
  }


  if (e1 != nullptr && e2 != nullptr) {
    mlir::OperationState result(location, op_name);
    result.addTypes(e1->getType()); // TODO: This works?
    result.addOperands({e1, e2});
    return builder.createOperation(result)->getResult(0);
  }
  _miss++;
    IF_LOG Logger::println("Unable to recoganize the Expression: '%s' : '%u': '%s'",
                    expression->toChars(), expression->op,
                    expression->type->toChars());
    return nullptr;
}

void MLIRDeclaration::mlirGen(TemplateInstance *decl) {
  IF_LOG Logger::println("MLIRCodeGen - TemplateInstance: '%s'",
     decl->toPrettyChars());
  LOG_SCOPE

  if (decl->ir->isDefined()) {
    Logger::println("Already defined, skipping.");
    return;
  }
  decl->ir->setDefined();

  if (isError(decl)) {
    Logger::println("Has errors, skipping.");
    return;
  }

  if (!decl->members) {
    Logger::println("Has no members, skipping.");
    return;
  }

  // Force codegen if this is a templated function with pragma(inline, true).
  if ((decl->members->dim == 1) && ((*decl->members)[0]->isFuncDeclaration()) &&
      ((*decl->members)[0]->isFuncDeclaration()->inlining == PINLINEalways)) {
    Logger::println("needsCodegen() == false, but function is marked with "
                    "pragma(inline, true), so it really does need "
                    "codegen.");
  } else {
    // FIXME: This is #673 all over again.
    if (!decl->needsCodegen()) {
      Logger::println("Does not need codegen, skipping.");
      return;
    }
    if (irState->dcomputetarget && (decl->tempdecl == Type::rtinfo ||
                                decl->tempdecl == Type::rtinfoImpl)) {
      // Emitting object.RTInfo(Impl) template instantiations in dcompute
      // modules would require dcompute support for global variables.
      Logger::println("Skipping object.RTInfo(Impl) template instantiations "
                    "in dcompute modules.");
      return;
    }
  }

  for (auto &m : *decl->members) {
    if (m->isDeclaration())
      mlirGen(m->isDeclaration());
    else {
      IF_LOG Logger::println("MLIRGEN Has to be implemented for: '%s'",
                             m->toChars());
      _miss++;
    }
  }
}

mlir::Type MLIRDeclaration::get_MLIRtype(Expression* expression){
    if(expression == nullptr)
        return mlir::NoneType::get(&context);

    _total++;
    auto basetype = expression->type->toBasetype();
    if (basetype->ty == Tchar || basetype->ty == Twchar ||
        basetype->ty == Tdchar || basetype->ty == Tnull ||
        basetype->ty == Tvoid || basetype->ty == Tnone) {
        return mlir::NoneType::get(&context); //TODO: Build these types on DDialect
    }else if(basetype->ty == Tbool){
      return builder.getIntegerType(1);
    }else if (basetype->ty == Tint8){
      return builder.getIntegerType(8);
    }else if(basetype->ty == Tint16){
      return builder.getIntegerType(16);
    }else if(basetype->ty == Tint32){
      return builder.getIntegerType(32);
    }else if(basetype->ty == Tint64){
      return builder.getIntegerType(64);
    }else if(basetype->ty == Tint128) {
        return builder.getIntegerType(128);
    } else if (basetype->ty == Tfloat32){
      return builder.getF32Type();
    }else if(basetype->ty == Tfloat64){
      return builder.getF64Type();
    }else if(basetype->ty == Tfloat80) {
      _miss++;     //TODO: Build F80 type on DDialect
    } else if (basetype->ty == Tvector) {
        mlir::TensorType tensor;
        return tensor;
    } else {
        _miss++;
        MLIRDeclaration *declaration = new MLIRDeclaration(irState, nullptr,
                                                           context, builder, symbolTable,_total, _miss);
        mlir::Value *value = declaration->mlirGen(expression);
        return value->getType();
    }
    _miss++;
    return nullptr;
}


#endif //LDC_MLIR_ENABLED
