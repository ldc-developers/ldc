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
    llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable, unsigned
    &total, unsigned &miss) : irState
    (irs), module(m), context(context), builder(builder_), symbolTable
    (symbolTable), _total(total), _miss(miss){}
    //Constructor

MLIRDeclaration::~MLIRDeclaration() = default;

mlir::Value MLIRDeclaration::mlirGen(Declaration *declaration){
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

mlir::Value MLIRDeclaration::mlirGen(VarDeclaration *vd){
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

mlir::Value MLIRDeclaration::DtoAssignMLIR(mlir::Location Loc,
            mlir::Value lhs, mlir::Value rhs,  StringRef lhs_name,
             StringRef rhs_name, int op, bool canSkipPostblit, Type* t1,
             Type* t2){
  IF_LOG Logger::println("MLIRCodeGen - DtoAssignMLIR:");
  IF_LOG Logger::println("lhs: %s @ '%s'", lhs_name.data(), t1->toChars());
  IF_LOG Logger::println("rhs: %s @ '%s'", rhs_name.data(), t2->toChars());
  LOG_SCOPE;

  assert(t1->ty != Tvoid && "Cannot assign values of type void.");

  if (t1->ty == Tbool) {
    IF_LOG Logger::println("DtoAssignMLIR == Tbool"); //TODO: DtoStoreZextI8
  }

 // lhs = rhs; TODO: Verify whats it's better
  return rhs;
}


////////////////////////////////////////////////////////////////////////////////
// Expressions to be evaluated

mlir::Value MLIRDeclaration::mlirGen(AddExp *addExp, AddAssignExp *addAssignExp){
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;

  if(addAssignExp) {
    IF_LOG Logger::println("MLIRCodeGen - AddExp: '%s' + '%s' @ %s",
                           addAssignExp->e1->toChars(), addAssignExp->e2->toChars(),
                           addAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location addAssignLoc = loc(addAssignExp->loc);
    location = &addAssignLoc;
    e1 = mlirGen(addAssignExp->e1);
    e2 = mlirGen(addAssignExp->e2);
  }else if(addExp){
    IF_LOG Logger::println("MLIRCodeGen - AddExp: '%s' + '%s' @ %s",
                           addExp->e1->toChars(), addExp->e2->toChars(), addExp->type->toChars());
    LOG_SCOPE
    mlir::Location addExpLoc = loc(addExp->loc);
    location = &addExpLoc;
    e1 = mlirGen(addExp->e1);
    e2 = mlirGen(addExp->e2);
  }else{
    _miss++;
    return nullptr;
  }

  if((e1->getType().isF16() || e1->getType().isF32() || e1->getType().isF64())
     && (e1->getType() == e2->getType())) {
    result = builder.create<mlir::AddFOp>(*location, e1,e2).getResult();
  }else
  if((e1->getType().isInteger(8) || e1->getType().isInteger(16) ||
      e1->getType().isInteger(32) || e1->getType().isInteger(64)) &&
      (e1->getType() == e2->getType())) {
    result = builder.create<mlir::AddIOp>(*location, e1, e2).getResult();
  }else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(AndExp *andExp, AndAssignExp *andAssignExp){
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;

  if(andAssignExp) {
    IF_LOG Logger::println("MLIRCodeGen - AndExp: '%s' + '%s' @ %s",
                           andAssignExp->e1->toChars(), andAssignExp->e2->toChars(),
                           andAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location andAssignLoc = loc(andAssignExp->loc);
    location = &andAssignLoc;
    e1 = mlirGen(andAssignExp->e1);
    e2 = mlirGen(andAssignExp->e2);
  }else if(andExp){
    IF_LOG Logger::println("MLIRCodeGen - AndExp: '%s' + '%s' @ %s",
                           andExp->e1->toChars(), andExp->e2->toChars(), andExp->type->toChars());
    LOG_SCOPE
    mlir::Location andExpLoc = loc(andExp->loc);
    location = &andExpLoc;
    e1 = mlirGen(andExp->e1);
    e2 = mlirGen(andExp->e2);
  }else{
    _miss++;
    return nullptr;
  }

  if((e1->getType().isInteger(8) || e1->getType().isInteger(16) ||
      e1->getType().isInteger(32) || e1->getType().isInteger(64)) &&
     (e2->getType().isInteger(8) || e2->getType().isInteger(16) ||
      e2->getType().isInteger(32) || e2->getType().isInteger(64))) {
    result = builder.create<mlir::AndOp>(*location, e1, e2).getResult();
  }else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(ArrayLiteralExp *arrayLiteralExp){
  IF_LOG Logger::println("MLIRCodeGen - ArrayLiteralExp: '%s'",
                         arrayLiteralExp->toChars());
  LOG_SCOPE
  _total++;

  // IF_LOG Logger::println("Basis: '%s'",arrayLiteralExp->basis->toChars());
  IF_LOG Logger::println("Elements: '%s'", arrayLiteralExp->elements->toChars());

  // The type of this attribute is tensor the type of the literal with it's
  // shape .
  mlir::Type elementType = get_MLIRtype(arrayLiteralExp->elements->data[0]);
  bool isFloat = elementType.isF16() || elementType.isF32() || elementType.isF64();


  int size = 0;
  if(elementType.isInteger(1))
    size = 1;
  else if(elementType.isInteger(8))
    size = 8;
  else if(elementType.isInteger(16))
    size = 16;
  else if(elementType.isInteger(32))
    size = 32;
  else if(elementType.isInteger(64))
    size = 64;
  else if(!isFloat)
  IF_LOG Logger::println("MLIR doesn't support integer of type different "
                         "from 1,8,16,32,64");

  std::vector<mlir::APInt> data;
  std::vector<mlir::APFloat> dataF; //Used by float and double
  mlir::Location Loc = loc(arrayLiteralExp->loc);

  for(auto e : *arrayLiteralExp->elements){
    if(elementType.isInteger(size)) {
      mlir::APInt integer(size, e->toInteger(), !e->type->isunsigned());
      data.push_back(integer);
    } else if(isFloat) {
      if(elementType.isF64()){
        mlir::APFloat apFloat((double)e->toReal());
        dataF.push_back(apFloat);
      } else{
        mlir::APFloat apFloat((float)e->toReal());
        dataF.push_back(apFloat);
      }
    } else {
      _miss++;
      return nullptr;
    }
  }

  //For now lets assume one-dimensional arrays
  std::vector<int64_t> dims;
  //dims.push_back(1);
  if(elementType.isInteger(size))
    dims.push_back(data.size());
  else if (isFloat)
    dims.push_back(dataF.size());

  //Getting the shape of the tensor. Ex.: tensor<4xf64>
  auto dataType = mlir::RankedTensorType::get(dims, elementType);

  // Set the actual attribute that holds the list of values for this
  // tensor literal and build the operation.
  if(elementType.isInteger(size)) {
    auto dataAttributes = mlir::DenseElementsAttr::get(dataType, data);
    return builder.create<mlir::ConstantOp>(Loc, dataAttributes.template
                                                  cast<mlir::DenseIntElementsAttr>());
  } else if(isFloat){
    auto dataAttributes = mlir::DenseFPElementsAttr::get(dataType, dataF);
    if(elementType.isF64())
      return builder.create<mlir::ConstantOp>(Loc, dataAttributes);
    else
      return builder.create<mlir::ConstantOp>(Loc, dataAttributes);
  } else {
    _miss++;
    Logger::println("Unable to build ArrayLiteralExp: '%s'",
        arrayLiteralExp->toChars());
  }
  return nullptr;
}

mlir::Value MLIRDeclaration::mlirGen(AssignExp *assignExp){
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
      mlir::Value rhs = mlirGen(assignExp->e2);
      //mlir::Value *lhs = mlirGen(assignExp->e1);
      mlir::Value value;

      if (!rhs) {
        _miss++;
        return nullptr;
      }
      //TODO: Create test to evaluate this case and transform it into dialect
      // operation
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
    return DtoAssignMLIR(loc(assignExp->loc),  mlirGen(assignExp->e1),
                mlirGen(assignExp->e2), assignExp->e1->toChars(),
                assignExp->e2->toChars(), assignExp->op, !lvalueElem, t1, t2);


  //check if it is a declared variable
 mlir::Value lhs = nullptr;
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

mlir::Value MLIRDeclaration::mlirGen(CallExp *callExp){
  IF_LOG Logger::println("MLIRCodeGen - CallExp: '%s'", callExp->toChars());
  LOG_SCOPE
  _total++;

  std::vector<mlir::Type> types;
  StringRef functionName = callExp->f->mangleString;

  // Codegen the operands first.
  llvm::SmallVector<mlir::Value, 4> operands;
  for(auto exp : *callExp->arguments){
    auto arg = mlirGen(exp);
    if(!arg) {
      _miss++;
      return nullptr;
    }
    operands.push_back(arg);
  }
  //Get the return type
  mlir::Type ret = nullptr;
  if(ReturnStatement *returnStatement = *callExp->f->returns->data)
    ret = get_MLIRtype(returnStatement->exp);

  if(ret)
    types.push_back(ret);
  else
    types.push_back(mlir::NoneType::get(&context));

  llvm::ArrayRef<mlir::Type> ret_type(types);
  return builder.create<mlir::CallOp>(loc(callExp->loc), functionName,
      ret_type, operands).getResult(0);
}

mlir::Value MLIRDeclaration::mlirGen(CastExp *castExp){
  IF_LOG Logger::print("MLIRCodeGen - CastExp: %s @ %s\n", castExp->toChars(),
                       castExp->type->toChars());
  LOG_SCOPE;

  // Getting mlir location
  mlir::Location location = loc(castExp->loc);

  // get the value to cast
  mlir::Value result;
  result = mlirGen(castExp->e1);

  // handle cast to void (usually created by frontend to avoid "has no effect"
  // error)
  if (castExp->to == Type::tvoid) {
    result = nullptr;
    return result;
  }
  // cast it to the 'to' type, if necessary
  if (castExp->to->equals(castExp->e1->type)) {
    return result;
  }

  // paint the type, if necessary
  if (!castExp->type->equals(castExp->to)) {
    //result = DtoPaintType(castExp->loc, result, castExp->type); //TODO:
    IF_LOG Logger::println("Unable to paint the type, function not "
                           "implemented");
    _miss++;
  }

  auto type = get_MLIRtype(castExp);
  if(((result.getType().isF16() || result.getType().isF32()) && type.isF64())||
      (result.getType().isF16() && type.isF32()))
    return builder.create<mlir::FPExtOp>(location, result, type);

  return builder.create<mlir::D::CastOp>(loc(castExp->loc), type, result);
}

mlir::Value MLIRDeclaration::mlirGen(ConstructExp *constructExp){
  IF_LOG Logger::println("MLIRCodeGen - ConstructExp: '%s'", constructExp->toChars());
  LOG_SCOPE
  _total++;
  //mlir::Value *lhs = mlirGen(constructExp->e1);
  mlir::Value rhs = mlirGen(constructExp->e2);

  if (failed(declare(constructExp->e1->toChars(), rhs))){
    _miss++;
    return nullptr;
  }
  return rhs;
}

mlir::Value MLIRDeclaration::mlirGen(DeclarationExp *decl_exp){
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

mlir::Value MLIRDeclaration::mlirGen(DivExp *divExp, DivAssignExp *divAssignExp) {
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;
  bool isSigned = true;

  if (divAssignExp) {
    IF_LOG Logger::println("MLIRCodeGen - DivExp: '%s' + '%s' @ %s",
                           divAssignExp->e1->toChars(),
                           divAssignExp->e2->toChars(),
                           divAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location divAssignLoc = loc(divAssignExp->loc);
    location = &divAssignLoc;
    e1 = mlirGen(divAssignExp->e1);
    e2 = mlirGen(divAssignExp->e2);
    if(divAssignExp->e1->type->isunsigned() || divAssignExp->e1->type->isunsigned())
      isSigned = false;
  } else if (divExp) {
    IF_LOG Logger::println("MLIRCodeGen - DivExp: '%s' + '%s' @ %s",
                           divExp->e1->toChars(), divExp->e2->toChars(),
                           divExp->type->toChars());
    LOG_SCOPE
    mlir::Location divExpLoc = loc(divExp->loc);
    location = &divExpLoc;
    e1 = mlirGen(divExp->e1);
    e2 = mlirGen(divExp->e2);
    if(divExp->e1->type->isunsigned() || divExp->e1->type->isunsigned())
      isSigned = false;
  } else {
    _miss++;
    return nullptr;
  }

  if((e1->getType().isF16() || e1->getType().isF32() || e1->getType().isF64())
     && (e2->getType().isF16() || e2->getType().isF32() || e2->getType().isF64())) {
    result = builder.create<mlir::DivFOp>(*location, e1, e2).getResult();
  }else
  if((e1->getType().isInteger(8) || e1->getType().isInteger(16) ||
      e1->getType().isInteger(32) || e1->getType().isInteger(64)) &&
     (e2->getType().isInteger(8) || e2->getType().isInteger(16) ||
      e2->getType().isInteger(32) || e2->getType().isInteger(64))) {
    if(isSigned)
      result = builder.create<mlir::SignedDivIOp>(*location, e1, e2).getResult();
    else
      result = builder.create<mlir::UnsignedDivIOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(Expression *expression, int func){
  CmpExp *cmpExp = static_cast<CmpExp*>(expression);
  IF_LOG Logger::println("MLIRCodeGen - CmpExp: '%s'", cmpExp->toChars());
  _total++;

  mlir::Location location = loc(cmpExp->loc);
  mlir::Value e1 = mlirGen(cmpExp->e1);
  mlir::Value e2 = mlirGen(cmpExp->e2);
  Type *t = cmpExp->e1->type->toBasetype();
  mlir::CmpIPredicate predicate;
  mlir::CmpFPredicate predicateF;

  bool isFloat = e1->getType().isF64() || e1->getType().isF32() ||
                                                        e1->getType().isBF16();

  switch(cmpExp->op){
    case TOKlt:
      if(isFloat)
        predicateF = mlir::CmpFPredicate::ULT;
      else
        predicate = t->isunsigned() ? mlir::CmpIPredicate::ult : mlir::CmpIPredicate::slt;
      break;
    case TOKle:
      if(isFloat)
        predicateF = mlir::CmpFPredicate::ULE;
      else
        predicate = t->isunsigned() ? mlir::CmpIPredicate::ule : mlir::CmpIPredicate::sle;
      break;
    case TOKgt:
      if(isFloat)
        predicateF = mlir::CmpFPredicate::UGT;
      else
        predicate = t->isunsigned() ? mlir::CmpIPredicate::ugt : mlir::CmpIPredicate::sgt;
      break;
    case TOKge:
      if(isFloat)
        predicateF = mlir::CmpFPredicate::UGE;
      else
        predicate = t->isunsigned() ? mlir::CmpIPredicate::uge : mlir::CmpIPredicate::sge;
      break;
    case TOKequal:
      if(isFloat)
        predicateF = mlir::CmpFPredicate::UEQ;
      else
        predicate = mlir::CmpIPredicate::eq;
      break;
    case TOKnotequal:
      if(isFloat)
        predicateF = mlir::CmpFPredicate::UNE;
      else
        predicate = mlir::CmpIPredicate::ne;
      break;
    default:
      _miss++;
      IF_LOG Logger::println("Invalid comparison operation");
      break;
  }

  if(isFloat)
    return builder.create<mlir::CmpFOp>(location, predicateF, e1,e2);
  else
    return builder.create<mlir::CmpIOp>(location, predicate, e1, e2);
}

mlir::Value MLIRDeclaration::mlirGen(IntegerExp *integerExp){
  _total++;
  dinteger_t dint = integerExp->value;
  Logger::println("MLIRGen - Integer: '%lu'", dint);
  auto type = get_MLIRtype(integerExp);
  auto location = loc(integerExp->loc);
    return builder.create<mlir::ConstantOp>(location, builder.getIntegerAttr(type, dint));
}

mlir::Value MLIRDeclaration::mlirGen(MinExp *minExp, MinAssignExp *minAssignExp) {
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;

  if (minAssignExp) {
    IF_LOG Logger::println("MLIRCodeGen - MinExp: '%s' + '%s' @ %s",
                           minAssignExp->e1->toChars(),
                           minAssignExp->e2->toChars(),
                           minAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location minAssignLoc = loc(minAssignExp->loc);
    location = &minAssignLoc;
    e1 = mlirGen(minAssignExp->e1);
    e2 = mlirGen(minAssignExp->e2);
  } else if (minExp) {
    IF_LOG Logger::println("MLIRCodeGen - MinExp: '%s' + '%s' @ %s",
                           minExp->e1->toChars(), minExp->e2->toChars(),
                           minExp->type->toChars());
    LOG_SCOPE
    mlir::Location minExpLoc = loc(minExp->loc);
    location = &minExpLoc;
    e1 = mlirGen(minExp->e1);
    e2 = mlirGen(minExp->e2);
  } else {
    _miss++;
    return nullptr;
  }

  if((e1->getType().isF16() || e1->getType().isF32() || e1->getType().isF64())
     && (e2->getType().isF16() || e2->getType().isF32() || e2->getType().isF64())) {
    result = builder.create<mlir::SubFOp>(*location, e1, e2).getResult();
  }else
  if((e1->getType().isInteger(8) || e1->getType().isInteger(16) ||
      e1->getType().isInteger(32) || e1->getType().isInteger(64)) &&
     (e2->getType().isInteger(8) || e2->getType().isInteger(16) ||
      e2->getType().isInteger(32) || e2->getType().isInteger(64))) {
    result = builder.create<mlir::SubIOp>(*location, e1, e2).getResult();
  }else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(ModExp *modExp, ModAssignExp *modAssignExp){
 mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;
  bool isSigned = true;

  if(modAssignExp) {
    IF_LOG Logger::println("MLIRCodeGen - ModExp: '%s' + '%s' @ %s",
                           modAssignExp->e1->toChars(), modAssignExp->e2->toChars(),
                           modAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location modAssignLoc = loc(modAssignExp->loc);
    location = &modAssignLoc;
    e1 = mlirGen(modAssignExp->e1);
    e2 = mlirGen(modAssignExp->e2);
    if(modAssignExp->e1->type->isunsigned() || modAssignExp->e1->type->isunsigned())
      isSigned = false;
  }else if(modExp){
    IF_LOG Logger::println("MLIRCodeGen - ModExp: '%s' + '%s' @ %s",
                           modExp->e1->toChars(), modExp->e2->toChars(), modExp->type->toChars());
    LOG_SCOPE
    mlir::Location modExpLoc = loc(modExp->loc);
    location = &modExpLoc;
    e1 = mlirGen(modExp->e1);
    e2 = mlirGen(modExp->e2);
    if(modExp->e1->type->isunsigned() || modExp->e1->type->isunsigned())
      isSigned = false;
  }else{
    _miss++;
    return nullptr;
  }

  if((e1->getType().isF16() || e1->getType().isF32() || e1->getType().isF64())
     && (e2->getType().isF16() || e2->getType().isF32() || e2->getType().isF64())) {
    result = builder.create<mlir::RemFOp>(*location, e1, e2).getResult();
  }else
  if((e1->getType().isInteger(8) || e1->getType().isInteger(16) ||
      e1->getType().isInteger(32) || e1->getType().isInteger(64)) &&
     (e2->getType().isInteger(8) || e2->getType().isInteger(16) ||
      e2->getType().isInteger(32) || e2->getType().isInteger(64))) {
    if(isSigned)
      result = builder.create<mlir::SignedRemIOp>(*location, e1, e2).getResult();
    else
      result = builder.create<mlir::UnsignedRemIOp>(*location, e1, e2).getResult();
  }else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(MulExp *mulExp, MulAssignExp *mulAssignExp) {
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;

  if (mulAssignExp) {
    IF_LOG Logger::println("MLIRCodeGen - MulExp: '%s' + '%s' @ %s",
                           mulAssignExp->e1->toChars(),
                           mulAssignExp->e2->toChars(),
                           mulAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location mulAssignLoc = loc(mulAssignExp->loc);
    location = &mulAssignLoc;
    e1 = mlirGen(mulAssignExp->e1);
    e2 = mlirGen(mulAssignExp->e2);
  } else if (mulExp) {
    IF_LOG Logger::println("MLIRCodeGen - MulExp: '%s' + '%s' @ %s",
                           mulExp->e1->toChars(), mulExp->e2->toChars(),
                           mulExp->type->toChars());
    LOG_SCOPE
    mlir::Location mulExpLoc = loc(mulExp->loc);
    location = &mulExpLoc;
    e1 = mlirGen(mulExp->e1);
    e2 = mlirGen(mulExp->e2);
  } else {
    _miss++;
    return nullptr;
  }

  if((e1->getType().isF16() || e1->getType().isF32() || e1->getType().isF64())
     && (e2->getType().isF16() || e2->getType().isF32() || e2->getType().isF64())) {
    result = builder.create<mlir::MulFOp>(*location, e1, e2).getResult();
  }else
  if((e1->getType().isInteger(8) || e1->getType().isInteger(16) ||
      e1->getType().isInteger(32) || e1->getType().isInteger(64)) &&
     (e2->getType().isInteger(8) || e2->getType().isInteger(16) ||
      e2->getType().isInteger(32) || e2->getType().isInteger(64))) {
    result = builder.create<mlir::MulIOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(OrExp *orExp, OrAssignExp *orAssignExp){
 mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;

  if(orAssignExp) {
    IF_LOG Logger::println("MLIRCodeGen - OrExp: '%s' + '%s' @ %s",
                           orAssignExp->e1->toChars(),
                           orAssignExp->e2->toChars(),
                           orAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location orAssignLoc = loc(orAssignExp->loc);
    location = &orAssignLoc;
    e1 = mlirGen(orAssignExp->e1);
    e2 = mlirGen(orAssignExp->e2);
  }else if(orExp){
    IF_LOG Logger::println("MLIRCodeGen - OrExp: '%s' + '%s' @ %s",
                           orExp->e1->toChars(), orExp->e2->toChars(),
                           orExp->type->toChars());
    LOG_SCOPE
    mlir::Location orExpLoc = loc(orExp->loc);
    location = &orExpLoc;
    e1 = mlirGen(orExp->e1);
    e2 = mlirGen(orExp->e2);
  }else{
    _miss++;
    return nullptr;
  }

  if((e1->getType().isInteger(8) || e1->getType().isInteger(16) ||
      e1->getType().isInteger(32) || e1->getType().isInteger(64)) &&
     (e2->getType().isInteger(8) || e2->getType().isInteger(16) ||
      e2->getType().isInteger(32) || e2->getType().isInteger(64))) {
    result = builder.create<mlir::OrOp>(*location, e1, e2).getResult();
  }else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(PostExp *postExp){
  IF_LOG Logger::print("MLIRGen - PostExp: %s @ %s\n", postExp->toChars(),
                       postExp->type->toChars());
  LOG_SCOPE;

 mlir::Value e1 = mlirGen(postExp->e1);

  if(e1 == nullptr){
    _miss++;
    IF_LOG Logger::println("Unable to build PostExp '%s'", postExp->toChars());
    return nullptr;
  }

  int size = 0;
  if(e1->getType().isInteger(1))
    size = 1;
  else if(e1->getType().isInteger(8))
    size = 8;
  else if(e1->getType().isInteger(16))
    size = 16;
  else if(e1->getType().isInteger(32))
    size = 32;
  else if(e1->getType().isInteger(64))
    size = 64;
  else
  IF_LOG Logger::println("MLIR doesn't support integer of type different "
                         "from 1,8,16,32,64");

 mlir::Value e2 = nullptr;
  mlir::Location location = loc(postExp->loc);
  if(e1->getType().isF32() || e1->getType().isF16() || e1->getType().isF64())
    e2 = builder.create<mlir::ConstantOp>(location, builder.getFloatAttr
                                                   (e1->getType(), 1.00));
  else if(e1->getType().isInteger(size))
    e2 = builder.create<mlir::ConstantOp>(location, builder.getIntegerAttr
                                                   (e1->getType(), 1));
  if(postExp->op == TOKplusplus) {
    if(e1->getType().isF32() || e1->getType().isF16() || e1->getType().isF64())
      return builder.create<mlir::AddFOp>(location, e1, e2);
    else if(e1->getType().isInteger(size))
      return builder.create<mlir::AddIOp>(location, e1, e2);
  } else if(postExp->op == TOKminusminus) {
    if (e1->getType().isF32() || e1->getType().isF16() || e1->getType().isF64())
      return builder.create<mlir::SubFOp>(location, e1, e2);
    else if (e1->getType().isInteger(size))
      return builder.create<mlir::SubIOp>(location, e1, e2);
  } else {
    _miss++;
    return nullptr;
  }
  return nullptr;
}

mlir::Value MLIRDeclaration::mlirGen(RealExp *realExp){
  _total++;
  real_t dfloat = realExp->value;
  Logger::println("MLIRCodeGen - RealExp: '%Lf'", dfloat);
  mlir::Location Loc = loc(realExp->loc);
  auto type = get_MLIRtype(realExp);
  return builder.create<mlir::ConstantOp>(Loc, builder.getFloatAttr(type,
                                                                       dfloat));

}

mlir::Value MLIRDeclaration::mlirGen(VarExp *varExp){
  _total++;
  Logger::println("MLIRCodeGen - VarExp: '%s'", varExp->var->toChars());
  auto var = symbolTable.lookup(varExp->var->toChars());
  if (!var) {
    IF_LOG Logger::println("Undeclared VarExp: '%s' | '%u'", varExp->toChars(), varExp->op);
    auto type = builder.getIntegerType(32);
    return builder.create<mlir::ConstantOp>(loc(varExp->loc),builder.getIntegerAttr(type, 0));
  }
  return var;
}

mlir::Value MLIRDeclaration::mlirGen(XorExp *xorExp, XorAssignExp *xorAssignExp){
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;

  if(xorAssignExp) {
    IF_LOG Logger::println("MLIRCodeGen - XorExp: '%s' + '%s' @ %s",
                           xorAssignExp->e1->toChars(),
                           xorAssignExp->e2->toChars(),
                           xorAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location xorAssignLoc = loc(xorAssignExp->loc);
    location = &xorAssignLoc;
    e1 = mlirGen(xorAssignExp->e1);
    e2 = mlirGen(xorAssignExp->e2);
  }else if(xorExp){
    IF_LOG Logger::println("MLIRCodeGen - XorExp: '%s' + '%s' @ %s",
                           xorExp->e1->toChars(), xorExp->e2->toChars(),
                           xorExp->type->toChars());
    LOG_SCOPE
    mlir::Location xorExpLoc = loc(xorExp->loc);
    location = &xorExpLoc;
    e1 = mlirGen(xorExp->e1);
    e2 = mlirGen(xorExp->e2);
  }else{
    _miss++;
    return nullptr;
  }

  if((e1->getType().isInteger(8) || e1->getType().isInteger(16) ||
      e1->getType().isInteger(32) || e1->getType().isInteger(64)) &&
     (e2->getType().isInteger(8) || e2->getType().isInteger(16) ||
      e2->getType().isInteger(32) || e2->getType().isInteger(64))) {
    result = builder.create<mlir::XOrOp>(*location, e1, e2).getResult();
  }else {
    _miss++;
    return nullptr;
  }

  return result;
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
      IF_LOG Logger::println("MLIRGen Has to be implemented for: '%s'",
                             m->toChars());
      _miss++;
    }
  }
}

mlir::Value MLIRDeclaration::mlirGen(Expression *expression, mlir::Block* block) {
  IF_LOG Logger::println("MLIRCodeGen - Expression: '%s' | '%u' -> '%s'",
                         expression->toChars(), expression->op, expression->type->toChars());
  LOG_SCOPE
  this->_total++;

  if(block != nullptr)
    builder.setInsertionPointToEnd(block);

  int op = expression->op;

  if(VarExp *varExp = expression->isVarExp())
    return mlirGen(varExp);
  else if (DeclarationExp *declarationExp = expression->isDeclarationExp())
    return mlirGen(declarationExp);
  else if (CastExp *castExp = expression->isCastExp())
    return mlirGen(castExp);
  else if (IntegerExp *integerExp = expression->isIntegerExp())
    return mlirGen(integerExp);
  else if (RealExp *realExp = expression->isRealExp())
    return mlirGen(realExp);
  else if (ConstructExp *constructExp = expression->isConstructExp())
    return mlirGen(constructExp);
  else if (AssignExp *assignExp = expression->isAssignExp())
    return mlirGen(assignExp);
  else if (CallExp *callExp = expression->isCallExp())
    return mlirGen(callExp);
  else if (ArrayLiteralExp *arrayLiteralExp = expression->isArrayLiteralExp())
    return mlirGen(arrayLiteralExp);
  else if (op >= 54 && op < 60)
    return mlirGen(expression, 1);
  else if (PostExp *postExp = expression->isPostExp())
    return mlirGen(postExp);
  else if (StringExp *stringExp = expression->isStringExp())
    return nullptr;//add mlirGen(stringlExp); //needs to implement with blocks
  else if (LogicalExp *logicalExp = expression->isLogicalExp())
    return nullptr; //add mlirGen(logicalExp); //needs to implement with blocks
  else if (expression->isAddExp() || expression->isAddAssignExp())
    return mlirGen(expression->isAddExp(), expression->isAddAssignExp());
  else if (expression->isMinExp() || expression->isMinAssignExp())
    return mlirGen(expression->isMinExp(), expression->isMinAssignExp());
  else if (expression->isMulExp() || expression->isMulAssignExp())
    return mlirGen(expression->isMulExp(), expression->isMulAssignExp());
  else if (expression->isDivExp() || expression->isDivAssignExp())
    return mlirGen(expression->isDivExp(), expression->isDivAssignExp());
  else if (expression->isModExp() || expression->isModAssignExp())
    return mlirGen(expression->isModExp(), expression->isModAssignExp());
  else if (expression->isAndExp() || expression->isAndAssignExp())
    return mlirGen(expression->isAndExp(),expression->isAndAssignExp());
  else if (expression->isOrExp() || expression->isOrAssignExp())
    return mlirGen(expression->isOrExp(),expression->isOrAssignExp());
  else if (expression->isXorExp() || expression->isXorAssignExp())
    return mlirGen(expression->isXorExp(),expression->isXorAssignExp());

  _miss++;
  IF_LOG Logger::println("Unable to recoganize the Expression: '%s' : '%u': '%s'",
                         expression->toChars(), expression->op,
                         expression->type->toChars());
  return nullptr;
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
        mlir::Value value = declaration->mlirGen(expression);
        return value.getType();
    }
    _miss++;
    return nullptr;
}


#endif //LDC_MLIR_ENABLED
