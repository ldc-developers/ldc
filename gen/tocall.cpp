//===-- tocall.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/compiler.h"
#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/mtype.h"
#include "dmd/target.h"
#include "gen/abi.h"
#include "gen/classes.h"
#include "gen/dvalue.h"
#include "gen/funcgenstate.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/nested.h"
#include "gen/pragma.h"
#include "gen/tollvm.h"
#include "gen/runtime.h"
#include "ir/irfunction.h"
#include "ir/irtype.h"
#include "llvm/IR/LLVMContext.h"

////////////////////////////////////////////////////////////////////////////////

IrFuncTy &DtoIrTypeFunction(DValue *fnval) {
  if (DFuncValue *dfnval = fnval->isFunc()) {
    if (dfnval->func) {
      return getIrFunc(dfnval->func)->irFty;
    }
  }

  Type *type = stripModifiers(fnval->type->toBasetype());
  DtoType(type);
  assert(type->ctype);
  return type->ctype->getIrFuncTy();
}

TypeFunction *DtoTypeFunction(DValue *fnval) {
  Type *type = fnval->type->toBasetype();
  if (type->ty == Tfunction) {
    return static_cast<TypeFunction *>(type);
  }
  if (type->ty == Tdelegate) {
    // FIXME: There is really no reason why the function type should be
    // unmerged at this stage, but the frontend still seems to produce such
    // cases; for example for the uint(uint) next type of the return type of
    // (&zero)(), leading to a crash in DtoCallFunction:
    // ---
    // void test8198() {
    //   uint delegate(uint) zero() { return null; }
    //   auto a = (&zero)()(0);
    // }
    // ---
    // Calling merge() here works around the symptoms, but does not fix the
    // root cause.

    Type *next = merge(type->nextOf());
    assert(next->ty == Tfunction);
    return static_cast<TypeFunction *>(next);
  }

  llvm_unreachable("Cannot get TypeFunction* from non lazy/function/delegate");
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DtoCallableValue(DValue *fn) {
  Type *type = fn->type->toBasetype();
  if (type->ty == Tfunction) {
    return DtoRVal(fn);
  }
  if (type->ty == Tdelegate) {
    if (fn->isLVal()) {
      LLValue *dg = DtoLVal(fn);
      LLValue *funcptr = DtoGEP(dg, 0, 1);
      return DtoLoad(funcptr, ".funcptr");
    }
    LLValue *dg = DtoRVal(fn);
    assert(isaStruct(dg));
    return gIR->ir->CreateExtractValue(dg, 1, ".funcptr");
  }

  llvm_unreachable("Not a callable type.");
}

////////////////////////////////////////////////////////////////////////////////

LLFunctionType *DtoExtractFunctionType(LLType *type) {
  if (LLFunctionType *fty = isaFunction(type)) {
    return fty;
  }
  if (LLPointerType *pty = isaPointer(type)) {
    if (LLFunctionType *fty = isaFunction(pty->getElementType())) {
      return fty;
    }
  }
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

static void addExplicitArguments(std::vector<LLValue *> &args, AttrSet &attrs,
                                 IrFuncTy &irFty, LLFunctionType *calleeType,
                                 Expressions &argexps,
                                 Parameters *formalParams) {
  // Number of arguments added to the LLVM type that are implicit on the
  // frontend side of things (this, context pointers, etc.)
  const size_t implicitLLArgCount = args.size();

  // Number of formal arguments in the LLVM type (i.e. excluding varargs).
  const size_t formalLLArgCount = irFty.args.size();

  // Number of formal arguments in the D call expression (excluding varargs).
  const size_t formalDArgCount = Parameter::dim(formalParams);

  // The number of explicit arguments in the D call expression (including
  // varargs), not all of which necessarily generate a LLVM argument.
  const size_t explicitDArgCount = argexps.size();

  // construct and initialize an IrFuncTyArg object for each vararg
  std::vector<IrFuncTyArg *> optionalIrArgs;
  for (size_t i = formalDArgCount; i < explicitDArgCount; i++) {
    Type *argType = argexps[i]->type;
    bool passByVal = gABI->passByVal(irFty.type, argType);

    llvm::AttrBuilder initialAttrs;
    if (passByVal) {
      initialAttrs.addAttribute(LLAttribute::ByVal);
      if (auto alignment = DtoAlignment(argType))
        initialAttrs.addAlignmentAttr(alignment);
    } else {
      DtoAddExtendAttr(argType, initialAttrs);
    }

    optionalIrArgs.push_back(new IrFuncTyArg(argType, passByVal, initialAttrs));
    optionalIrArgs.back()->parametersIdx = i;
  }

  // let the ABI rewrite the IrFuncTyArg objects
  gABI->rewriteVarargs(irFty, optionalIrArgs);

  const size_t explicitLLArgCount = formalLLArgCount + optionalIrArgs.size();
  args.resize(implicitLLArgCount + explicitLLArgCount,
              static_cast<llvm::Value *>(nullptr));

  // Iterate the explicit arguments from left to right in the D source,
  // which is the reverse of the LLVM order if irFty.reverseParams is true.
  size_t dArgIndex = 0;
  for (size_t i = 0; i < explicitLLArgCount; ++i, ++dArgIndex) {
    const bool isVararg = (i >= formalLLArgCount);
    IrFuncTyArg *irArg = nullptr;
    if (isVararg) {
      irArg = optionalIrArgs[i - formalLLArgCount];
    } else {
      irArg = irFty.args[i];
    }

    // Make sure to evaluate argument expressions for which there's no LL
    // parameter (e.g., empty structs for some ABIs).
    if (irArg->parametersIdx < formalDArgCount) {
      for (; dArgIndex < irArg->parametersIdx; ++dArgIndex) {
        toElem(argexps[dArgIndex]);
      }
    }

    Expression *const argexp = argexps[dArgIndex];
    Parameter *const formalParam =
        isVararg ? nullptr : Parameter::getNth(formalParams, dArgIndex);

    // evaluate argument expression
    DValue *const dval = DtoArgument(formalParam, argexp);

    // load from lvalue/let TargetABI rewrite it/...
    llvm::Value *llVal = irFty.putArg(*irArg, dval, argexp->isLvalue(),
                                      dArgIndex == explicitDArgCount - 1);

    const size_t llArgIdx =
        implicitLLArgCount +
        (irFty.reverseParams ? explicitLLArgCount - i - 1 : i);
    llvm::Type *const paramType =
        (isVararg ? nullptr : calleeType->getParamType(llArgIdx));

    // Hack around LDC assuming structs and static arrays are in memory:
    // If the function wants a struct, and the argument value is a
    // pointer to a struct, load from it before passing it in.
    if (isaPointer(llVal) && DtoIsInMemoryOnly(argexp->type) &&
        ((!isVararg && !isaPointer(paramType)) ||
         (isVararg && !irArg->byref && !irArg->isByVal()))) {
      Logger::println("Loading struct type for function argument");
      llVal = DtoLoad(llVal);
    }

    // parameter type mismatch, this is hard to get rid of
    if (!isVararg && llVal->getType() != paramType) {
      IF_LOG {
        Logger::cout() << "arg:     " << *llVal << '\n';
        Logger::cout() << "expects: " << *paramType << '\n';
      }
      if (isaStruct(llVal)) {
        llVal = DtoAggrPaint(llVal, paramType);
      } else {
        llVal = DtoBitCast(llVal, paramType);
      }
    }

    args[llArgIdx] = llVal;
    attrs.addToParam(llArgIdx, irArg->attrs);

    if (isVararg) {
      delete irArg;
    }
  }

  for (; dArgIndex < explicitDArgCount; ++dArgIndex) {
    toElem(argexps[dArgIndex]);
  }
}

////////////////////////////////////////////////////////////////////////////////

static LLValue *getTypeinfoArrayArgumentForDVarArg(Expressions *argexps,
                                                   int begin) {
  IF_LOG Logger::println("doing d-style variadic arguments");
  LOG_SCOPE

  // number of non variadic args
  IF_LOG Logger::println("num non vararg params = %d", begin);

  const size_t numArgExps = argexps ? argexps->size() : 0;
  const size_t numVariadicArgs = numArgExps - begin;

  // build type info array
  LLType *typeinfotype = DtoType(getTypeInfoType());
  LLArrayType *typeinfoarraytype =
      LLArrayType::get(typeinfotype, numVariadicArgs);

  auto typeinfomem = new llvm::GlobalVariable(
      gIR->module, typeinfoarraytype, true, llvm::GlobalValue::InternalLinkage,
      nullptr, "._arguments.storage");
  IF_LOG Logger::cout() << "_arguments storage: " << *typeinfomem << '\n';

  std::vector<LLConstant *> vtypeinfos;
  vtypeinfos.reserve(numVariadicArgs);
  for (size_t i = begin; i < numArgExps; i++) {
    vtypeinfos.push_back(DtoTypeInfoOf((*argexps)[i]->type));
  }

  // apply initializer
  LLConstant *tiinits = LLConstantArray::get(typeinfoarraytype, vtypeinfos);
  typeinfomem->setInitializer(tiinits);

  // put data in d-array
  LLConstant *pinits[] = {
      DtoConstSize_t(numVariadicArgs),
      llvm::ConstantExpr::getBitCast(typeinfomem, getPtrToType(typeinfotype))};
  LLType *tiarrty = DtoType(getTypeInfoType()->arrayOf());
  tiinits = LLConstantStruct::get(isaStruct(tiarrty),
                                  llvm::ArrayRef<LLConstant *>(pinits));
  LLValue *typeinfoarrayparam = new llvm::GlobalVariable(
      gIR->module, tiarrty, true, llvm::GlobalValue::InternalLinkage, tiinits,
      "._arguments.array");

  return DtoLoad(typeinfoarrayparam);
}

////////////////////////////////////////////////////////////////////////////////

static LLType *getPtrToAtomicType(LLType *type) {
  switch (const size_t N = getTypeBitSize(type)) {
  case 8:
  case 16:
  case 32:
  case 64:
  case 128:
    return LLType::getIntNPtrTy(gIR->context(), static_cast<unsigned>(N));
  default:
    return nullptr;
  }
}

bool DtoLowerMagicIntrinsic(IRState *p, FuncDeclaration *fndecl, CallExp *e,
                            DValue *&result) {
  // va_start instruction
  if (fndecl->llvmInternal == LLVMva_start) {
    if (e->arguments->length < 1 || e->arguments->length > 2) {
      e->error("`va_start` instruction expects 1 (or 2) arguments");
      fatal();
    }
    DLValue *ap = toElem((*e->arguments)[0])->isLVal(); // va_list
    assert(ap);
    // variadic extern(D) function with implicit _argptr?
    if (LLValue *argptrMem = p->func()->_argptr) {
      DtoMemCpy(DtoLVal(ap), argptrMem); // ap = _argptr
    } else {
      LLValue *llAp = gABI->prepareVaStart(ap);
      p->ir->CreateCall(GET_INTRINSIC_DECL(vastart), llAp, "");
    }
    result = nullptr;
    return true;
  }

  // va_copy instruction
  if (fndecl->llvmInternal == LLVMva_copy) {
    if (e->arguments->length != 2) {
      e->error("`va_copy` instruction expects 2 arguments");
      fatal();
    }
    DLValue *dest = toElem((*e->arguments)[0])->isLVal(); // va_list
    assert(dest);
    DValue *src = toElem((*e->arguments)[1]);             // va_list
    gABI->vaCopy(dest, src);
    result = nullptr;
    return true;
  }

  // va_arg instruction
  if (fndecl->llvmInternal == LLVMva_arg) {
    if (e->arguments->length != 1) {
      e->error("`va_arg` instruction expects 1 argument");
      fatal();
    }
    if (DtoIsInMemoryOnly(e->type)) {
      e->error("`va_arg` instruction does not support structs and static arrays");
      fatal();
    }
    DLValue *ap = toElem((*e->arguments)[0])->isLVal(); // va_list
    assert(ap);
    LLValue *llAp = gABI->prepareVaArg(ap);
    LLType *llType = DtoType(e->type);
    result = new DImValue(e->type, p->ir->CreateVAArg(llAp, llType));
    return true;
  }

  // C alloca
  if (fndecl->llvmInternal == LLVMalloca) {
    if (e->arguments->length != 1) {
      e->error("`alloca` expects 1 argument");
      fatal();
    }
    Expression *exp = (*e->arguments)[0];
    DValue *expv = toElem(exp);
    if (expv->type->toBasetype()->ty != Tint32) {
      expv = DtoCast(e->loc, expv, Type::tint32);
    }
    result = new DImValue(e->type,
                          p->ir->CreateAlloca(LLType::getInt8Ty(p->context()),
                                              DtoRVal(expv), ".alloca"));
    return true;
  }

  // fence instruction
  if (fndecl->llvmInternal == LLVMfence) {
    if (e->arguments->length < 1 || e->arguments->length > 2) {
      e->error("`fence` instruction expects 1 (or 2) arguments");
      fatal();
    }
    auto atomicOrdering =
        static_cast<llvm::AtomicOrdering>((*e->arguments)[0]->toInteger());
    llvm::SyncScope::ID scope = llvm::SyncScope::System;
    if (e->arguments->length == 2) {
      scope = static_cast<llvm::SyncScope::ID>((*e->arguments)[1]->toInteger());
    }
    p->ir->CreateFence(atomicOrdering, scope);
    return true;
  }

  // atomic store instruction
  if (fndecl->llvmInternal == LLVMatomic_store) {
    if (e->arguments->length != 3) {
      e->error("atomic store instruction expects 3 arguments");
      fatal();
    }
    Expression *exp1 = (*e->arguments)[0];
    Expression *exp2 = (*e->arguments)[1];
    int atomicOrdering = (*e->arguments)[2]->toInteger();

    DValue *dval = toElem(exp1);
    LLValue *ptr = DtoRVal(exp2);
    LLType *pointeeType = ptr->getType()->getContainedType(0);

    LLValue *val = nullptr;
    if (pointeeType->isIntegerTy()) {
      val = DtoRVal(dval);
    } else if (auto intPtrType = getPtrToAtomicType(pointeeType)) {
      ptr = DtoBitCast(ptr, intPtrType);
      auto lval = makeLValue(exp1->loc, dval);
      val = DtoLoad(DtoBitCast(lval, intPtrType));
    } else {
      e->error(
          "atomic store only supports types of size 1/2/4/8/16 bytes, not `%s`",
          exp1->type->toChars());
      fatal();
    }

    llvm::StoreInst *ret = p->ir->CreateStore(val, ptr);
    ret->setAtomic(llvm::AtomicOrdering(atomicOrdering));
    ret->setAlignment(LLMaybeAlign(getTypeAllocSize(val->getType())));
    return true;
  }

  // atomic load instruction
  if (fndecl->llvmInternal == LLVMatomic_load) {
    if (e->arguments->length != 2) {
      e->error("atomic load instruction expects 2 arguments");
      fatal();
    }

    Expression *exp = (*e->arguments)[0];
    int atomicOrdering = (*e->arguments)[1]->toInteger();

    LLValue *ptr = DtoRVal(exp);
    LLType *pointeeType = ptr->getType()->getContainedType(0);
    Type *retType = exp->type->nextOf();

    if (!pointeeType->isIntegerTy()) {
      if (auto intPtrType = getPtrToAtomicType(pointeeType)) {
        ptr = DtoBitCast(ptr, intPtrType);
      } else {
        e->error("atomic load only supports types of size 1/2/4/8/16 bytes, "
                 "not `%s`",
                 retType->toChars());
        fatal();
      }
    }

    llvm::LoadInst *load = p->ir->CreateLoad(ptr);
    load->setAlignment(LLMaybeAlign(getTypeAllocSize(load->getType())));
    load->setAtomic(llvm::AtomicOrdering(atomicOrdering));
    llvm::Value *val = load;
    if (val->getType() != pointeeType) {
      val = DtoAllocaDump(val, retType);
      result = new DLValue(retType, val);
    } else {
      result = new DImValue(retType, val);
    }
    return true;
  }

  // cmpxchg instruction
  if (fndecl->llvmInternal == LLVMatomic_cmp_xchg) {
    if (e->arguments->length != 6) {
      e->error("`cmpxchg` instruction expects 6 arguments");
      fatal();
    }
    if (e->type->ty != Tstruct) {
      e->error("`cmpxchg` instruction returns a struct");
      fatal();
    }
    Expression *exp1 = (*e->arguments)[0];
    Expression *exp2 = (*e->arguments)[1];
    Expression *exp3 = (*e->arguments)[2];
    const auto successOrdering =
        llvm::AtomicOrdering((*e->arguments)[3]->toInteger());
    const auto failureOrdering =
        llvm::AtomicOrdering((*e->arguments)[4]->toInteger());
    const bool isWeak = (*e->arguments)[5]->toInteger() != 0;

    LLValue *ptr = DtoRVal(exp1);
    LLType *pointeeType = ptr->getType()->getContainedType(0);
    DValue *dcmp = toElem(exp2);
    DValue *dval = toElem(exp3);

    LLValue *cmp = nullptr;
    LLValue *val = nullptr;
    if (pointeeType->isIntegerTy()) {
      cmp = DtoRVal(dcmp);
      val = DtoRVal(dval);
    } else if (auto intPtrType = getPtrToAtomicType(pointeeType)) {
      ptr = DtoBitCast(ptr, intPtrType);
      auto cmpLVal = makeLValue(exp2->loc, dcmp);
      cmp = DtoLoad(DtoBitCast(cmpLVal, intPtrType));
      auto lval = makeLValue(exp3->loc, dval);
      val = DtoLoad(DtoBitCast(lval, intPtrType));
    } else {
      e->error(
          "`cmpxchg` only supports types of size 1/2/4/8/16 bytes, not `%s`",
          exp2->type->toChars());
      fatal();
    }

    auto ret = p->ir->CreateAtomicCmpXchg(ptr, cmp, val, successOrdering,
                                           failureOrdering);
    ret->setWeak(isWeak);

    // we return a struct; allocate on stack and store to both fields manually
    // (avoiding DtoAllocaDump() due to bad optimized codegen, most likely
    // because of i1)
    auto mem = DtoAlloca(e->type);
    DtoStore(p->ir->CreateExtractValue(ret, 0),
             DtoBitCast(DtoGEP(mem, 0u, 0), ptr->getType()));
    DtoStoreZextI8(p->ir->CreateExtractValue(ret, 1), DtoGEP(mem, 0, 1));

    result = new DLValue(e->type, mem);
    return true;
  }

  // atomicrmw instruction
  if (fndecl->llvmInternal == LLVMatomic_rmw) {
    if (e->arguments->length != 3) {
      e->error("`atomicrmw` instruction expects 3 arguments");
      fatal();
    }

    assert(fndecl->intrinsicName);
    static const char *ops[] = {"xchg", "add", "sub", "and",  "nand", "or",
                                "xor",  "max", "min", "umax", "umin", nullptr};

    int op = 0;
    for (;; ++op) {
      if (ops[op] == nullptr) {
        e->error("unknown `atomicrmw` operation `%s`",
                 fndecl->intrinsicName);
        fatal();
      }
      if (strcmp(fndecl->intrinsicName, ops[op]) == 0) {
        break;
      }
    }

    Expression *exp1 = (*e->arguments)[0];
    Expression *exp2 = (*e->arguments)[1];
    int atomicOrdering = (*e->arguments)[2]->toInteger();
    LLValue *ptr = DtoRVal(exp1);
    LLValue *val = DtoRVal(exp2);
    LLValue *ret =
        p->ir->CreateAtomicRMW(llvm::AtomicRMWInst::BinOp(op), ptr, val,
                               llvm::AtomicOrdering(atomicOrdering));
    result = new DImValue(exp2->type, ret);
    return true;
  }

  // bitop
  if (fndecl->llvmInternal == LLVMbitop_bt ||
      fndecl->llvmInternal == LLVMbitop_btr ||
      fndecl->llvmInternal == LLVMbitop_btc ||
      fndecl->llvmInternal == LLVMbitop_bts) {
    if (e->arguments->length != 2) {
      e->error("bitop intrinsic expects 2 arguments");
      fatal();
    }

    Expression *exp1 = (*e->arguments)[0];
    Expression *exp2 = (*e->arguments)[1];
    LLValue *ptr = DtoRVal(exp1);
    LLValue *bitnum = DtoRVal(exp2);

    unsigned bitmask = DtoSize_t()->getBitWidth() - 1;
    assert(bitmask == 31 || bitmask == 63);
    // auto q = cast(size_t*)ptr + (bitnum >> (64bit ? 6 : 5));
    LLValue *q = DtoBitCast(ptr, DtoSize_t()->getPointerTo());
    q = DtoGEP1(q, p->ir->CreateLShr(bitnum, bitmask == 63 ? 6 : 5), "bitop.q");

    // auto mask = 1 << (bitnum & bitmask);
    LLValue *mask =
        p->ir->CreateAnd(bitnum, DtoConstSize_t(bitmask), "bitop.tmp");
    mask = p->ir->CreateShl(DtoConstSize_t(1), mask, "bitop.mask");

    // auto result = (*q & mask) ? -1 : 0;
    LLValue *val =
        p->ir->CreateZExt(DtoLoad(q, "bitop.tmp"), DtoSize_t(), "bitop.val");
    LLValue *ret = p->ir->CreateAnd(val, mask, "bitop.tmp");
    ret = p->ir->CreateICmpNE(ret, DtoConstSize_t(0), "bitop.tmp");
    ret = p->ir->CreateSelect(ret, DtoConstInt(-1), DtoConstInt(0),
                              "bitop.result");

    if (fndecl->llvmInternal != LLVMbitop_bt) {
      llvm::Instruction::BinaryOps op;
      if (fndecl->llvmInternal == LLVMbitop_btc) {
        // *q ^= mask;
        op = llvm::Instruction::Xor;
      } else if (fndecl->llvmInternal == LLVMbitop_btr) {
        // *q &= ~mask;
        mask = p->ir->CreateNot(mask);
        op = llvm::Instruction::And;
      } else if (fndecl->llvmInternal == LLVMbitop_bts) {
        // *q |= mask;
        op = llvm::Instruction::Or;
      } else {
        llvm_unreachable("Unrecognized bitop intrinsic.");
      }

      LLValue *newVal = p->ir->CreateBinOp(op, val, mask, "bitop.new_val");
      newVal = p->ir->CreateTrunc(newVal, DtoSize_t(), "bitop.tmp");
      DtoStore(newVal, q);
    }

    result = new DImValue(e->type, ret);
    return true;
  }

  if (fndecl->llvmInternal == LLVMbitop_vld) {
    if (e->arguments->length != 1) {
      e->error("`bitop.vld` intrinsic expects 1 argument");
      fatal();
    }
    // TODO: Check types

    Expression *exp1 = (*e->arguments)[0];
    LLValue *ptr = DtoRVal(exp1);
    result = new DImValue(e->type, DtoVolatileLoad(ptr));
    return true;
  }

  if (fndecl->llvmInternal == LLVMbitop_vst) {
    if (e->arguments->length != 2) {
      e->error("`bitop.vst` intrinsic expects 2 arguments");
      fatal();
    }
    // TODO: Check types

    Expression *exp1 = (*e->arguments)[0];
    Expression *exp2 = (*e->arguments)[1];
    LLValue *ptr = DtoRVal(exp1);
    LLValue *val = DtoRVal(exp2);
    DtoVolatileStore(val, ptr);
    return true;
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////

class ImplicitArgumentsBuilder {
public:
  ImplicitArgumentsBuilder(std::vector<LLValue *> &args, AttrSet &attrs,
                           Loc &loc, DValue *fnval,
                           LLFunctionType *llCalleeType, Expressions *argexps,
                           Type *resulttype, LLValue *sretPointer)
      : args(args), attrs(attrs), loc(loc), fnval(fnval), argexps(argexps),
        resulttype(resulttype), sretPointer(sretPointer),
        // computed:
        isDelegateCall(fnval->type->toBasetype()->ty == Tdelegate),
        dfnval(fnval->isFunc()), irFty(DtoIrTypeFunction(fnval)),
        tf(DtoTypeFunction(fnval)),
        llArgTypesBegin(llCalleeType->param_begin()) {}

  void addImplicitArgs() {
    if (gABI->passThisBeforeSret(tf)) {
      addContext();
      addSret();
    } else {
      addSret();
      addContext();
    }

    addArguments();
  }

private:
  // passed:
  std::vector<LLValue *> &args;
  AttrSet &attrs;
  Loc &loc;
  DValue *const fnval;
  Expressions *const argexps;
  Type *const resulttype;
  LLValue *const sretPointer;

  // computed:
  const bool isDelegateCall;
  DFuncValue *const dfnval;
  IrFuncTy &irFty;
  TypeFunction *const tf;
  LLFunctionType::param_iterator llArgTypesBegin;

  // Adds an optional sret pointer argument.
  void addSret() {
    if (!irFty.arg_sret) {
      return;
    }

    size_t index = args.size();
    LLType *llArgType = *(llArgTypesBegin + index);

    LLValue *pointer = sretPointer;
    if (!pointer) {
      pointer = DtoRawAlloca(llArgType->getContainedType(0),
                             DtoAlignment(resulttype), ".sret_tmp");
    }

    args.push_back(pointer);
    attrs.addToParam(index, irFty.arg_sret->attrs);

    // verify that sret and/or inreg attributes are set
    const auto &sretAttrs = irFty.arg_sret->attrs;
    (void)sretAttrs;
    assert((sretAttrs.contains(LLAttribute::StructRet) ||
            sretAttrs.contains(LLAttribute::InReg)) &&
           "Sret arg not sret or inreg?");
  }

  // Adds an optional context/this pointer argument.
  void addContext() {
    bool thiscall = irFty.arg_this;
    bool nestedcall = irFty.arg_nest;

    if (!thiscall && !isDelegateCall && !nestedcall)
      return;

    size_t index = args.size();
    LLType *llArgType = *(llArgTypesBegin + index);

    if (dfnval && (dfnval->func->ident == Id::ensure ||
                   dfnval->func->ident == Id::require)) {
      // can be the this "context" argument for a contract invocation
      // (pass a pointer to the aggregate `this` pointer, which can naturally be
      // used as the contract's parent context in case the contract features
      // nested functions capturing `this` from the contract's parent)
      LLValue *thisptrLval = gIR->func()->thisArg;
      if (auto parentfd = dfnval->func->parent->isFuncDeclaration()) {
        if (auto iface = parentfd->parent->isInterfaceDeclaration()) {
          // an interface contract expects the interface pointer, not the
          //  class pointer
          Type *thistype = gIR->func()->decl->vthis->type;
          if (thistype != iface->type) {
            DImValue *dthis = new DImValue(thistype, DtoLoad(thisptrLval));
            thisptrLval = DtoAllocaDump(DtoCastClass(loc, dthis, iface->type));
          }
        }
      }
      LLValue *contextptr = DtoBitCast(thisptrLval, getVoidPtrType());
      args.push_back(contextptr);
    } else if (thiscall && dfnval && dfnval->vthis) {
      // ... or a normal 'this' argument
      LLValue *thisarg = DtoBitCast(dfnval->vthis, llArgType);
      args.push_back(thisarg);
    } else if (isDelegateCall) {
      // ... or a delegate context arg
      LLValue *ctxarg;
      if (fnval->isLVal()) {
        ctxarg = DtoLoad(DtoGEP(DtoLVal(fnval), 0u, 0), ".ptr");
      } else {
        ctxarg = gIR->ir->CreateExtractValue(DtoRVal(fnval), 0, ".ptr");
      }
      ctxarg = DtoBitCast(ctxarg, llArgType);
      args.push_back(ctxarg);
    } else if (nestedcall) {
      // ... or a nested function context arg
      if (dfnval) {
        LLValue *contextptr = DtoNestedContext(loc, dfnval->func);
        contextptr = DtoBitCast(contextptr, getVoidPtrType());
        args.push_back(contextptr);
      } else {
        args.push_back(llvm::UndefValue::get(getVoidPtrType()));
      }
    } else {
      error(loc, "Context argument required but none given");
      fatal();
    }

    // add attributes
    if (irFty.arg_this) {
      attrs.addToParam(index, irFty.arg_this->attrs);
    } else if (irFty.arg_nest) {
      attrs.addToParam(index, irFty.arg_nest->attrs);
    }

    if (irFty.arg_objcSelector) {
      assert(dfnval);
      const auto selector = dfnval->func->selector;
      assert(selector);
      LLGlobalVariable *selptr = gIR->objc.getMethVarRef(*selector);
      args.push_back(DtoBitCast(DtoLoad(selptr), getVoidPtrType()));
    }
  }

  // D vararg functions need a "TypeInfo[] _arguments" argument.
  void addArguments() {
    if (!irFty.arg_arguments) {
      return;
    }

    int numFormalParams = tf->parameterList.length();
    LLValue *argumentsArg =
        getTypeinfoArrayArgumentForDVarArg(argexps, numFormalParams);

    args.push_back(argumentsArg);
    attrs.addToParam(args.size() - 1, irFty.arg_arguments->attrs);
  }
};

////////////////////////////////////////////////////////////////////////////////

// FIXME: this function is a mess !
DValue *DtoCallFunction(Loc &loc, Type *resulttype, DValue *fnval,
                        Expressions *arguments, LLValue *sretPointer) {
  IF_LOG Logger::println("DtoCallFunction()");
  LOG_SCOPE

  // make sure the D callee type has been processed
  DtoType(fnval->type);

  // get func value if any
  DFuncValue *const dfnval = fnval->isFunc();

  // get function type info
  IrFuncTy &irFty = DtoIrTypeFunction(fnval);
  TypeFunction *const tf = DtoTypeFunction(fnval);
  Type *const returntype = tf->next;
  const TY returnTy = returntype->toBasetype()->ty;

  if (resulttype == nullptr) {
    resulttype = returntype;
  }

  // get callee llvm value
  LLValue *callable = DtoCallableValue(fnval);
  LLFunctionType *const callableTy =
      DtoExtractFunctionType(callable->getType());
  assert(callableTy);

  const auto callconv =
      gABI->callingConv(tf->linkage, tf, dfnval ? dfnval->func : nullptr);

  //     IF_LOG Logger::cout() << "callable: " << *callable << '\n';

  // parameter attributes
  AttrSet attrs;

  // return attrs
  attrs.addToReturn(irFty.ret->attrs);

  std::vector<LLValue *> args;
  args.reserve(irFty.args.size());

  // handle implicit arguments (sret, context/this, _arguments)
  ImplicitArgumentsBuilder iab(args, attrs, loc, fnval, callableTy, arguments,
                               resulttype, sretPointer);
  iab.addImplicitArgs();

  // handle explicit arguments

  Logger::println("doing normal arguments");
  IF_LOG {
    Logger::println("Arguments so far: (%d)", static_cast<int>(args.size()));
    Logger::indent();
    for (auto &arg : args) {
      Logger::cout() << *arg << '\n';
    }
    Logger::undent();
    Logger::cout() << "Function type: " << tf->toChars() << '\n';
    // Logger::cout() << "LLVM functype: " << *callable->getType() << '\n';
  }

  if (arguments) {
    addExplicitArguments(args, attrs, irFty, callableTy, *arguments,
                         tf->parameterList.parameters);
  }

  if (irFty.arg_objcSelector) {
    // Use runtime msgSend function bitcasted as original call
    const char *msgSend = gABI->objcMsgSendFunc(resulttype, irFty);
    LLType *t = callable->getType();
    callable = getRuntimeFunction(loc, gIR->module, msgSend);
    callable = DtoBitCast(callable, t);
  }

  // call the function
  LLCallSite call = gIR->CreateCallOrInvoke(callable, args, "", tf->isnothrow);

  // PGO: Insert instrumentation or attach profile metadata at indirect call
  // sites.
  if (!call.getCalledFunction()) {
    auto &PGO = gIR->funcGen().pgo;
    PGO.emitIndirectCallPGO(call.getInstruction(), callable);
  }

  // get return value
  const int sretArgIndex =
      (irFty.arg_sret && irFty.arg_this && gABI->passThisBeforeSret(tf) ? 1
                                                                        : 0);
  LLValue *retllval =
      (irFty.arg_sret ? args[sretArgIndex] : call.getInstruction());
  bool retValIsLVal =
      (tf->isref && returnTy != Tvoid) || (irFty.arg_sret != nullptr);

  if (!retValIsLVal) {
    // let the ABI transform the return value back
    if (DtoIsInMemoryOnly(returntype)) {
      retllval = irFty.getRetLVal(returntype, retllval);
      retValIsLVal = true;
    } else {
      retllval = irFty.getRetRVal(returntype, retllval);
    }
  }

  // repaint the type if necessary
  Type *rbase = stripModifiers(resulttype->toBasetype(), true);
  Type *nextbase = stripModifiers(returntype->toBasetype(), true);
  if (!rbase->equals(nextbase)) {
    IF_LOG Logger::println("repainting return value from '%s' to '%s'",
                           returntype->toChars(), rbase->toChars());
    switch (rbase->ty) {
    case Tarray:
      if (tf->isref) {
        retllval = DtoBitCast(retllval, DtoType(rbase->pointerTo()));
      } else {
        retllval = DtoAggrPaint(retllval, DtoType(rbase));
      }
      break;

    case Tsarray:
      if (nextbase->ty == Tvector && !tf->isref) {
        if (retValIsLVal) {
          retllval = DtoBitCast(retllval, DtoType(rbase->pointerTo()));
        } else {
          // static arrays need to be dumped to memory; use vector alignment
          retllval =
              DtoAllocaDump(retllval, DtoType(rbase), DtoAlignment(nextbase),
                            ".vector_to_sarray_tmp");
          retValIsLVal = true;
        }
        break;
      }
      goto unknownMismatch;

    case Tclass:
    case Taarray:
    case Tpointer:
      if (tf->isref) {
        retllval = DtoBitCast(retllval, DtoType(rbase->pointerTo()));
      } else {
        retllval = DtoBitCast(retllval, DtoType(rbase));
      }
      break;

    case Tstruct:
      if (nextbase->ty == Taarray && !tf->isref) {
        // In the D2 frontend, the associative array type and its
        // object.AssociativeArray representation are used
        // interchangably in some places. However, AAs are returned
        // by value and not in an sret argument, so if the struct
        // type will be used, give the return value storage here
        // so that we get the right amount of indirections.
        LLValue *val =
            DtoInsertValue(llvm::UndefValue::get(DtoType(rbase)), retllval, 0);
        retllval = DtoAllocaDump(val, rbase, ".aalvaluetmp");
        retValIsLVal = true;
        break;
      }
      goto unknownMismatch;

    default:
    unknownMismatch:
      // Unfortunately, DMD has quirks resp. bugs with regard to name
      // mangling: For voldemort-type functions which return a nested
      // struct, the mangled name of the return type changes during
      // semantic analysis.
      //
      // (When the function deco is first computed as part of
      // determining the return type deco, its return type part is
      // left off to avoid cycles. If mangle/toDecoBuffer is then
      // called again for the type, it will pick up the previous
      // result and return the full deco string for the nested struct
      // type, consisting of both the full mangled function name, and
      // the struct identifier.)
      //
      // Thus, the type merging in stripModifiers does not work
      // reliably, and the equality check above can fail even if the
      // types only differ in a qualifier.
      //
      // Because a proper fix for this in the frontend is hard, we
      // just carry on and hope that the frontend didn't mess up,
      // i.e. that the LLVM types really match up.
      //
      // An example situation where this case occurs is:
      // ---
      // auto iota() {
      //     static struct Result {
      //         this(int) {}
      //         inout(Result) test() inout { return cast(inout)Result(0); }
      //     }
      //     return Result.init;
      // }
      // void main() { auto r = iota(); }
      // ---
      Logger::println("Unknown return mismatch type, ignoring.");
      break;
    }
    IF_LOG Logger::cout() << "final return value: " << *retllval << '\n';
  }

  // set calling convention and parameter attributes
  LLAttributeList &attrlist = attrs;
  if (dfnval && dfnval->func) {
    LLFunction *llfunc = llvm::dyn_cast<LLFunction>(DtoRVal(dfnval));
    if (llfunc && llfunc->isIntrinsic()) // override intrinsic attrs
    {
      attrlist = llvm::Intrinsic::getAttributes(
          gIR->context(),
          static_cast<llvm::Intrinsic::ID>(llfunc->getIntrinsicID()));
    } else {
      call.setCallingConv(callconv);
    }
  } else {
    call.setCallingConv(callconv);
  }
  // merge in function attributes set in callOrInvoke
  attrlist = attrlist.addAttributes(
      gIR->context(), LLAttributeList::FunctionIndex,
      llvm::AttrBuilder(call.getAttributes(), LLAttributeList::FunctionIndex));
  call.setAttributes(attrlist);

  // Special case for struct constructor calls: For temporaries, using the
  // this pointer value returned from the constructor instead of the alloca
  // passed as a parameter (which has the same value anyway) might lead to
  // instruction dominance issues because of the way it interacts with the
  // cleanups (see struct ctor hack in ToElemVisitor::visit(CallExp *)).
  if (dfnval && dfnval->func && dfnval->func->isCtorDeclaration() &&
      dfnval->func->isMember2()->isStructDeclaration()) {
    return new DLValue(resulttype, dfnval->vthis);
  }

  if (retValIsLVal) {
    return new DLValue(resulttype, retllval);
  }

  if (rbase->ty == Tarray) {
    return new DSliceValue(resulttype, retllval);
  }

  return new DImValue(resulttype, retllval);
}
