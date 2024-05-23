//===-- classes.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/classes.h"

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/identifier.h"
#include "dmd/init.h"
#include "dmd/mtype.h"
#include "dmd/target.h"
#include "gen/arrays.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/nested.h"
#include "gen/optimizer.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "ir/iraggr.h"
#include "ir/irdsymbol.h"
#include "ir/irfunction.h"
#include "ir/irtypeclass.h"

////////////////////////////////////////////////////////////////////////////////

// FIXME: this needs to be cleaned up

void DtoResolveClass(ClassDeclaration *cd) {
  if (cd->ir->isResolved()) {
    return;
  }
  cd->ir->setResolved();

  IF_LOG Logger::println("DtoResolveClass(%s): %s", cd->toPrettyChars(),
                         cd->loc.toChars());
  LOG_SCOPE;

  // make sure the base classes are processed first
  for (auto bc : *cd->baseclasses) {
    DtoResolveClass(bc->sym);
  }

  // make sure type exists
  DtoType(cd->type);

  // create IrAggr
  getIrAggr(cd, true);

  // make sure all fields really get their ir field
  for (auto vd : cd->fields) {
    IF_LOG {
      if (isIrFieldCreated(vd)) {
        Logger::println("class field already exists");
      }
    }
    getIrField(vd, true);
  }
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoNewClass(const Loc &loc, TypeClass *tc, NewExp *newexp) {
  // resolve type
  DtoResolveClass(tc->sym);
  const auto irClass = getIrAggr(tc->sym);

  // allocate
  LLValue *mem;
  bool doInit = true;
  if (newexp->onstack) {
    mem = DtoRawAlloca(irClass->getLLStructType(), tc->sym->alignsize,
                       ".newclass_alloca");
  } else {
    const bool useEHAlloc = global.params.ehnogc && newexp->thrownew;
    llvm::Function *fn = getRuntimeFunction(
        loc, gIR->module, useEHAlloc ? "_d_newThrowable" : "_d_allocclass");
    LLConstant *ci = irClass->getClassInfoSymbol();
    mem = gIR->CreateCallOrInvoke(
        fn, ci, useEHAlloc ? ".newthrowable" : ".newclass_gc");
    doInit = !useEHAlloc;
  }

  // init
  if (doInit)
    DtoInitClass(tc, mem);

  // init inner-class outer reference
  if (newexp->thisexp) {
    Logger::println("Resolving outer class");
    LOG_SCOPE;
    unsigned idx = getFieldGEPIndex(tc->sym, tc->sym->vthis);
    LLValue *src = DtoRVal(newexp->thisexp);
    LLValue *dst = DtoGEP(irClass->getLLStructType(), mem, 0, idx);
    IF_LOG Logger::cout() << "dst: " << *dst << "\nsrc: " << *src << '\n';
    DtoStore(src, dst);
  }
  // set the context for nested classes
  else if (tc->sym->isNested() && tc->sym->vthis) {
    DtoResolveNestedContext(loc, tc->sym, mem);
  }

  // call constructor
  if (newexp->member) {
    // evaluate argprefix
    if (newexp->argprefix) {
      toElemDtor(newexp->argprefix);
    }

    Logger::println("Calling constructor");
    assert(newexp->arguments != NULL);
    DFuncValue dfn(newexp->member, DtoCallee(newexp->member), mem);
    // ignore ctor return value (C++ ctors on Posix may not return `this`)
    DtoCallFunction(newexp->loc, tc, &dfn, newexp->arguments);
    return new DImValue(tc, mem);
  }

  assert(newexp->argprefix == NULL);

  // return default constructed class
  return new DImValue(tc, mem);
}

////////////////////////////////////////////////////////////////////////////////

void DtoInitClass(TypeClass *tc, LLValue *dst) {
  DtoResolveClass(tc->sym);

  IrClass *irClass = getIrAggr(tc->sym);
  llvm::StructType *st = irClass->getLLStructType();

  // Set vtable field. Doing this seperately might be optimized better.
  LLValue *tmp = DtoGEP(st, dst, 0u, 0, "vtbl");
  LLValue *val = irClass->getVtblSymbol();
  DtoStore(val, tmp);

  // For D classes, set the monitor field to null.
  const bool isCPPclass = tc->sym->isCPPclass() ? true : false;
  if (!isCPPclass) {
    tmp = DtoGEP(st, dst, 0, 1, "monitor");
    val = LLConstant::getNullValue(st->getElementType(1));
    DtoStore(val, tmp);
  }

  // Copy the rest from the static initializer, if any.
  unsigned const firstDataIdx = isCPPclass ? 1 : 2;
  uint64_t const dataBytes =
      tc->sym->structsize - target.ptrsize * firstDataIdx;
  if (dataBytes == 0) {
    return;
  }

  LLValue *dstarr = DtoGEP(st, dst, 0, firstDataIdx);

  // init symbols might not have valid types
  LLValue *initsym = irClass->getInitSymbol();
  LLValue *srcarr = DtoGEP(st, initsym, 0, firstDataIdx);

  DtoMemCpy(dstarr, srcarr, DtoConstSize_t(dataBytes));
}

////////////////////////////////////////////////////////////////////////////////

void DtoFinalizeClass(const Loc &loc, LLValue *inst) {
  // get runtime function
  llvm::Function *fn =
      getRuntimeFunction(loc, gIR->module, "_d_callfinalizer");

  gIR->CreateCallOrInvoke(fn, inst, "");
}

////////////////////////////////////////////////////////////////////////////////

void DtoFinalizeScopeClass(const Loc &loc, DValue *dval,
                           bool dynTypeMatchesStaticType) {
  llvm::Value *inst = DtoRVal(dval);

  if (!isOptimizationEnabled() || !dynTypeMatchesStaticType) {
    DtoFinalizeClass(loc, inst);
    return;
  }

  bool hasDtor = false;
  const auto cd = dval->type->toBasetype()->isTypeClass()->sym;
  for (auto cd2 = cd; cd2; cd2 = cd2->baseClass) {
    if (cd2->dtor) {
      hasDtor = true;
      break;
    }
  }

  if (hasDtor) {
    DtoFinalizeClass(loc, inst);
    return;
  }

  // no dtors => only finalize (via druntime call) if monitor is set,
  // see https://github.com/ldc-developers/ldc/issues/2515
  llvm::BasicBlock *ifbb = gIR->insertBB("if");
  llvm::BasicBlock *endbb = gIR->insertBBAfter(ifbb, "endif");

  llvm::StructType *st =
      isaStruct(getIrType(cd->type, true)->isClass()->getMemoryLLType());
  const auto monitor =
      DtoLoad(st->getElementType(1), DtoGEP(st, inst, 0, 1), ".monitor");
  const auto hasMonitor =
      gIR->ir->CreateICmp(llvm::CmpInst::ICMP_NE, monitor,
                          getNullValue(monitor->getType()), ".hasMonitor");
  llvm::BranchInst::Create(ifbb, endbb, hasMonitor, gIR->scopebb());

  gIR->ir->SetInsertPoint(ifbb);
  DtoFinalizeClass(loc, inst);
  gIR->ir->CreateBr(endbb);

  gIR->ir->SetInsertPoint(endbb);
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoCastClass(const Loc &loc, DValue *val, Type *_to) {
  IF_LOG Logger::println("DtoCastClass(%s, %s)", val->type->toChars(),
                         _to->toChars());
  LOG_SCOPE;

  Type *to = _to->toBasetype();

  // class -> pointer
  if (to->ty == TY::Tpointer) {
    IF_LOG Logger::println("to pointer");
    return new DImValue(_to, DtoRVal(val));
  }
  // class -> bool
  if (to->ty == TY::Tbool) {
    IF_LOG Logger::println("to bool");
    LLValue *llval = DtoRVal(val);
    LLValue *zero = LLConstant::getNullValue(llval->getType());
    return new DImValue(_to, gIR->ir->CreateICmpNE(llval, zero));
  }
  // class -> integer
  if (to->isintegral()) {
    IF_LOG Logger::println("to %s", to->toChars());

    // get class ptr
    LLValue *v = DtoRVal(val);
    // cast to size_t
    v = gIR->ir->CreatePtrToInt(v, DtoSize_t(), "");
    // cast to the final int type
    DImValue im(Type::tsize_t, v);
    return DtoCastInt(loc, &im, _to);
  }
  // class -> typeof(null)
  if (to->ty == TY::Tnull) {
    IF_LOG Logger::println("to %s", to->toChars());
    return new DImValue(_to, LLConstant::getNullValue(DtoType(_to)));
  }

  // must be class/interface
  assert(to->ty == TY::Tclass);
  TypeClass *tc = static_cast<TypeClass *>(to);

  // from type
  Type *from = val->type->toBasetype();
  TypeClass *fc = static_cast<TypeClass *>(from);

  // copy DMD logic:
  // if to isBaseOf from with offset:   (to ? to + offset : null)
  // else if from is C++ and to is C++:  to
  // else if from is C++ and to is D:    null
  // else if from is interface:          _d_interface_cast(to)
  // else if from is class:              _d_dynamic_cast(to)

  int offset = 0;
  if (tc->sym->isBaseOf(fc->sym, &offset)) {
    Logger::println("static down cast");
    // interface types don't cover the full object in case of multiple inheritence
    //  so GEP on the original type is inappropriate

    // offset pointer
    LLValue *orig = DtoRVal(val);
    LLValue *v = orig;
    if (offset != 0) {
      assert(offset > 0);
      v = DtoGEP1(getI8Type(), v, DtoConstUint(offset));
    }
    IF_LOG {
      Logger::cout() << "V = " << *v << std::endl;
    }

    // Check whether the original value was null, and return null if so.
    // Sure we could have jumped over the code above in this case, but
    // it's just a GEP and (maybe) a pointer-to-pointer BitCast, so it
    // should be pretty cheap and perfectly safe even if the original was
    // null.
    LLValue *isNull = gIR->ir->CreateICmpEQ(
        orig, LLConstant::getNullValue(orig->getType()), ".nullcheck");
    v = gIR->ir->CreateSelect(
        isNull, LLConstant::getNullValue(getVoidPtrType()), v, ".interface");
    // return r-value
    return new DImValue(_to, v);
  }

  if (fc->sym->classKind == ClassKind::cpp) {
    Logger::println("C++ class/interface cast");
    LLValue *v = tc->sym->classKind == ClassKind::cpp
                     ? DtoRVal(val)
                     : LLConstant::getNullValue(getVoidPtrType());
    return new DImValue(_to, v);
  }

  // from interface
  if (fc->sym->isInterfaceDeclaration()) {
    Logger::println("interface cast");
    return DtoDynamicCastInterface(loc, val, _to);
  }
  // from class
  Logger::println("dynamic up cast");
  return DtoDynamicCastObject(loc, val, _to);
}

////////////////////////////////////////////////////////////////////////////////

static void resolveObjectAndClassInfoClasses() {
  // check declarations in object.d
  getObjectType();
  getClassInfoType();

  DtoResolveClass(ClassDeclaration::object);
  DtoResolveClass(Type::typeinfoclass);
}

DValue *DtoDynamicCastObject(const Loc &loc, DValue *val, Type *_to) {
  // call:
  // Object _d_dynamic_cast(Object o, ClassInfo c)

  llvm::Function *func =
      getRuntimeFunction(loc, gIR->module, "_d_dynamic_cast");
  LLFunctionType *funcTy = func->getFunctionType();

  resolveObjectAndClassInfoClasses();

  // Object o
  LLValue *obj = DtoRVal(val);
  assert(funcTy->getParamType(0) == obj->getType());

  // ClassInfo c
  TypeClass *to = static_cast<TypeClass *>(_to->toBasetype());
  DtoResolveClass(to->sym);

  LLValue *cinfo = getIrAggr(to->sym)->getClassInfoSymbol();
  assert(funcTy->getParamType(1) == cinfo->getType());

  // call it
  LLValue *ret = gIR->CreateCallOrInvoke(func, obj, cinfo);

  return new DImValue(_to, ret);
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoDynamicCastInterface(const Loc &loc, DValue *val, Type *_to) {
  // call:
  // Object _d_interface_cast(void* p, ClassInfo c)

  llvm::Function *func =
      getRuntimeFunction(loc, gIR->module, "_d_interface_cast");

  resolveObjectAndClassInfoClasses();

  // void* p
  LLValue *ptr = DtoRVal(val);

  // ClassInfo c
  TypeClass *to = static_cast<TypeClass *>(_to->toBasetype());
  DtoResolveClass(to->sym);
  LLValue *cinfo = getIrAggr(to->sym)->getClassInfoSymbol();

  // call it
  LLValue *ret = gIR->CreateCallOrInvoke(func, ptr, cinfo);

  return new DImValue(_to, ret);
}

////////////////////////////////////////////////////////////////////////////////

std::pair<llvm::Value *, llvm::Value *>
DtoVirtualFunctionPointer(DValue *inst, FuncDeclaration *fdecl) {
  // sanity checks
  assert(fdecl->isVirtual());
  assert(!fdecl->isFinalFunc());
  TypeClass *tc = inst->type->toBasetype()->isTypeClass();
  assert(tc);
  // slot 0 is always ClassInfo/Interface* unless it is a CPP class
  assert(fdecl->vtblIndex > 0 ||
         (fdecl->vtblIndex == 0 &&
          inst->type->toBasetype()->isTypeClass()->sym->isCPPclass()));

  // get instance
  LLValue *vthis = DtoRVal(inst);
  IF_LOG Logger::cout() << "vthis: " << *vthis << '\n';

  const auto irtc = getIrType(tc->sym->type, true)->isClass();
  const auto vtblType = irtc->getVtblType();

  LLValue *vtable = vthis;
  // get the vtbl for objects
  vtable = DtoGEP(irtc->getMemoryLLType(), vthis, 0u, 0);
  // load vtbl ptr
  vtable = DtoLoad(vtblType->getPointerTo(), vtable);
  // index vtbl
  const std::string name = fdecl->toChars();
  const auto vtblname = name + "@vtbl";
  LLValue *funcval =
      DtoGEP(vtblType, vtable, 0, fdecl->vtblIndex, vtblname.c_str());
  // load opaque pointer.
  funcval = DtoAlignedLoad(vtblType->getElementType(), funcval);
  // Because vtables are immutable, LLVM's !invariant.load
  // can be applied (helps with devirtualization).
  llvm::cast<llvm::LoadInst>(funcval)->setMetadata(
      "invariant.load", llvm::MDNode::get(gIR->context(), {}));

  IF_LOG Logger::cout() << "funcval: " << *funcval << '\n';

  // postpone naming until after casting to get the name in call instructions
  funcval->setName(name);

  IF_LOG Logger::cout() << "funcval casted: " << *funcval << '\n';

  return std::make_pair(funcval, vtable);
}
