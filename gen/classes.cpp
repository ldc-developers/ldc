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

  // allocate
  LLValue *mem;
  bool doInit = true;
  if (newexp->onstack) {
    unsigned alignment = tc->sym->alignsize;
    if (alignment == STRUCTALIGN_DEFAULT)
      alignment = 0;
    mem = DtoRawAlloca(DtoType(tc)->getContainedType(0), alignment,
                       ".newclass_alloca");
  }
  // custom allocator
  else if (newexp->allocator) {
    DFuncValue dfn(newexp->allocator, DtoCallee(newexp->allocator));
    DValue *res = DtoCallFunction(newexp->loc, nullptr, &dfn, newexp->newargs);
    mem = DtoBitCast(DtoRVal(res), DtoType(tc), ".newclass_custom");
  }
  // default allocator
  else {
    const bool useEHAlloc = global.params.ehnogc && newexp->thrownew;
    llvm::Function *fn = getRuntimeFunction(
        loc, gIR->module, useEHAlloc ? "_d_newThrowable" : "_d_allocclass");
    LLConstant *ci = DtoBitCast(getIrAggr(tc->sym)->getClassInfoSymbol(),
                                DtoType(getClassInfoType()));
    mem = gIR->CreateCallOrInvoke(
        fn, ci, useEHAlloc ? ".newthrowable_alloc" : ".newclass_gc_alloc");
    mem = DtoBitCast(mem, DtoType(tc),
                     useEHAlloc ? ".newthrowable" : ".newclass_gc");
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
    LLValue *dst = DtoGEP(mem, 0, idx);
    IF_LOG Logger::cout() << "dst: " << *dst << "\nsrc: " << *src << '\n';
    DtoStore(src, DtoBitCast(dst, getPtrToType(src->getType())));
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

  // Set vtable field. Doing this seperately might be optimized better.
  LLValue *tmp = DtoGEP(dst, 0u, 0, "vtbl");
  LLValue *val =
      DtoBitCast(irClass->getVtblSymbol(), tmp->getType()->getContainedType(0));
  DtoStore(val, tmp);

  // For D classes, set the monitor field to null.
  const bool isCPPclass = tc->sym->isCPPclass() ? true : false;
  if (!isCPPclass) {
    tmp = DtoGEP(dst, 0, 1, "monitor");
    val = LLConstant::getNullValue(tmp->getType()->getContainedType(0));
    DtoStore(val, tmp);
  }

  // Copy the rest from the static initializer, if any.
  unsigned const firstDataIdx = isCPPclass ? 1 : 2;
  uint64_t const dataBytes =
      tc->sym->structsize - target.ptrsize * firstDataIdx;
  if (dataBytes == 0) {
    return;
  }

  LLValue *dstarr = DtoGEP(dst, 0, firstDataIdx);

  // init symbols might not have valid types
  LLValue *initsym = irClass->getInitSymbol();
  initsym = DtoBitCast(initsym, DtoType(tc));
  LLValue *srcarr = DtoGEP(initsym, 0, firstDataIdx);

  unsigned align = target.ptrsize;
  DtoMemCpy(dstarr, srcarr, align, align, dataBytes);
}

////////////////////////////////////////////////////////////////////////////////

void DtoFinalizeClass(const Loc &loc, LLValue *inst) {
  // get runtime function
  llvm::Function *fn =
      getRuntimeFunction(loc, gIR->module, "_d_callfinalizer");

  gIR->CreateCallOrInvoke(
      fn, DtoBitCast(inst, fn->getFunctionType()->getParamType(0)), "");
}

////////////////////////////////////////////////////////////////////////////////

void DtoFinalizeScopeClass(const Loc &loc, LLValue *inst, bool hasDtor) {
  if (!isOptimizationEnabled() || hasDtor) {
    DtoFinalizeClass(loc, inst);
    return;
  }

  // no dtors => only finalize (via druntime call) if monitor is set,
  // see https://github.com/ldc-developers/ldc/issues/2515
  llvm::BasicBlock *ifbb = gIR->insertBB("if");
  llvm::BasicBlock *endbb = gIR->insertBBAfter(ifbb, "endif");

  const auto monitor = DtoLoad(DtoGEP(inst, 0, 1), ".monitor");
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
  if (to->ty == Tpointer) {
    IF_LOG Logger::println("to pointer");
    LLType *tolltype = DtoType(_to);
    LLValue *rval = DtoBitCast(DtoRVal(val), tolltype);
    return new DImValue(_to, rval);
  }
  // class -> bool
  if (to->ty == Tbool) {
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
  if (to->ty == Tnull) {
    IF_LOG Logger::println("to %s", to->toChars());
    return new DImValue(_to, LLConstant::getNullValue(DtoType(_to)));
  }

  // must be class/interface
  assert(to->ty == Tclass);
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

  LLType *toType = DtoType(_to);
  int offset = 0;
  if (tc->sym->isBaseOf(fc->sym, &offset)) {
    Logger::println("static down cast");
    // interface types don't cover the full object in case of multiple inheritence
    //  so GEP on the original type is inappropriate

    // offset pointer
    LLValue *orig = DtoRVal(val);
    LLValue *v = orig;
    if (offset != 0) {
      v = DtoBitCast(v, getVoidPtrType());
      LLValue *off =
          LLConstantInt::get(LLType::getInt32Ty(gIR->context()), offset);
      v = gIR->ir->CreateGEP(v, off);
    }
    IF_LOG {
      Logger::cout() << "V = " << *v << std::endl;
      Logger::cout() << "T = " << *toType << std::endl;
    }
    v = DtoBitCast(v, toType);

    // Check whether the original value was null, and return null if so.
    // Sure we could have jumped over the code above in this case, but
    // it's just a GEP and (maybe) a pointer-to-pointer BitCast, so it
    // should be pretty cheap and perfectly safe even if the original was
    // null.
    LLValue *isNull = gIR->ir->CreateICmpEQ(
        orig, LLConstant::getNullValue(orig->getType()), ".nullcheck");
    v = gIR->ir->CreateSelect(isNull, LLConstant::getNullValue(toType), v,
                              ".interface");
    // return r-value
    return new DImValue(_to, v);
  }

  if (fc->sym->classKind == ClassKind::cpp) {
    Logger::println("C++ class/interface cast");
    LLValue *v = tc->sym->classKind == ClassKind::cpp
                     ? DtoBitCast(DtoRVal(val), toType)
                     : LLConstant::getNullValue(toType);
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
  obj = DtoBitCast(obj, funcTy->getParamType(0));
  assert(funcTy->getParamType(0) == obj->getType());

  // ClassInfo c
  TypeClass *to = static_cast<TypeClass *>(_to->toBasetype());
  DtoResolveClass(to->sym);

  LLValue *cinfo = getIrAggr(to->sym)->getClassInfoSymbol();
  // unfortunately this is needed as the implementation of object differs
  // somehow from the declaration
  // this could happen in user code as well :/
  cinfo = DtoBitCast(cinfo, funcTy->getParamType(1));
  assert(funcTy->getParamType(1) == cinfo->getType());

  // call it
  LLValue *ret = gIR->CreateCallOrInvoke(func, obj, cinfo);

  // cast return value
  ret = DtoBitCast(ret, DtoType(_to));

  return new DImValue(_to, ret);
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoDynamicCastInterface(const Loc &loc, DValue *val, Type *_to) {
  // call:
  // Object _d_interface_cast(void* p, ClassInfo c)

  llvm::Function *func =
      getRuntimeFunction(loc, gIR->module, "_d_interface_cast");
  LLFunctionType *funcTy = func->getFunctionType();

  resolveObjectAndClassInfoClasses();

  // void* p
  LLValue *ptr = DtoRVal(val);
  ptr = DtoBitCast(ptr, funcTy->getParamType(0));

  // ClassInfo c
  TypeClass *to = static_cast<TypeClass *>(_to->toBasetype());
  DtoResolveClass(to->sym);
  LLValue *cinfo = getIrAggr(to->sym)->getClassInfoSymbol();
  // unfortunately this is needed as the implementation of object differs
  // somehow from the declaration
  // this could happen in user code as well :/
  cinfo = DtoBitCast(cinfo, funcTy->getParamType(1));

  // call it
  LLValue *ret = gIR->CreateCallOrInvoke(func, ptr, cinfo);

  // cast return value
  ret = DtoBitCast(ret, DtoType(_to));

  return new DImValue(_to, ret);
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DtoVirtualFunctionPointer(DValue *inst, FuncDeclaration *fdecl) {
  // sanity checks
  assert(fdecl->isVirtual());
  assert(!fdecl->isFinalFunc());
  assert(inst->type->toBasetype()->ty == Tclass);
  // slot 0 is always ClassInfo/Interface* unless it is a CPP class
  assert(fdecl->vtblIndex > 0 ||
         (fdecl->vtblIndex == 0 &&
          inst->type->toBasetype()->isTypeClass()->sym->isCPPclass()));

  // get instance
  LLValue *vthis = DtoRVal(inst);
  IF_LOG Logger::cout() << "vthis: " << *vthis << '\n';

  LLValue *funcval = vthis;
  // get the vtbl for objects
  funcval = DtoGEP(funcval, 0u, 0);
  // load vtbl ptr
  funcval = DtoLoad(funcval);
  // index vtbl
  const std::string name = fdecl->toChars();
  const auto vtblname = name + "@vtbl";
  funcval = DtoGEP(funcval, 0, fdecl->vtblIndex, vtblname.c_str());
  // load opaque pointer
  funcval = DtoLoad(funcval);

  IF_LOG Logger::cout() << "funcval: " << *funcval << '\n';

  // cast to funcptr type
  funcval = DtoBitCast(funcval, getPtrToType(DtoFunctionType(fdecl)));

  // postpone naming until after casting to get the name in call instructions
  funcval->setName(name);

  IF_LOG Logger::cout() << "funcval casted: " << *funcval << '\n';

  return funcval;
}
