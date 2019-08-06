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
#include "gen/rttibuilder.h"
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
  IrAggr *irAggr = getIrAggr(cd, true);

  // make sure all fields really get their ir field
  for (auto vd : cd->fields) {
    IF_LOG {
      if (isIrFieldCreated(vd)) {
        Logger::println("class field already exists");
      }
    }
    getIrField(vd, true);
  }

  // interface only emit typeinfo and classinfo
  if (cd->isInterfaceDeclaration()) {
    irAggr->initializeInterface();
  }
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoNewClass(Loc &loc, TypeClass *tc, NewExp *newexp) {
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
    DtoResolveFunction(newexp->allocator);
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
    mem = gIR->CreateCallOrInvoke(fn, ci,
                                  useEHAlloc ? ".newthrowable_alloc"
                                             : ".newclass_gc_alloc")
              .getInstruction();
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
    LLValue *dst = DtoGEPi(mem, 0, idx);
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
    DtoResolveFunction(newexp->member);
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

  // Set vtable field. Doing this seperately might be optimized better.
  LLValue *tmp = DtoGEPi(dst, 0, 0, "vtbl");
  LLValue *val = DtoBitCast(getIrAggr(tc->sym)->getVtblSymbol(),
                            tmp->getType()->getContainedType(0));
  DtoStore(val, tmp);

  // For D classes, set the monitor field to null.
  const bool isCPPclass = tc->sym->isCPPclass() ? true : false;
  if (!isCPPclass) {
    tmp = DtoGEPi(dst, 0, 1, "monitor");
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

  LLValue *dstarr = DtoGEPi(dst, 0, firstDataIdx);

  // init symbols might not have valid types
  LLValue *initsym = getIrAggr(tc->sym)->getInitSymbol();
  initsym = DtoBitCast(initsym, DtoType(tc));
  LLValue *srcarr = DtoGEPi(initsym, 0, firstDataIdx);

  DtoMemCpy(dstarr, srcarr, DtoConstSize_t(dataBytes));
}

////////////////////////////////////////////////////////////////////////////////

void DtoFinalizeClass(Loc &loc, LLValue *inst) {
  // get runtime function
  llvm::Function *fn =
      getRuntimeFunction(loc, gIR->module, "_d_callfinalizer");

  gIR->CreateCallOrInvoke(
      fn, DtoBitCast(inst, fn->getFunctionType()->getParamType(0)), "");
}

////////////////////////////////////////////////////////////////////////////////

void DtoFinalizeScopeClass(Loc &loc, LLValue *inst, bool hasDtor) {
  if (!isOptimizationEnabled() || hasDtor) {
    DtoFinalizeClass(loc, inst);
    return;
  }

  // no dtors => only finalize (via druntime call) if monitor is set,
  // see https://github.com/ldc-developers/ldc/issues/2515
  llvm::BasicBlock *ifbb = gIR->insertBB("if");
  llvm::BasicBlock *endbb = gIR->insertBBAfter(ifbb, "endif");

  const auto monitor = DtoLoad(DtoGEPi(inst, 0, 1), ".monitor");
  const auto hasMonitor =
      gIR->ir->CreateICmp(llvm::CmpInst::ICMP_NE, monitor,
                          getNullValue(monitor->getType()), ".hasMonitor");
  llvm::BranchInst::Create(ifbb, endbb, hasMonitor, gIR->scopebb());

  gIR->scope() = IRScope(ifbb);
  DtoFinalizeClass(loc, inst);
  gIR->ir->CreateBr(endbb);

  gIR->scope() = IRScope(endbb);
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoCastClass(Loc &loc, DValue *val, Type *_to) {
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

DValue *DtoDynamicCastObject(Loc &loc, DValue *val, Type *_to) {
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
  LLValue *ret = gIR->CreateCallOrInvoke(func, obj, cinfo).getInstruction();

  // cast return value
  ret = DtoBitCast(ret, DtoType(_to));

  return new DImValue(_to, ret);
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoDynamicCastInterface(Loc &loc, DValue *val, Type *_to) {
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
  LLValue *ret = gIR->CreateCallOrInvoke(func, ptr, cinfo).getInstruction();

  // cast return value
  ret = DtoBitCast(ret, DtoType(_to));

  return new DImValue(_to, ret);
}

////////////////////////////////////////////////////////////////////////////////

LLValue *DtoVirtualFunctionPointer(DValue *inst, FuncDeclaration *fdecl,
                                   const char *name) {
  // sanity checks
  assert(fdecl->isVirtual());
  assert(!fdecl->isFinalFunc());
  assert(inst->type->toBasetype()->ty == Tclass);
  // 0 is always ClassInfo/Interface* unless it is a CPP interface
  assert(fdecl->vtblIndex > 0 ||
         (fdecl->vtblIndex == 0 && fdecl->linkage == LINKcpp));

  // get instance
  LLValue *vthis = DtoRVal(inst);
  IF_LOG Logger::cout() << "vthis: " << *vthis << '\n';

  LLValue *funcval = vthis;
  // get the vtbl for objects
  funcval = DtoGEPi(funcval, 0, 0);
  // load vtbl ptr
  funcval = DtoLoad(funcval);
  // index vtbl
  std::string vtblname = name;
  vtblname.append("@vtbl");
  funcval = DtoGEPi(funcval, 0, fdecl->vtblIndex, vtblname.c_str());
  // load opaque pointer
  funcval = DtoAlignedLoad(funcval);

  IF_LOG Logger::cout() << "funcval: " << *funcval << '\n';

  // cast to funcptr type
  funcval = DtoBitCast(funcval, getPtrToType(DtoFunctionType(fdecl)));

  // postpone naming until after casting to get the name in call instructions
  funcval->setName(name);

  IF_LOG Logger::cout() << "funcval casted: " << *funcval << '\n';

  return funcval;
}

////////////////////////////////////////////////////////////////////////////////

#if GENERATE_OFFTI

// build a single element for the OffsetInfo[] of ClassInfo
static LLConstant *build_offti_entry(ClassDeclaration *cd, VarDeclaration *vd) {
  std::vector<LLConstant *> inits(2);

  // size_t offset;
  //
  assert(vd->ir.irField);
  // grab the offset from llvm and the formal class type
  size_t offset =
      gDataLayout->getStructLayout(isaStruct(cd->type->ir.type->get()))
          ->getElementOffset(vd->ir.irField->index);
  // offset nested struct/union fields
  offset += vd->ir.irField->unionOffset;

  // assert that it matches DMD
  Logger::println("offsets: %lu vs %u", offset, vd->offset);
  assert(offset == vd->offset);

  inits[0] = DtoConstSize_t(offset);

  // TypeInfo ti;
  inits[1] = DtoTypeInfoOf(vd->type);

  // done
  return llvm::ConstantStruct::get(inits);
}

static LLConstant *build_offti_array(ClassDeclaration *cd, LLType *arrayT) {
  IrAggr *iraggr = cd->ir->irAggr;

  size_t nvars = iraggr->varDecls.size();
  std::vector<LLConstant *> arrayInits(nvars);

  for (size_t i = 0; i < nvars; i++) {
    arrayInits[i] = build_offti_entry(cd, iraggr->varDecls[i]);
  }

  LLConstant *size = DtoConstSize_t(nvars);
  LLConstant *ptr;

  if (nvars == 0)
    return LLConstant::getNullValue(arrayT);

  // array type
  LLArrayType *arrTy = llvm::ArrayType::get(arrayInits[0]->getType(), nvars);
  LLConstant *arrInit = LLConstantArray::get(arrTy, arrayInits);

  // create symbol
  llvm::GlobalVariable *gvar =
      getOrCreateGlobal(cd->loc, gIR->module, arrTy, true,
                        llvm::GlobalValue::InternalLinkage, arrInit, ".offti");
  ptr = DtoBitCast(gvar, getPtrToType(arrTy->getElementType()));

  return DtoConstSlice(size, ptr);
}

#endif // GENERATE_OFFTI

static LLConstant *build_class_dtor(ClassDeclaration *cd) {
  FuncDeclaration *dtor = cd->tidtor;

  // if no destructor emit a null
  if (!dtor) {
    return getNullPtr(getVoidPtrType());
  }

  DtoResolveFunction(dtor);
  return llvm::ConstantExpr::getBitCast(
      DtoCallee(dtor), getPtrToType(LLType::getInt8Ty(gIR->context())));
}

static unsigned build_classinfo_flags(ClassDeclaration *cd) {
  // adapted from original dmd code:
  // toobj.c: ToObjFile::visit(ClassDeclaration*) and
  // ToObjFile::visit(InterfaceDeclaration*)

  auto flags = ClassFlags::hasOffTi | ClassFlags::hasTypeInfo;
  if (cd->isInterfaceDeclaration()) {
    if (cd->isCOMinterface()) {
      flags |= ClassFlags::isCOMclass;
    }
    return flags;
  }

  if (cd->isCOMclass()) {
    flags |= ClassFlags::isCOMclass;
  }
  if (cd->isCPPclass()) {
    flags |= ClassFlags::isCPPclass;
  }
  flags |= ClassFlags::hasGetMembers;
  if (cd->ctor) {
    flags |= ClassFlags::hasCtor;
  }
  for (ClassDeclaration *pc = cd; pc; pc = pc->baseClass) {
    if (pc->dtor) {
      flags |= ClassFlags::hasDtor;
      break;
    }
  }
  if (cd->isAbstract()) {
    flags |= ClassFlags::isAbstract;
  }
  for (ClassDeclaration *pc = cd; pc; pc = pc->baseClass) {
    if (pc->members) {
      for (Dsymbol *sm : *pc->members) {
        // printf("sm = %s %s\n", sm->kind(), sm->toChars());
        if (sm->hasPointers()) {
          return flags;
        }
      }
    }
  }
  flags |= ClassFlags::noPointers;

  return flags;
}

LLConstant *DtoDefineClassInfo(ClassDeclaration *cd) {
  //     The layout is:
  //       {
  //         void **vptr;
  //         monitor_t monitor;
  //         byte[] init;
  //         string name;
  //         void*[] vtbl;
  //         Interface[] interfaces;
  //         TypeInfo_Class base;
  //         void *destructor;
  //         void function(Object) classInvariant;
  //         ClassFlags m_flags;
  //         void* deallocator;
  //         OffsetTypeInfo[] m_offTi;
  //         void function(Object) defaultConstructor;
  //         immutable(void)* m_RTInfo;
  //       }

  IF_LOG Logger::println("DtoDefineClassInfo(%s)", cd->toChars());
  LOG_SCOPE;

  assert(cd->type->ty == Tclass);

  IrAggr *ir = getIrAggr(cd);
  Type *const cinfoType = getClassInfoType(); // check declaration in object.d
  ClassDeclaration *const cinfo = Type::typeinfoclass;

  if (cinfo->fields.dim != 12) {
    error(Loc(), "Unexpected number of fields in `object.ClassInfo`; "
                 "druntime version does not match compiler (see -v)");
    fatal();
  }

  // use the rtti builder
  RTTIBuilder b(cinfoType);

  LLConstant *c;

  LLType *voidPtr = getVoidPtrType();
  LLType *voidPtrPtr = getPtrToType(voidPtr);

  // adapted from original dmd code
  // init[]
  if (cd->isInterfaceDeclaration()) {
    b.push_null_void_array();
  } else {
    size_t initsz = cd->size(Loc());
    b.push_void_array(initsz, ir->getInitSymbol());
  }

  // name[]
  const char *name = cd->ident->toChars();
  size_t namelen = strlen(name);
  if (!(namelen > 9 && memcmp(name, "TypeInfo_", 9) == 0)) {
    name = cd->toPrettyChars();
    namelen = strlen(name);
  }
  b.push_string(name);

  // vtbl[]
  if (cd->isInterfaceDeclaration()) {
    b.push_array(0, getNullValue(voidPtrPtr));
  } else {
    c = DtoBitCast(ir->getVtblSymbol(), voidPtrPtr);
    b.push_array(cd->vtbl.dim, c);
  }

  // interfaces[]
  b.push(ir->getClassInfoInterfaces());

  // base
  // interfaces never get a base, just the interfaces[]
  if (cd->baseClass && !cd->isInterfaceDeclaration()) {
    b.push_classinfo(cd->baseClass);
  } else {
    b.push_null(cinfoType);
  }

  // destructor
  if (cd->isInterfaceDeclaration()) {
    b.push_null_vp();
  } else {
    b.push(build_class_dtor(cd));
  }

  // invariant
  VarDeclaration *invVar = cinfo->fields[6];
  b.push_funcptr(cd->inv, invVar->type);

  // flags
  const unsigned flags = build_classinfo_flags(cd);
  b.push_uint(flags);

  // deallocator
  b.push_funcptr(cd->aggDelete, Type::tvoid->pointerTo());

  // offset typeinfo
  VarDeclaration *offTiVar = cinfo->fields[9];
#if GENERATE_OFFTI
  if (cd->isInterfaceDeclaration())
    b.push_null(offTiVar->type);
  else
    b.push(build_offti_array(cd, DtoType(offTiVar->type)));
#else
  b.push_null(offTiVar->type);
#endif

  // defaultConstructor
  VarDeclaration *defConstructorVar = cinfo->fields.data[10];
  CtorDeclaration *defConstructor = cd->defaultCtor;
  if (defConstructor && (defConstructor->storage_class & STCdisable)) {
    defConstructor = nullptr;
  }
  b.push_funcptr(defConstructor, defConstructorVar->type);

  // m_RTInfo
  // The cases where getRTInfo is null are not quite here, but the code is
  // modelled after what DMD does.
  if (cd->getRTInfo) {
    b.push(toConstElem(cd->getRTInfo, gIR));
  } else if (flags & ClassFlags::noPointers) {
    b.push_size_as_vp(0); // no pointers
  } else {
    b.push_size_as_vp(1); // has pointers
  }

  /*size_t n = inits.size();
  for (size_t i=0; i<n; ++i)
  {
      Logger::cout() << "inits[" << i << "]: " << *inits[i] << '\n';
  }*/

  // build the initializer
  LLType *initType = ir->classInfo->getType()->getContainedType(0);
  LLConstant *finalinit = b.get_constant(isaStruct(initType));

  // Logger::cout() << "built the classinfo initializer:\n" << *finalinit
  // <<'\n';
  ir->constClassInfo = finalinit;

  // sanity check
  assert(finalinit->getType() == initType &&
         "__ClassZ initializer does not match the ClassInfo type");

  // return initializer
  return finalinit;
}
