//===-- irclass.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/hdrgen.h" // for parametersTypeToChars()
#include "dmd/identifier.h"
#include "dmd/mangle.h"
#include "dmd/mtype.h"
#include "dmd/target.h"
#include "gen/abi/abi.h"
#include "gen/arrays.h"
#include "gen/funcgenstate.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/llvmhelpers.h"
#include "gen/mangling.h"
#include "gen/passes/metadata.h"
#include "gen/rttibuilder.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "gen/typinf.h"
#include "ir/iraggr.h"
#include "ir/irdsymbol.h"
#include "ir/irfunction.h"
#include "ir/irtypeclass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"

#ifndef NDEBUG
#include "llvm/Support/raw_ostream.h"
#endif

//////////////////////////////////////////////////////////////////////////////

IrClass::IrClass(ClassDeclaration *cd) : IrAggr(cd) {
  addInterfaceVtbls(cd);

  assert(interfacesWithVtbls.size() ==
             getIrType(type)->isClass()->getNumInterfaceVtbls() &&
         "inconsistent number of interface vtables in this class");
}

void IrClass::addInterfaceVtbls(ClassDeclaration *cd) {
  if (cd->baseClass && !cd->isInterfaceDeclaration()) {
    addInterfaceVtbls(cd->baseClass);
  }

  if (cd->vtblInterfaces) {
    for (auto bc : *cd->vtblInterfaces) {
      interfacesWithVtbls.push_back(bc);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////

LLGlobalVariable *IrClass::getVtblSymbol(bool define) {
  if (!vtbl) {
    const auto irMangle = getIRMangledVTableSymbolName(aggrdecl);

    LLType *vtblTy = getIrType(type)->isClass()->getVtblType();

    vtbl = declareGlobal(aggrdecl->loc, gIR->module, vtblTy, irMangle,
                         /*isConstant=*/true, false, useDLLImport());

    if (!define)
      define = defineOnDeclare(aggrdecl, /*isFunction=*/false);
  }

  if (define) {
    auto init = getVtblInit(); // might define vtbl
    if (!vtbl->hasInitializer())
      defineGlobal(vtbl, init, aggrdecl);
  }

  return vtbl;
}

//////////////////////////////////////////////////////////////////////////////

LLGlobalVariable *IrClass::getClassInfoSymbol(bool define) {
  if (!typeInfo) {
    const auto irMangle = getIRMangledClassInfoSymbolName(aggrdecl);

    // The type is also ClassInfo for interfaces – the actual TypeInfo for them
    // is a TypeInfo_Interface instance that references __ClassZ in its "base"
    // member.
    IrTypeClass *tc = getIrType(getClassInfoType(), true)->isClass();
    assert(tc && "invalid ClassInfo type");

    // We need to keep the symbol mutable as the type is not declared as
    // immutable on the D side, and e.g. synchronized() can be used on the
    // implicit monitor.
    typeInfo = declareGlobal(aggrdecl->loc, gIR->module, tc->getMemoryLLType(),
                             irMangle, /*isConstant=*/false, false, useDLLImport());

    // Generate some metadata on this ClassInfo if it's for a class.
    if (!aggrdecl->isInterfaceDeclaration()) {
      // regular TypeInfo metadata
      emitTypeInfoMetadata(typeInfo, aggrdecl->type);

      // Gather information
      LLType *bodyType = getLLStructType();
      bool hasDestructor = (aggrdecl->dtor != nullptr);
      // Construct the fields
      llvm::Metadata *mdVals[CD_NumFields];
      mdVals[CD_BodyType] =
          llvm::ConstantAsMetadata::get(llvm::UndefValue::get(bodyType));
      mdVals[CD_Finalize] = llvm::ConstantAsMetadata::get(
          LLConstantInt::get(LLType::getInt1Ty(gIR->context()), hasDestructor));
      // Construct the metadata and insert it into the module.
      const auto metaname = getMetadataName(CD_PREFIX, typeInfo);
      llvm::NamedMDNode *node = gIR->module.getOrInsertNamedMetadata(metaname);
      node->addOperand(llvm::MDNode::get(gIR->context(), mdVals));
    }

    if (!define)
      define = defineOnDeclare(aggrdecl, /*isFunction=*/false);
  }

  if (define) {
    auto init = getClassInfoInit();
    if (!typeInfo->hasInitializer())
      defineGlobal(typeInfo, init, aggrdecl);
  }

  return typeInfo;
}

//////////////////////////////////////////////////////////////////////////////

static Type *getInterfacesArrayType() {
  getClassInfoType(); // check declaration in object.d
  return Type::typeinfoclass->fields[3]->type;
}

LLGlobalVariable *IrClass::getInterfaceArraySymbol() {
  if (classInterfacesArray) {
    return classInterfacesArray;
  }

  ClassDeclaration *cd = aggrdecl->isClassDeclaration();

  size_t n = getIrType(type)->isClass()->getNumInterfaceVtbls();
  assert(n > 0 && "getting ClassInfo.interfaces storage symbol, but we "
                  "don't implement any interfaces");

  LLType *InterfaceTy = DtoType(getInterfacesArrayType()->nextOf());

  // create Interface[N]
  const auto irMangle = getIRMangledInterfaceInfosSymbolName(cd);

  LLArrayType *array_type = llvm::ArrayType::get(InterfaceTy, n);

  classInterfacesArray =
      declareGlobal(cd->loc, gIR->module, array_type, irMangle,
                    /*isConstant=*/true, false, false);

  return classInterfacesArray;
}

//////////////////////////////////////////////////////////////////////////////

LLConstant *IrClass::getVtblInit() {
  if (constVtbl) {
    return constVtbl;
  }

  IF_LOG Logger::println("Building vtbl initializer");
  LOG_SCOPE;

  ClassDeclaration *cd = aggrdecl->isClassDeclaration();
  assert(cd && "not class");

  std::vector<llvm::Constant *> constants;
  constants.reserve(cd->vtbl.length);

  const auto voidPtrType = getVoidPtrType();

  // start with the classinfo
  llvm::Constant *c;
  if (!cd->isCPPclass()) {
    if (!suppressTypeInfo()) {
      c = getClassInfoSymbol();
      c = DtoBitCast(c, voidPtrType);
    } else {
      // use null if there are no TypeInfos
      c = llvm::Constant::getNullValue(voidPtrType);
    }
    constants.push_back(c);
  }

  // add virtual function pointers
  size_t n = cd->vtbl.length;
  for (size_t i = cd->vtblOffset(); i < n; i++) {
    Dsymbol *dsym = cd->vtbl[i];
    assert(dsym && "null vtbl member");

    FuncDeclaration *fd = dsym->isFuncDeclaration();
    assert(fd && "vtbl entry not a function");

    if (cd->isAbstract() || (fd->isAbstract() && !fd->fbody)) {
      c = getNullValue(voidPtrType);
    } else {
      // If inferring return type and semantic3 has not been run, do it now.
      // This pops up in some other places in the frontend as well, however
      // it is probably a bug that it still occurs that late.
      if (fd->inferRetType() && !fd->type->nextOf()) {
        Logger::println("Running late functionSemantic to infer return type.");
        if (!fd->functionSemantic()) {
          if (fd->hasSemantic3Errors()) {
            Logger::println(
                "functionSemantic failed; using null for vtbl entry.");
            constants.push_back(getNullValue(voidPtrType));
            continue;
          }
          error(fd->loc,
                "failed to infer return type of `%s` for vtbl initializer",
                fd->toPrettyChars());
          fatal();
        }
      }

      c = DtoBitCast(DtoCallee(fd), voidPtrType);

      if (cd->isFuncHidden(fd) && !fd->isFuture()) {
        // fd is hidden from the view of this class. If fd overlaps with any
        // function in the vtbl[], issue error.
        for (size_t j = cd->vtblOffset(); j < n; j++) {
          if (j == i) {
            continue;
          }
          auto fd2 = cd->vtbl[j]->isFuncDeclaration();
          if (!fd2->ident->equals(fd->ident)) {
            continue;
          }
          if (fd2->isFuture()) {
            continue;
          }
          if (fd->leastAsSpecialized(fd2, nullptr) != MATCH::nomatch ||
              fd2->leastAsSpecialized(fd, nullptr) != MATCH::nomatch) {
            TypeFunction *tf = static_cast<TypeFunction *>(fd->type);
            if (tf->ty == TY::Tfunction) {
              error(cd->loc,
                    "%s `%s` use of `%s%s` is hidden by `%s`; use `alias %s = "
                    "%s.%s;` to introduce base class overload set",
                    cd->kind(), cd->toPrettyChars(), fd->toPrettyChars(),
                    parametersTypeToChars(tf->parameterList), cd->toChars(),
                    fd->toChars(), fd->parent->toChars(), fd->toChars());
            } else {
              error(cd->loc, "%s `%s` use of `%s` is hidden by `%s`",
                    cd->kind(), cd->toPrettyChars(), fd->toPrettyChars(),
                    cd->toChars());
            }
            fatal();
            break;
          }
        }
      }
    }

    constants.push_back(c);
  }

  // build the constant array
  LLArrayType *vtblTy = LLArrayType::get(voidPtrType, constants.size());
  constVtbl = LLConstantArray::get(vtblTy, constants);

  return constVtbl;
}

//////////////////////////////////////////////////////////////////////////////

/**
 * The following code should be kept in sync with upstream, dmd/toobj.d:
 * - genClassInfoForClass()
 * - genClassInfoForInterface()
 */

namespace {
unsigned buildClassinfoFlags(ClassDeclaration *cd) {
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
}

LLConstant *IrClass::getClassInfoInit() {
  if (constTypeInfo) {
    return constTypeInfo;
  }

  auto cd = aggrdecl->isClassDeclaration();
  IF_LOG Logger::println("Defining ClassInfo for: %s", cd->toChars());
  LOG_SCOPE;

  Type *const cinfoType = getClassInfoType(); // check declaration in object.d
  ClassDeclaration *const cinfo = Type::typeinfoclass;

  if (cinfo->fields.length != 12) {
    error(Loc(), "Unexpected number of fields in `object.ClassInfo`; "
                 "druntime version does not match compiler (see -v)");
    fatal();
  }

  const bool isInterface = cd->isInterfaceDeclaration();

  RTTIBuilder b(cinfoType);

  LLType *voidPtr = getVoidPtrType();
  LLType *voidPtrPtr = getPtrToType(voidPtr);

  // adapted from original dmd code
  // byte[] m_init
  if (isInterface) {
    b.push_null_void_array();
  } else {
    b.push_void_array(cd->size(Loc()), getInitSymbol());
  }

  // string name
  const char *name = cd->ident->toChars();
  if (strncmp(name, "TypeInfo_", 9) != 0) {
    name = cd->toPrettyChars(/*QualifyTypes=*/true);
  }
  b.push_string(name);

  // void*[] vtbl
  if (isInterface) {
    b.push_array(0, getNullValue(voidPtrPtr));
  } else {
    b.push_array(cd->vtbl.length, DtoBitCast(getVtblSymbol(), voidPtrPtr));
  }

  // Interface[] interfaces
  b.push(getClassInfoInterfaces());

  // TypeInfo_Class base
  assert(!isInterface || !cd->baseClass);
  if (cd->baseClass) {
    b.push_typeinfo(cd->baseClass->type);
  } else {
    b.push_null(cinfoType);
  }

  // void* destructor
  assert(!isInterface || !cd->tidtor);
  b.push_funcptr(cd->tidtor, Type::tvoidptr);

  // void function(Object) classInvariant
  assert(!isInterface || !cd->inv);
  VarDeclaration *invVar = cinfo->fields[6];
  b.push_funcptr(cd->inv, invVar->type);

  // ClassFlags m_flags
  const auto flags = buildClassinfoFlags(cd);
  b.push_uint(flags);

  // void* deallocator
  b.push_null_vp();

  // OffsetTypeInfo[] m_offTi
  VarDeclaration *offTiVar = cinfo->fields[9];
  b.push_null(offTiVar->type);

  // void function(Object) defaultConstructor
  VarDeclaration *defConstructorVar = cinfo->fields[10];
  CtorDeclaration *defConstructor = cd->defaultCtor;
  if (defConstructor && (defConstructor->storage_class & STCdisable)) {
    defConstructor = nullptr;
  }
  assert(!isInterface || !defConstructor);
  b.push_funcptr(defConstructor, defConstructorVar->type);

  // immutable(void)* m_RTInfo
  if (cd->getRTInfo) {
    b.push(toConstElem(cd->getRTInfo, gIR));
  } else if (isInterface || (flags & ClassFlags::noPointers)) {
    b.push_size_as_vp(0); // no pointers
  } else {
    b.push_size_as_vp(1); // has pointers
  }

  // build the initializer
  LLType *initType = getClassInfoSymbol()->getValueType();
  constTypeInfo = b.get_constant(isaStruct(initType));

  return constTypeInfo;
}

//////////////////////////////////////////////////////////////////////////////

llvm::GlobalVariable *IrClass::getInterfaceVtblSymbol(BaseClass *b,
                                                      size_t interfaces_index,
                                                      bool define) {
  LLGlobalVariable *gvar = nullptr;

  const auto it = interfaceVtblMap.find({b->sym, interfaces_index});
  if (it != interfaceVtblMap.end()) {
    gvar = it->second;
  } else {
    llvm::Type *vtblType =
        LLArrayType::get(getVoidPtrType(), b->sym->vtbl.length);

    // Thunk prefix
    char thunkPrefix[16];
    int thunkLen = snprintf(thunkPrefix, 16, "Thn%d_", b->offset);
    char thunkPrefixLen[16];
    snprintf(thunkPrefixLen, 16, "%d", thunkLen);

    OutBuffer mangledName;
    mangledName.writestring("_D");
    mangleToBuffer(aggrdecl, mangledName);
    mangledName.writestring("11__interface");
    mangleToBuffer(b->sym, mangledName);
    mangledName.writestring(thunkPrefixLen);
    mangledName.writestring(thunkPrefix);
    mangledName.writestring("6__vtblZ");

    const auto irMangle = getIRMangledVarName(mangledName.peekChars(), LINK::d);

    gvar = declareGlobal(aggrdecl->loc, gIR->module, vtblType, irMangle,
                         /*isConstant=*/true, false, false);

    // insert into the vtbl map
    interfaceVtblMap.insert({{b->sym, interfaces_index}, gvar});

    if (!define)
      define = defineOnDeclare(aggrdecl, /*isFunction=*/false);
  }

  if (define && !gvar->hasInitializer()) {
    auto init = getInterfaceVtblInit(b, interfaces_index);
    defineGlobal(gvar, init, aggrdecl);
  }

  return gvar;
}

LLConstant *IrClass::getInterfaceVtblInit(BaseClass *b,
                                          size_t interfaces_index) {
  IF_LOG Logger::println(
      "Defining vtbl for implementation of interface %s in class %s",
      b->sym->toPrettyChars(), aggrdecl->toPrettyChars());
  LOG_SCOPE;

  ClassDeclaration *cd = aggrdecl->isClassDeclaration();
  assert(cd && "not a class aggregate");

  FuncDeclarations vtbl_array;
  const bool new_instance = b->sym == cd;
  b->fillVtbl(cd, &vtbl_array, new_instance);

  std::vector<llvm::Constant *> constants;
  constants.reserve(vtbl_array.length);

  char thunkPrefix[16];
  snprintf(thunkPrefix, 16, "Thn%d_", b->offset);

  const auto voidPtrTy = getVoidPtrType();

  if (!b->sym->isCPPinterface()) { // skip interface info for CPP interfaces
    if (!suppressTypeInfo()) {
      // index into the interfaces array
      llvm::Constant *idxs[2] = {DtoConstSize_t(0),
                                 DtoConstSize_t(interfaces_index)};

      llvm::GlobalVariable *interfaceInfosZ = getInterfaceArraySymbol();
      llvm::Constant *c = llvm::ConstantExpr::getGetElementPtr(
            interfaceInfosZ->getValueType(), interfaceInfosZ, idxs, true);

      constants.push_back(DtoBitCast(c, voidPtrTy));
    } else {
      // use null if there are no TypeInfos
      constants.push_back(llvm::Constant::getNullValue(voidPtrTy));
    }
  }

  // add virtual function pointers
  size_t n = vtbl_array.length;
  for (size_t i = b->sym->vtblOffset(); i < n; i++) {
    FuncDeclaration *fd = vtbl_array[i];
    if (!fd) {
      // FIXME
      // why is this null?
      // happens for mini/s.d
      constants.push_back(getNullValue(voidPtrTy));
      continue;
    }

    assert((!fd->isAbstract() || fd->fbody) &&
           "null symbol in interface implementation vtable");

    DtoResolveFunction(fd);
    assert(isIrFuncCreated(fd) && "invalid vtbl function");

    IrFunction *irFunc = getIrFunc(fd);

    assert(irFunc->irFty.arg_this);

    int thunkOffset = b->offset;
    if (fd->interfaceVirtual)
      thunkOffset -= fd->interfaceVirtual->offset;
    if (thunkOffset == 0) {
      constants.push_back(DtoBitCast(irFunc->getLLVMCallee(), voidPtrTy));
      continue;
    }

    // Create the thunk function if it does not already exist in this
    // module.
    OutBuffer nameBuf;
    const auto mangledTargetName = mangleExact(fd);
    nameBuf.write(mangledTargetName, 2);
    nameBuf.writestring(thunkPrefix);
    nameBuf.writestring(mangledTargetName + 2);

    const auto thunkIRMangle =
        getIRMangledFuncName(nameBuf.peekChars(), fd->resolvedLinkage());

    llvm::Function *thunk = gIR->module.getFunction(thunkIRMangle);
    if (!thunk) {
      const LinkageWithCOMDAT lwc(LLGlobalValue::LinkOnceODRLinkage,
                                  needsCOMDAT());
      const auto callee = irFunc->getLLVMCallee();
      thunk = LLFunction::Create(
          callee->getFunctionType(), lwc.first,
          thunkIRMangle, &gIR->module);
      setLinkage(lwc, thunk);
      thunk->copyAttributesFrom(callee);

      // Thunks themselves don't have an identity, only the target function has.
      thunk->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

      // thunks don't need exception handling themselves
      thunk->setPersonalityFn(nullptr);

      // It is necessary to add debug information to the thunk in case it is
      // subject to inlining. See https://llvm.org/bugs/show_bug.cgi?id=26833
      IF_LOG Logger::println("Doing function body for thunk to: %s",
                             fd->toChars());

      // Create a dummy FuncDeclaration with enough information to satisfy the
      // DIBuilder
      FuncDeclaration *thunkFd = reinterpret_cast<FuncDeclaration *>(
          memcpy(new char[sizeof(FuncDeclaration)], (void *)fd,
                 sizeof(FuncDeclaration)));
      thunkFd->ir = new IrDsymbol();
      auto thunkFunc = getIrFunc(thunkFd, true); // create the IrFunction
      thunkFunc->setLLVMFunc(thunk);
      thunkFunc->type = irFunc->type;
      gIR->funcGenStates.emplace_back(new FuncGenState(*thunkFunc, *gIR));

      // debug info
      thunkFunc->diSubprogram = nullptr;
      thunkFunc->diSubprogram = gIR->DBuilder.EmitThunk(thunk, thunkFd);

      // create entry and end blocks
      llvm::BasicBlock *beginbb =
          llvm::BasicBlock::Create(gIR->context(), "", thunk);

      // set up the IRBuilder scope for the thunk
      const auto savedIRBuilderScope = gIR->setInsertPoint(beginbb);
      gIR->DBuilder.EmitFuncStart(thunkFd);

      // Copy the function parameters, so later we can pass them to the
      // real function and set their names from the original function (the
      // latter being just for IR readablilty).
      std::vector<LLValue *> args;
      llvm::Function::arg_iterator thunkArg = thunk->arg_begin();
      llvm::Function::arg_iterator origArg = callee->arg_begin();
      for (; thunkArg != thunk->arg_end(); ++thunkArg, ++origArg) {
        thunkArg->setName(origArg->getName());
        args.push_back(&(*thunkArg));
      }

      // cast 'this' to Object
      const int thisArgIndex =
          (!irFunc->irFty.arg_sret || gABI->passThisBeforeSret(irFunc->type))
              ? 0
              : 1;
      LLValue *&thisArg = args[thisArgIndex];
      LLType *targetThisType = thisArg->getType();
      thisArg = DtoBitCast(thisArg, getVoidPtrType());
      thisArg = DtoGEP1(LLType::getInt8Ty(gIR->context()), thisArg, DtoConstInt(-thunkOffset));
      thisArg = DtoBitCast(thisArg, targetThisType);

      // all calls that might be subject to inlining into a caller with debug
      // info should have debug info, too
      gIR->DBuilder.EmitStopPoint(fd->loc);

      // call the real vtbl function.
      llvm::CallInst *call = gIR->ir->CreateCall(callee, args);
      call->setCallingConv(callee->getCallingConv());
      call->setAttributes(callee->getAttributes());
      call->setTailCallKind(callee->isVarArg() ? llvm::CallInst::TCK_MustTail
                                               : llvm::CallInst::TCK_Tail);

      // return from the thunk
      if (thunk->getReturnType() == LLType::getVoidTy(gIR->context())) {
        llvm::ReturnInst::Create(gIR->context(), beginbb);
      } else {
        llvm::ReturnInst::Create(gIR->context(), call, beginbb);
      }

      gIR->funcGenStates.pop_back();
    }

    constants.push_back(DtoBitCast(thunk, voidPtrTy));
  }

  // build the vtbl constant
  llvm::Constant *vtbl_constant = LLConstantArray::get(
      LLArrayType::get(voidPtrTy, constants.size()), constants);

  return vtbl_constant;
}

void IrClass::defineInterfaceVtbls() {
  const size_t n = interfacesWithVtbls.size();
  assert(n == getIrType(type)->isClass()->getNumInterfaceVtbls() &&
         "inconsistent number of interface vtables in this class");

  for (size_t i = 0; i < n; ++i) {
    auto baseClass = interfacesWithVtbls[i];
    getInterfaceVtblSymbol(baseClass, i, /*define=*/true);
  }
}

//////////////////////////////////////////////////////////////////////////////

LLConstant *IrClass::getClassInfoInterfaces() {
  IF_LOG Logger::println("Building ClassInfo.interfaces");
  LOG_SCOPE;

  ClassDeclaration *cd = aggrdecl->isClassDeclaration();
  assert(cd);

  size_t n = interfacesWithVtbls.size();
  assert(getIrType(type)->isClass()->getNumInterfaceVtbls() == n &&
         "inconsistent number of interface vtables in this class");

  Type *interfacesArrayType = getInterfacesArrayType();

  if (n == 0) {
    return getNullValue(DtoType(interfacesArrayType));
  }

  // Build array of:
  //
  // struct Interface
  // {
  //     ClassInfo   classinfo;
  //     void*[]     vtbl;
  //     ptrdiff_t   offset;
  // }

  LLSmallVector<LLConstant *, 6> constants;
  constants.reserve(cd->vtblInterfaces->length);

  LLType *classinfo_type = DtoType(getClassInfoType());
  LLType *voidptrptr_type = DtoType(Type::tvoid->pointerTo()->pointerTo());
  LLStructType *interface_type =
      isaStruct(DtoType(interfacesArrayType->nextOf()));
  assert(interface_type);

  for (size_t i = 0; i < n; ++i) {
    BaseClass *it = interfacesWithVtbls[i];

    IF_LOG Logger::println("Adding interface %s", it->sym->toPrettyChars());

    IrClass *irinter = getIrAggr(it->sym);
    assert(irinter && "interface has null IrStruct");
    IrTypeClass *itc = getIrType(irinter->type)->isClass();
    assert(itc && "null interface IrTypeClass");

    // classinfo
    LLConstant *ci = irinter->getClassInfoSymbol();
    ci = DtoBitCast(ci, classinfo_type);

    // vtbl
    LLConstant *vtb;
    // interface get a null
    if (cd->isInterfaceDeclaration()) {
      vtb = DtoConstSlice(DtoConstSize_t(0), getNullValue(voidptrptr_type));
    } else {
      vtb = getInterfaceVtblSymbol(it, i);
      vtb = DtoBitCast(vtb, voidptrptr_type);
      auto vtblSize = itc->getVtblType()->getNumContainedTypes();
      vtb = DtoConstSlice(DtoConstSize_t(vtblSize), vtb);
    }

    // offset
    LLConstant *off = DtoConstSize_t(it->offset);

    // create Interface struct
    LLConstant *inits[3] = {ci, vtb, off};
    LLConstant *entry = LLConstantStruct::get(interface_type, inits);
    constants.push_back(entry);
  }

  // create Interface[N]
  LLArrayType *array_type = llvm::ArrayType::get(interface_type, n);

  // create and apply initializer
  LLConstant *arr = LLConstantArray::get(array_type, constants);
  auto ciarr = getInterfaceArraySymbol();
  defineGlobal(ciarr, arr, cd);

  // return null, only baseclass provide interfaces
  if (cd->vtblInterfaces->length == 0) {
    return getNullValue(DtoType(interfacesArrayType));
  }

  // only the interface explicitly implemented by this class
  // (not super classes) should show in ClassInfo
  LLConstant *idxs[2] = {DtoConstSize_t(0),
                         DtoConstSize_t(n - cd->vtblInterfaces->length)};

  LLConstant *ptr = llvm::ConstantExpr::getGetElementPtr(ciarr->getValueType(),
                                                         ciarr, idxs, true);

  // return as a slice
  return DtoConstSlice(DtoConstSize_t(cd->vtblInterfaces->length), ptr);
}
