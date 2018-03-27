//===-- irclass.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/SmallString.h"
#ifndef NDEBUG
#include "llvm/Support/raw_ostream.h"
#endif

#include "aggregate.h"
#include "declaration.h"
#include "hdrgen.h" // for parametersTypeToChars()
#include "mtype.h"
#include "target.h"
#include "gen/funcgenstate.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/arrays.h"
#include "gen/metadata.h"
#include "gen/runtime.h"
#include "gen/functions.h"
#include "gen/abi.h"
#include "gen/mangling.h"

#include "ir/iraggr.h"
#include "ir/irfunction.h"
#include "ir/irtypeclass.h"

//////////////////////////////////////////////////////////////////////////////

extern LLConstant *DtoDefineClassInfo(ClassDeclaration *cd);

//////////////////////////////////////////////////////////////////////////////

LLGlobalVariable *IrAggr::getVtblSymbol() {
  if (vtbl) {
    return vtbl;
  }

  // create the vtblZ symbol
  const auto irMangle = getIRMangledVTableSymbolName(aggrdecl);

  LLType *vtblTy = stripModifiers(type)->ctype->isClass()->getVtblType();

  vtbl =
      getOrCreateGlobal(aggrdecl->loc, gIR->module, vtblTy, true,
                        llvm::GlobalValue::ExternalLinkage, nullptr, irMangle);

  return vtbl;
}

//////////////////////////////////////////////////////////////////////////////

LLGlobalVariable *IrAggr::getClassInfoSymbol() {
  if (classInfo) {
    return classInfo;
  }

  // create the ClassZ / InterfaceZ symbol
  const auto irMangle = getIRMangledClassInfoSymbolName(aggrdecl);

  // The type is also ClassInfo for interfaces – the actual TypeInfo for them
  // is a TypeInfo_Interface instance that references __ClassZ in its "base"
  // member.
  ClassDeclaration *cinfo = Type::typeinfoclass;
  DtoType(cinfo->type);
  IrTypeClass *tc = stripModifiers(cinfo->type)->ctype->isClass();
  assert(tc && "invalid ClassInfo type");

  // classinfos cannot be constants since they're used as locks for synchronized
  classInfo = getOrCreateGlobal(
      aggrdecl->loc, gIR->module, tc->getMemoryLLType(), false,
      llvm::GlobalValue::ExternalLinkage, nullptr, irMangle);

  // Generate some metadata on this ClassInfo if it's for a class.
  ClassDeclaration *classdecl = aggrdecl->isClassDeclaration();
  if (classdecl && !aggrdecl->isInterfaceDeclaration()) {
    // Gather information
    LLType *type = DtoType(aggrdecl->type);
    LLType *bodyType = llvm::cast<LLPointerType>(type)->getElementType();
    bool hasDestructor = (classdecl->dtor != nullptr);
    bool hasCustomDelete = (classdecl->aggDelete != nullptr);
    // Construct the fields
    llvm::Metadata *mdVals[CD_NumFields];
    mdVals[CD_BodyType] =
        llvm::ConstantAsMetadata::get(llvm::UndefValue::get(bodyType));
    mdVals[CD_Finalize] = llvm::ConstantAsMetadata::get(
        LLConstantInt::get(LLType::getInt1Ty(gIR->context()), hasDestructor));
    mdVals[CD_CustomDelete] = llvm::ConstantAsMetadata::get(
        LLConstantInt::get(LLType::getInt1Ty(gIR->context()), hasCustomDelete));
    // Construct the metadata and insert it into the module.
    OutBuffer debugName;
    debugName.writestring(CD_PREFIX);
    if (irMangle[0] == '\1') {
      debugName.write(irMangle.data() + 1, irMangle.length() - 1);
    } else {
      debugName.write(irMangle.data(), irMangle.length());
    }
    llvm::NamedMDNode *node =
        gIR->module.getOrInsertNamedMetadata(debugName.peekString());
    node->addOperand(llvm::MDNode::get(
        gIR->context(), llvm::makeArrayRef(mdVals, CD_NumFields)));
  }

  return classInfo;
}

//////////////////////////////////////////////////////////////////////////////

LLGlobalVariable *IrAggr::getInterfaceArraySymbol() {
  if (classInterfacesArray) {
    return classInterfacesArray;
  }

  ClassDeclaration *cd = aggrdecl->isClassDeclaration();

  size_t n = stripModifiers(type)->ctype->isClass()->getNumInterfaceVtbls();
  assert(n > 0 && "getting ClassInfo.interfaces storage symbol, but we "
                  "don't implement any interfaces");

  LLType *InterfaceTy = DtoType(Type::typeinfoclass->fields[3]->type->nextOf());

  // create Interface[N]
  const auto irMangle = getIRMangledInterfaceInfosSymbolName(cd);

  LLArrayType *array_type = llvm::ArrayType::get(InterfaceTy, n);

  // We keep the global as external for now and only consider template linkage
  // if we emit the initializer later.
  classInterfacesArray =
      getOrCreateGlobal(cd->loc, gIR->module, array_type, true,
                        llvm::GlobalValue::ExternalLinkage, nullptr, irMangle);

  return classInterfacesArray;
}

//////////////////////////////////////////////////////////////////////////////

LLConstant *IrAggr::getVtblInit() {
  if (constVtbl) {
    return constVtbl;
  }

  IF_LOG Logger::println("Building vtbl initializer");
  LOG_SCOPE;

  ClassDeclaration *cd = aggrdecl->isClassDeclaration();
  assert(cd && "not class");

  std::vector<llvm::Constant *> constants;
  constants.reserve(cd->vtbl.dim);

  const auto voidPtrType = getVoidPtrType();

  // start with the classinfo
  llvm::Constant *c;
  if (!cd->isCPPclass()) {
    c = getClassInfoSymbol();
    c = DtoBitCast(c, voidPtrType);
    constants.push_back(c);
  }

  // add virtual function pointers
  size_t n = cd->vtbl.dim;
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
      if (fd->inferRetType && !fd->type->nextOf()) {
        Logger::println("Running late functionSemantic to infer return type.");
        if (!fd->functionSemantic()) {
          fd->error("failed to infer return type for vtbl initializer");
          fatal();
        }
      }

      DtoResolveFunction(fd);
      assert(isIrFuncCreated(fd) && "invalid vtbl function");
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
          if (fd->leastAsSpecialized(fd2) || fd2->leastAsSpecialized(fd)) {
            TypeFunction *tf = static_cast<TypeFunction *>(fd->type);
            if (tf->ty == Tfunction) {
              cd->error("use of `%s%s` is hidden by `%s`; use `alias %s = "
                        "%s.%s;` to introduce base class overload set",
                        fd->toPrettyChars(),
                        parametersTypeToChars(tf->parameters, tf->varargs),
                        cd->toChars(), fd->toChars(), fd->parent->toChars(),
                        fd->toChars());
            } else {
              cd->error("use of `%s` is hidden by `%s`", fd->toPrettyChars(),
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

LLConstant *IrAggr::getClassInfoInit() {
  if (constClassInfo) {
    return constClassInfo;
  }
  constClassInfo = DtoDefineClassInfo(aggrdecl->isClassDeclaration());
  return constClassInfo;
}

//////////////////////////////////////////////////////////////////////////////

llvm::GlobalVariable *IrAggr::getInterfaceVtbl(BaseClass *b, bool new_instance,
                                               size_t interfaces_index) {
  auto it = interfaceVtblMap.find({b->sym, interfaces_index});
  if (it != interfaceVtblMap.end()) {
    return it->second;
  }

  IF_LOG Logger::println(
      "Building vtbl for implementation of interface %s in class %s",
      b->sym->toPrettyChars(), aggrdecl->toPrettyChars());
  LOG_SCOPE;

  ClassDeclaration *cd = aggrdecl->isClassDeclaration();
  assert(cd && "not a class aggregate");

  FuncDeclarations vtbl_array;
  b->fillVtbl(cd, &vtbl_array, new_instance);

  std::vector<llvm::Constant *> constants;
  constants.reserve(vtbl_array.dim);

  const auto voidPtrTy = getVoidPtrType();

  if (!b->sym->isCPPinterface()) { // skip interface info for CPP interfaces
    // index into the interfaces array
    llvm::Constant *idxs[2] = {DtoConstSize_t(0),
                               DtoConstSize_t(interfaces_index)};

    llvm::GlobalVariable *interfaceInfosZ = getInterfaceArraySymbol();
    llvm::Constant *c = llvm::ConstantExpr::getGetElementPtr(
        isaPointer(interfaceInfosZ)->getElementType(),
        interfaceInfosZ, idxs, true);

    constants.push_back(DtoBitCast(c, voidPtrTy));
  }

  // Thunk prefix
  char thunkPrefix[16];
  int thunkLen = sprintf(thunkPrefix, "Thn%d_", b->offset);
  char thunkPrefixLen[16];
  sprintf(thunkPrefixLen, "%d", thunkLen);

  // add virtual function pointers
  size_t n = vtbl_array.dim;
  for (size_t i = b->sym->vtblOffset(); i < n; i++) {
    Dsymbol *dsym = static_cast<Dsymbol *>(vtbl_array.data[i]);
    if (dsym == nullptr) {
      // FIXME
      // why is this null?
      // happens for mini/s.d
      constants.push_back(getNullValue(voidPtrTy));
      continue;
    }

    FuncDeclaration *fd = dsym->isFuncDeclaration();
    assert(fd && "vtbl entry not a function");

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
        getIRMangledFuncName(nameBuf.peekString(), fd->linkage);

    llvm::Function *thunk = gIR->module.getFunction(thunkIRMangle);
    if (!thunk) {
      const LinkageWithCOMDAT lwc(LLGlobalValue::LinkOnceODRLinkage,
                                  supportsCOMDAT());
      const auto callee = irFunc->getLLVMCallee();
      thunk = LLFunction::Create(
          isaFunction(callee->getType()->getContainedType(0)), lwc.first,
          thunkIRMangle, &gIR->module);
      setLinkage(lwc, thunk);
      thunk->copyAttributesFrom(callee);

// Thunks themselves don't have an identity, only the target
// function has.
#if LDC_LLVM_VER >= 309
      thunk->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
#else
      thunk->setUnnamedAddr(true);
#endif

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
      thunkFunc->diSubprogram = gIR->DBuilder.EmitThunk(thunk, thunkFd);

      // create entry and end blocks
      llvm::BasicBlock *beginbb =
          llvm::BasicBlock::Create(gIR->context(), "", thunk);
      gIR->scopes.push_back(IRScope(beginbb));

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
      thisArg = DtoGEP1(thisArg, DtoConstInt(-thunkOffset), true);
      thisArg = DtoBitCast(thisArg, targetThisType);

      // all calls that might be subject to inlining into a caller with debug
      // info should have debug info, too
      gIR->DBuilder.EmitStopPoint(fd->loc);

      // call the real vtbl function.
      llvm::CallInst *call = gIR->ir->CreateCall(callee, args);
      call->setCallingConv(irFunc->getCallingConv());
      call->setTailCallKind(thunk->isVarArg() ? llvm::CallInst::TCK_MustTail
                                              : llvm::CallInst::TCK_Tail);

      // return from the thunk
      if (thunk->getReturnType() == LLType::getVoidTy(gIR->context())) {
        llvm::ReturnInst::Create(gIR->context(), beginbb);
      } else {
        llvm::ReturnInst::Create(gIR->context(), call, beginbb);
      }

      gIR->DBuilder.EmitFuncEnd(thunkFd);

      // clean up
      gIR->scopes.pop_back();

      gIR->funcGenStates.pop_back();
    }

    constants.push_back(DtoBitCast(thunk, voidPtrTy));
  }

  // build the vtbl constant
  llvm::Constant *vtbl_constant = LLConstantArray::get(
      LLArrayType::get(voidPtrTy, constants.size()), constants);

  OutBuffer mangledName;
  mangledName.writestring("_D");
  mangleToBuffer(cd, &mangledName);
  mangledName.writestring("11__interface");
  mangleToBuffer(b->sym, &mangledName);
  mangledName.writestring(thunkPrefixLen);
  mangledName.writestring(thunkPrefix);
  mangledName.writestring("6__vtblZ");

  const auto irMangle = getIRMangledVarName(mangledName.peekString(), LINKd);

  const auto lwc = DtoLinkage(cd);
  LLGlobalVariable *GV =
      getOrCreateGlobal(cd->loc, gIR->module, vtbl_constant->getType(), true,
                        lwc.first, vtbl_constant, irMangle);
  setLinkage(lwc, GV);

  // insert into the vtbl map
  interfaceVtblMap.insert({{b->sym, interfaces_index}, GV});

  return GV;
}

bool IrAggr::isPacked() const {
  return static_cast<IrTypeAggr *>(type->ctype)->packed;
}

//////////////////////////////////////////////////////////////////////////////

LLConstant *IrAggr::getClassInfoInterfaces() {
  IF_LOG Logger::println("Building ClassInfo.interfaces");
  LOG_SCOPE;

  ClassDeclaration *cd = aggrdecl->isClassDeclaration();
  assert(cd);

  size_t n = interfacesWithVtbls.size();
  assert(stripModifiers(type)->ctype->isClass()->getNumInterfaceVtbls() == n &&
         "inconsistent number of interface vtables in this class");

  VarDeclaration *interfaces_idx = Type::typeinfoclass->fields[3];

  if (n == 0) {
    return getNullValue(DtoType(interfaces_idx->type));
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
  constants.reserve(cd->vtblInterfaces->dim);

  LLType *classinfo_type = DtoType(Type::typeinfoclass->type);
  LLType *voidptrptr_type = DtoType(Type::tvoid->pointerTo()->pointerTo());
  VarDeclaration *idx = Type::typeinfoclass->fields[3];
  LLStructType *interface_type = isaStruct(DtoType(idx->type->nextOf()));
  assert(interface_type);

  for (size_t i = 0; i < n; ++i) {
    BaseClass *it = interfacesWithVtbls[i];

    IF_LOG Logger::println("Adding interface %s", it->sym->toPrettyChars());

    IrAggr *irinter = getIrAggr(it->sym);
    assert(irinter && "interface has null IrStruct");
    IrTypeClass *itc = stripModifiers(irinter->type)->ctype->isClass();
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
      auto itv = interfaceVtblMap.find({it->sym, i});
      assert(itv != interfaceVtblMap.end() && "interface vtbl not found");
      vtb = itv->second;
      vtb = DtoBitCast(vtb, voidptrptr_type);
      auto vtblSize = itc->getVtblType()->getNumContainedTypes();
      vtb = DtoConstSlice(DtoConstSize_t(vtblSize), vtb);
    }

    // offset
    LLConstant *off = DtoConstSize_t(it->offset);

    // create Interface struct
    LLConstant *inits[3] = {ci, vtb, off};
    LLConstant *entry =
        LLConstantStruct::get(interface_type, llvm::makeArrayRef(inits, 3));
    constants.push_back(entry);
  }

  // create Interface[N]
  LLArrayType *array_type = llvm::ArrayType::get(interface_type, n);

  // create and apply initializer
  LLConstant *arr = LLConstantArray::get(array_type, constants);
  auto ciarr = getInterfaceArraySymbol();
  ciarr->setInitializer(arr);
  setLinkage(cd, ciarr);

  // return null, only baseclass provide interfaces
  if (cd->vtblInterfaces->dim == 0) {
    return getNullValue(DtoType(interfaces_idx->type));
  }

  // only the interface explicitly implemented by this class
  // (not super classes) should show in ClassInfo
  LLConstant *idxs[2] = {DtoConstSize_t(0),
                         DtoConstSize_t(n - cd->vtblInterfaces->dim)};

  LLConstant *ptr = llvm::ConstantExpr::getGetElementPtr(
      isaPointer(ciarr)->getElementType(), ciarr, idxs, true);

  // return as a slice
  return DtoConstSlice(DtoConstSize_t(cd->vtblInterfaces->dim), ptr);
}

//////////////////////////////////////////////////////////////////////////////

void IrAggr::initializeInterface() {
  InterfaceDeclaration *base = aggrdecl->isInterfaceDeclaration();
  assert(base && "not interface");

  // has interface vtbls?
  if (!base->vtblInterfaces) {
    return;
  }

  for (auto bc : *base->vtblInterfaces) {
    // add to the interface list
    interfacesWithVtbls.push_back(bc);
  }
}

//////////////////////////////////////////////////////////////////////////////
