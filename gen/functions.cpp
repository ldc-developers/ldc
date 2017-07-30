//===-- functions.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/functions.h"

#include "aggregate.h"
#include "declaration.h"
#include "id.h"
#include "init.h"
#include "module.h"
#include "mtype.h"
#include "statement.h"
#include "template.h"
#include "gen/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/dvalue.h"
#include "gen/inlineir.h"
#include "gen/irstate.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/nested.h"
#include "gen/optimizer.h"
#include "gen/pragma.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "gen/uda.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/CFG.h"
#include <iostream>

llvm::FunctionType *DtoFunctionType(Type *type, IrFuncTy &irFty, Type *thistype,
                                    Type *nesttype, bool isMain, bool isCtor,
                                    bool isIntrinsic) {
  IF_LOG Logger::println("DtoFunctionType(%s)", type->toChars());
  LOG_SCOPE

  // sanity check
  assert(type->ty == Tfunction);
  TypeFunction *f = static_cast<TypeFunction *>(type);
  assert(f->next && "Encountered function type with invalid return type; "
                    "trying to codegen function ignored by the frontend?");

  // Return cached type if available
  if (irFty.funcType) {
    return irFty.funcType;
  }

  TargetABI *abi = (isIntrinsic ? TargetABI::getIntrinsic() : gABI);

  // Do not modify irFty yet; this function may be called recursively if any
  // of the argument types refer to this type.
  IrFuncTy newIrFty;

  // The index of the next argument on the LLVM level.
  unsigned nextLLArgIdx = 0;

  if (isMain) {
    // _Dmain always returns i32, no matter what the type in the D main() is.
    newIrFty.ret = new IrFuncTyArg(Type::tint32, false);
  } else {
    Type *rt = f->next;
    const bool byref = f->isref && rt->toBasetype()->ty != Tvoid;
    AttrBuilder attrs;

    if (abi->returnInArg(f)) {
      // sret return
      newIrFty.arg_sret = new IrFuncTyArg(
          rt, true,
          AttrBuilder().add(LLAttribute::StructRet).add(LLAttribute::NoAlias));
      const unsigned alignment = DtoAlignment(rt);
      if (alignment &&
          // FIXME: LLVM inliner issues for std.bitmanip and std.uni on Win64
          !global.params.targetTriple.isOSMSVCRT()) {
        newIrFty.arg_sret->attrs.addAlignment(alignment);
      }
      rt = Type::tvoid;
      ++nextLLArgIdx;
    } else {
      // sext/zext return
      attrs.add(DtoShouldExtend(byref ? rt->pointerTo() : rt));
    }
    newIrFty.ret = new IrFuncTyArg(rt, byref, attrs);
  }
  ++nextLLArgIdx;

  if (thistype) {
    // Add the this pointer for member functions
    AttrBuilder attrs;
    attrs.add(LLAttribute::NonNull);
    if (isCtor) {
      attrs.add(LLAttribute::Returned);
    }
    newIrFty.arg_this =
        new IrFuncTyArg(thistype, thistype->toBasetype()->ty == Tstruct, attrs);
    ++nextLLArgIdx;
  } else if (nesttype) {
    // Add the context pointer for nested functions
    AttrBuilder attrs;
    attrs.add(LLAttribute::NonNull);
    newIrFty.arg_nest = new IrFuncTyArg(nesttype, false, attrs);
    ++nextLLArgIdx;
  }

  // vararg functions are special too
  if (f->varargs) {
    if (f->linkage == LINKd) {
      // d style with hidden args
      // 2 (array) is handled by the frontend
      if (f->varargs == 1) {
        // _arguments
        newIrFty.arg_arguments =
            new IrFuncTyArg(Type::dtypeinfo->type->arrayOf(), false);
        ++nextLLArgIdx;
      }
    }

    newIrFty.c_vararg = true;
  }

  // if this _Dmain() doesn't have an argument, we force it to have one
  const size_t numExplicitDArgs = Parameter::dim(f->parameters);

  if (isMain && numExplicitDArgs == 0) {
    Type *mainargs = Type::tchar->arrayOf()->arrayOf();
    newIrFty.args.push_back(new IrFuncTyArg(mainargs, false));
    ++nextLLArgIdx;
  }

  for (size_t i = 0; i < numExplicitDArgs; ++i) {
    Parameter *arg = Parameter::getNth(f->parameters, i);

    // Whether the parameter is passed by LLVM value or as a pointer to the
    // alloca/….
    bool passPointer = arg->storageClass & (STCref | STCout);

    Type *loweredDType = arg->type;
    AttrBuilder attrs;
    if (arg->storageClass & STClazy) {
      // Lazy arguments are lowered to delegates.
      Logger::println("lazy param");
      auto ltf = new TypeFunction(nullptr, arg->type, 0, LINKd);
      auto ltd = new TypeDelegate(ltf);
      loweredDType = ltd;
    } else if (passPointer) {
      // ref/out
      attrs.addDereferenceable(loweredDType->size());
    } else {
      if (abi->passByVal(loweredDType)) {
        // LLVM ByVal parameters are pointers to a copy in the function
        // parameters stack. The caller needs to provide a pointer to the
        // original argument.
        attrs.addByVal(DtoAlignment(loweredDType));
        passPointer = true;
      } else {
        // Add sext/zext as needed.
        attrs.add(DtoShouldExtend(loweredDType));
      }
    }

    newIrFty.args.push_back(new IrFuncTyArg(loweredDType, passPointer, attrs));
    newIrFty.args.back()->parametersIdx = i;
    ++nextLLArgIdx;
  }

  // let the ABI rewrite the types as necessary
  abi->rewriteFunctionType(f, newIrFty);

  // Now we can modify irFty safely.
  irFty = llvm_move(newIrFty);

  // Finally build the actual LLVM function type.
  llvm::SmallVector<llvm::Type *, 16> argtypes;
  argtypes.reserve(nextLLArgIdx);

  if (irFty.arg_sret) {
    argtypes.push_back(irFty.arg_sret->ltype);
  }
  if (irFty.arg_this) {
    argtypes.push_back(irFty.arg_this->ltype);
  }
  if (irFty.arg_nest) {
    argtypes.push_back(irFty.arg_nest->ltype);
  }
  if (irFty.arg_arguments) {
    argtypes.push_back(irFty.arg_arguments->ltype);
  }

  if (irFty.arg_sret && irFty.arg_this && abi->passThisBeforeSret(f)) {
    std::swap(argtypes[0], argtypes[1]);
  }

  const size_t firstExplicitArg = argtypes.size();
  const size_t numExplicitLLArgs = irFty.args.size();
  for (size_t i = 0; i < numExplicitLLArgs; i++) {
    argtypes.push_back(irFty.args[i]->ltype);
  }

  // reverse params?
  if (irFty.reverseParams && numExplicitLLArgs > 1) {
    std::reverse(argtypes.begin() + firstExplicitArg, argtypes.end());
  }

  irFty.funcType =
      LLFunctionType::get(irFty.ret->ltype, argtypes, irFty.c_vararg);

  IF_LOG Logger::cout() << "Final function type: " << *irFty.funcType << "\n";

  return irFty.funcType;
}

////////////////////////////////////////////////////////////////////////////////

static llvm::FunctionType *DtoVaFunctionType(FuncDeclaration *fdecl) {
  IrFuncTy &irFty = getIrFunc(fdecl, true)->irFty;
  if (irFty.funcType) {
    return irFty.funcType;
  }

  irFty.ret = new IrFuncTyArg(Type::tvoid, false);

  irFty.args.push_back(new IrFuncTyArg(Type::tvoid->pointerTo(), false));

  if (fdecl->llvmInternal == LLVMva_start) {
    irFty.funcType = GET_INTRINSIC_DECL(vastart)->getFunctionType();
  } else if (fdecl->llvmInternal == LLVMva_copy) {
    irFty.funcType = GET_INTRINSIC_DECL(vacopy)->getFunctionType();
    irFty.args.push_back(new IrFuncTyArg(Type::tvoid->pointerTo(), false));
  } else if (fdecl->llvmInternal == LLVMva_end) {
    irFty.funcType = GET_INTRINSIC_DECL(vaend)->getFunctionType();
  }
  assert(irFty.funcType);

  return irFty.funcType;
}

////////////////////////////////////////////////////////////////////////////////

llvm::FunctionType *DtoFunctionType(FuncDeclaration *fdecl) {
  // handle for C vararg intrinsics
  if (DtoIsVaIntrinsic(fdecl)) {
    return DtoVaFunctionType(fdecl);
  }

  Type *dthis = nullptr, *dnest = nullptr;

  if (fdecl->ident == Id::ensure || fdecl->ident == Id::require) {
    FuncDeclaration *p = fdecl->parent->isFuncDeclaration();
    assert(p);
    AggregateDeclaration *ad = p->isMember2();
    assert(ad);
    dnest = Type::tvoid->pointerTo();
  } else if (fdecl->needThis()) {
    if (AggregateDeclaration *ad = fdecl->isMember2()) {
      IF_LOG Logger::println("isMember = this is: %s", ad->type->toChars());
      dthis = ad->type;
      LLType *thisty = DtoType(dthis);
      // Logger::cout() << "this llvm type: " << *thisty << '\n';
      if (ad->isStructDeclaration()) {
        thisty = getPtrToType(thisty);
      }
    } else {
      IF_LOG Logger::println("chars: %s type: %s kind: %s", fdecl->toChars(),
                             fdecl->type->toChars(), fdecl->kind());
      llvm_unreachable("needThis, but invalid parent declaration.");
    }
  } else if (fdecl->isNested()) {
    dnest = Type::tvoid->pointerTo();
  }

  LLFunctionType *functype = DtoFunctionType(
      fdecl->type, getIrFunc(fdecl, true)->irFty, dthis, dnest, fdecl->isMain(),
      fdecl->isCtorDeclaration(), DtoIsIntrinsic(fdecl));

  return functype;
}

////////////////////////////////////////////////////////////////////////////////

static llvm::Function *DtoDeclareVaFunction(FuncDeclaration *fdecl) {
  DtoVaFunctionType(fdecl);
  llvm::Function *func = nullptr;

  if (fdecl->llvmInternal == LLVMva_start) {
    func = GET_INTRINSIC_DECL(vastart);
  } else if (fdecl->llvmInternal == LLVMva_copy) {
    func = GET_INTRINSIC_DECL(vacopy);
  } else if (fdecl->llvmInternal == LLVMva_end) {
    func = GET_INTRINSIC_DECL(vaend);
  }
  assert(func);

  getIrFunc(fdecl)->func = func;
  return func;
}

////////////////////////////////////////////////////////////////////////////////

void DtoResolveFunction(FuncDeclaration *fdecl) {
  if ((!global.params.useUnitTests || !fdecl->type) &&
      fdecl->isUnitTestDeclaration()) {
    IF_LOG Logger::println("Ignoring unittest %s", fdecl->toPrettyChars());
    return; // ignore declaration completely
  }

  if (fdecl->ir.isResolved()) {
    return;
  }
  fdecl->ir.setResolved();

  Type *type = fdecl->type;
  // If errors occurred compiling it, such as bugzilla 6118
  if (type && type->ty == Tfunction) {
    Type *next = static_cast<TypeFunction *>(type)->next;
    if (!next || next->ty == Terror) {
      return;
    }
  }

  // printf("resolve function: %s\n", fdecl->toPrettyChars());

  if (fdecl->parent) {
    if (TemplateInstance *tinst = fdecl->parent->isTemplateInstance()) {
      if (TemplateDeclaration *tempdecl =
              tinst->tempdecl->isTemplateDeclaration()) {
        if (tempdecl->llvmInternal == LLVMva_arg) {
          Logger::println("magic va_arg found");
          fdecl->llvmInternal = LLVMva_arg;
          fdecl->ir.setDefined();
          return; // this gets mapped to an instruction so a declaration makes
                  // no sence
        }
        if (tempdecl->llvmInternal == LLVMva_start) {
          Logger::println("magic va_start found");
          fdecl->llvmInternal = LLVMva_start;
        } else if (tempdecl->llvmInternal == LLVMintrinsic) {
          Logger::println("overloaded intrinsic found");
          assert(fdecl->llvmInternal == LLVMintrinsic);
          assert(fdecl->mangleOverride);
        } else if (tempdecl->llvmInternal == LLVMinline_asm) {
          Logger::println("magic inline asm found");
          TypeFunction *tf = static_cast<TypeFunction *>(fdecl->type);
          if (tf->varargs != 1 ||
              (fdecl->parameters && fdecl->parameters->dim != 0)) {
            tempdecl->error("invalid __asm declaration, must be a D style "
                            "variadic with no explicit parameters");
            fatal();
          }
          fdecl->llvmInternal = LLVMinline_asm;
          fdecl->ir.setDefined();
          return; // this gets mapped to a special inline asm call, no point in
                  // going on.
        } else if (tempdecl->llvmInternal == LLVMinline_ir) {
          Logger::println("magic inline ir found");
          fdecl->llvmInternal = LLVMinline_ir;
          fdecl->linkage = LINKc;
          Type *type = fdecl->type;
          assert(type->ty == Tfunction);
          static_cast<TypeFunction *>(type)->linkage = LINKc;

          DtoFunctionType(fdecl);
          DtoDeclareFunction(fdecl);
          fdecl->ir.setDefined();
          return;
        }
      }
    }
  }

  DtoFunctionType(fdecl);

  IF_LOG Logger::println("DtoResolveFunction(%s): %s", fdecl->toPrettyChars(),
                         fdecl->loc.toChars());
  LOG_SCOPE;

  // queue declaration unless the function is abstract without body
  if (!fdecl->isAbstract() || fdecl->fbody) {
    DtoDeclareFunction(fdecl);
  }
}

////////////////////////////////////////////////////////////////////////////////

void applyParamAttrsToLLFunc(TypeFunction *f, IrFuncTy &irFty,
                             llvm::Function *func) {
  AttrSet newAttrs = AttrSet::extractFunctionAndReturnAttributes(func);
  newAttrs.merge(irFty.getParamAttrs(gABI->passThisBeforeSret(f)));
  func->setAttributes(newAttrs);
}

////////////////////////////////////////////////////////////////////////////////

void DtoDeclareFunction(FuncDeclaration *fdecl) {
  DtoResolveFunction(fdecl);

  if (fdecl->ir.isDeclared()) {
    return;
  }
  fdecl->ir.setDeclared();

  IF_LOG Logger::println("DtoDeclareFunction(%s): %s", fdecl->toPrettyChars(),
                         fdecl->loc.toChars());
  LOG_SCOPE;

  if (fdecl->isUnitTestDeclaration() && !global.params.useUnitTests) {
    Logger::println("unit tests not enabled");
    return;
  }

  // printf("declare function: %s\n", fdecl->toPrettyChars());

  // intrinsic sanity check
  if (DtoIsIntrinsic(fdecl) && fdecl->fbody) {
    error(fdecl->loc, "intrinsics cannot have function bodies");
    fatal();
  }

  // get TypeFunction*
  Type *t = fdecl->type->toBasetype();
  TypeFunction *f = static_cast<TypeFunction *>(t);

  // create IrFunction
  IrFunction *irFunc = getIrFunc(fdecl, true);

  LLFunction *vafunc = nullptr;
  if (DtoIsVaIntrinsic(fdecl)) {
    vafunc = DtoDeclareVaFunction(fdecl);
  }

  // calling convention
  LINK link = f->linkage;
  if (vafunc || DtoIsIntrinsic(fdecl)
      // DMD treats _Dmain as having C calling convention and this has been
      // hardcoded into druntime, even if the frontend type has D linkage.
      // See Bugzilla issue 9028.
      || fdecl->isMain()) {
    link = LINKc;
  }

  // mangled name
  std::string mangledName(mangleExact(fdecl));
  mangledName = gABI->mangleForLLVM(mangledName, link);

  // construct function
  LLFunctionType *functype = DtoFunctionType(fdecl);
  LLFunction *func = vafunc ? vafunc : gIR->module.getFunction(mangledName);
  if (!func) {
    if (fdecl->llvmInternal == LLVMinline_ir) {
      func = DtoInlineIRFunction(fdecl);
    } else {
      // All function declarations are "external" - any other linkage type
      // is set when actually defining the function.
      func = LLFunction::Create(functype, llvm::GlobalValue::ExternalLinkage,
                                mangledName, &gIR->module);
    }
  } else if (func->getFunctionType() != functype) {
    error(fdecl->loc, "Function type does not match previously declared "
                      "function with the same mangled name: %s",
          mangleExact(fdecl));
    fatal();
  }

  func->setCallingConv(gABI->callingConv(func->getFunctionType(), link, fdecl));

  IF_LOG Logger::cout() << "func = " << *func << std::endl;

  // add func to IRFunc
  irFunc->func = func;

  // parameter attributes
  if (!DtoIsIntrinsic(fdecl)) {
    applyParamAttrsToLLFunc(f, getIrFunc(fdecl)->irFty, func);
    if (global.params.disableRedZone) {
      func->addFnAttr(LLAttribute::NoRedZone);
    }
  }

  applyFuncDeclUDAs(fdecl, func);

  // main
  if (fdecl->isMain()) {
    // Detect multiple main functions, which is disallowed. DMD checks this
    // in the glue code, so we need to do it here as well.
    if (gIR->mainFunc) {
      error(fdecl->loc, "only one main function allowed");
    }
    gIR->mainFunc = func;
  }

  if (fdecl->neverInline) {
    irFunc->setNeverInline();
  }

  if (fdecl->llvmInternal == LLVMglobal_crt_ctor ||
      fdecl->llvmInternal == LLVMglobal_crt_dtor) {
    AppendFunctionToLLVMGlobalCtorsDtors(
        func, fdecl->priority, fdecl->llvmInternal == LLVMglobal_crt_ctor);
  }

  IrFuncTy &irFty = irFunc->irFty;

  // name parameters
  llvm::Function::arg_iterator iarg = func->arg_begin();

  const bool passThisBeforeSret =
      irFty.arg_sret && irFty.arg_this && gABI->passThisBeforeSret(f);

  if (irFty.arg_sret && !passThisBeforeSret) {
    iarg->setName(".sret_arg");
    irFunc->retArg = &(*iarg);
    ++iarg;
  }

  if (irFty.arg_this) {
    iarg->setName(".this_arg");
    irFunc->thisArg = &(*iarg);

    VarDeclaration *v = fdecl->vthis;
    if (v) {
      // We already build the this argument here if we will need it
      // later for codegen'ing the function, just as normal
      // parameters below, because it can be referred to in nested
      // context types. Will be given storage in DtoDefineFunction.
      assert(!isIrParameterCreated(v));
      IrParameter *irParam = getIrParameter(v, true);
      irParam->value = &(*iarg);
      irParam->arg = irFty.arg_this;
      irParam->isVthis = true;
    }

    ++iarg;
  } else if (irFty.arg_nest) {
    iarg->setName(".nest_arg");
    irFunc->nestArg = &(*iarg);
    assert(irFunc->nestArg);
    ++iarg;
  }

  if (passThisBeforeSret) {
    iarg->setName(".sret_arg");
    irFunc->retArg = &(*iarg);
    ++iarg;
  }

  if (irFty.arg_arguments) {
    iarg->setName("._arguments");
    irFunc->_arguments = &(*iarg);
    ++iarg;
  }

  unsigned int k = 0;
  for (; iarg != func->arg_end(); ++iarg) {
    size_t llExplicitIdx = irFty.reverseParams ? irFty.args.size() - k - 1 : k;
    ++k;
    IrFuncTyArg *arg = irFty.args[llExplicitIdx];

    if (!fdecl->parameters || arg->parametersIdx >= fdecl->parameters->dim) {
      iarg->setName("unnamed");
      continue;
    }

    Dsymbol *const argsym = (*fdecl->parameters)[arg->parametersIdx];
    VarDeclaration *argvd = argsym->isVarDeclaration();
    assert(argvd);

    iarg->setName(argvd->ident->toChars() + llvm::Twine("_arg"));

    IrParameter *irParam = getIrParameter(argvd, true);
    irParam->arg = arg;
    irParam->value = &(*iarg);
  }
}

////////////////////////////////////////////////////////////////////////////////

static LinkageWithCOMDAT lowerFuncLinkage(FuncDeclaration *fdecl) {
  // Intrinsics are always external.
  if (DtoIsIntrinsic(fdecl)) {
    return LinkageWithCOMDAT(LLGlobalValue::ExternalLinkage, false);
  }

  // Generated array op functions behave like templates in that they might be
  // emitted into many different modules.
  if (fdecl->isArrayOp && (willInline() || !isDruntimeArrayOp(fdecl))) {
    return LinkageWithCOMDAT(templateLinkage, supportsCOMDAT());
  }

  // A body-less declaration always needs to be marked as external in LLVM
  // (also e.g. naked template functions which would otherwise be weak_odr,
  // but where the definition is in module-level inline asm).
  if (!fdecl->fbody || fdecl->naked) {
    return LinkageWithCOMDAT(LLGlobalValue::ExternalLinkage, false);
  }

  return DtoLinkage(fdecl);
}

void DtoDefineFunction(FuncDeclaration *fd) {
  IF_LOG Logger::println("DtoDefineFunction(%s): %s", fd->toPrettyChars(),
                         fd->loc.toChars());
  LOG_SCOPE;

  if (fd->ir.isDefined()) {
    return;
  }

  if ((fd->type && fd->type->ty == Terror) ||
      (fd->type && fd->type->ty == Tfunction &&
       static_cast<TypeFunction *>(fd->type)->next == nullptr) ||
      (fd->type && fd->type->ty == Tfunction &&
       static_cast<TypeFunction *>(fd->type)->next->ty == Terror)) {
    IF_LOG Logger::println(
        "Ignoring; has error type, no return type or returns error type");
    fd->ir.setDefined();
    return;
  }

  if (fd->semanticRun == PASSsemanticdone) {
    /* What happened is this function failed semantic3() with errors,
     * but the errors were gagged.
     * Try to reproduce those errors, and then fail.
     */
    error(fd->loc, "errors compiling function %s", fd->toPrettyChars());
    fd->ir.setDefined();
    return;
  }

  DtoResolveFunction(fd);

  if (fd->isUnitTestDeclaration() && !global.params.useUnitTests) {
    IF_LOG Logger::println("No code generation for unit test declaration %s",
                           fd->toChars());
    fd->ir.setDefined();
    return;
  }

  // Skip array ops implemented in druntime
  if (fd->isArrayOp && !willInline() && isDruntimeArrayOp(fd)) {
    IF_LOG Logger::println(
        "No code generation for array op %s implemented in druntime",
        fd->toChars());
    fd->ir.setDefined();
    return;
  }

  // Check whether the frontend knows that the function is already defined
  // in some other module (see DMD's FuncDeclaration::toObjFile).
  for (FuncDeclaration *f = fd; f;) {
    if (!f->isInstantiated() && f->inNonRoot()) {
      IF_LOG Logger::println("Skipping '%s'.", fd->toPrettyChars());
      // TODO: Emit as available_externally for inlining purposes instead
      // (see #673).
      fd->ir.setDefined();
      return;
    }
    if (f->isNested()) {
      f = f->toParent2()->isFuncDeclaration();
    } else {
      break;
    }
  }

  DtoDeclareFunction(fd);
  assert(fd->ir.isDeclared());

  // DtoResolveFunction might also set the defined flag for functions we
  // should not touch.
  if (fd->ir.isDefined()) {
    return;
  }
  fd->ir.setDefined();

  // We cannot emit nested functions with parents that have not gone through
  // semantic analysis. This can happen as DMD leaks some template instances
  // from constraints into the module member list. DMD gets away with being
  // sloppy as functions in template contraints obviously never need to access
  // data from the template function itself, but it would still mess up our
  // nested context creation code.
  FuncDeclaration *parent = fd;
  while ((parent = getParentFunc(parent, true))) {
    if (parent->semanticRun != PASSsemantic3done || parent->semantic3Errors) {
      IF_LOG Logger::println(
          "Ignoring nested function with unanalyzed parent.");
      return;
    }
  }

  assert(fd->semanticRun == PASSsemantic3done);
  assert(fd->ident != Id::empty);

  if (fd->isUnitTestDeclaration()) {
    getIrModule(gIR->dmodule)->unitTests.push_back(fd);
  } else if (fd->isSharedStaticCtorDeclaration()) {
    getIrModule(gIR->dmodule)->sharedCtors.push_back(fd);
  } else if (StaticDtorDeclaration *dtorDecl =
                 fd->isSharedStaticDtorDeclaration()) {
    getIrModule(gIR->dmodule)->sharedDtors.push_front(fd);
    if (dtorDecl->vgate) {
      getIrModule(gIR->dmodule)->sharedGates.push_front(dtorDecl->vgate);
    }
  } else if (fd->isStaticCtorDeclaration()) {
    getIrModule(gIR->dmodule)->ctors.push_back(fd);
  } else if (StaticDtorDeclaration *dtorDecl = fd->isStaticDtorDeclaration()) {
    getIrModule(gIR->dmodule)->dtors.push_front(fd);
    if (dtorDecl->vgate) {
      getIrModule(gIR->dmodule)->gates.push_front(dtorDecl->vgate);
    }
  }

  // if this function is naked, we take over right away! no standard processing!
  if (fd->naked) {
    DtoDefineNakedFunction(fd);
    return;
  }

  IrFunction *irFunc = getIrFunc(fd);
  IrFuncTy &irFty = irFunc->irFty;

  // debug info
  irFunc->diSubprogram = gIR->DBuilder.EmitSubProgram(fd);

  Type *t = fd->type->toBasetype();
  TypeFunction *f = static_cast<TypeFunction *>(t);
  // assert(f->ctype);

  llvm::Function *func = irFunc->func;

  // is there a body?
  if (fd->fbody == nullptr) {
    return;
  }

  IF_LOG Logger::println("Doing function body for: %s", fd->toChars());
  gIR->functions.push_back(irFunc);

  const auto lwc = lowerFuncLinkage(fd);
  setLinkage(lwc, func);

  // On x86_64, always set 'uwtable' for System V ABI compatibility.
  // TODO: Find a better place for this.
  // TODO: Is this required for Win64 as well?
  if (global.params.targetTriple.getArch() == llvm::Triple::x86_64) {
    func->addFnAttr(LLAttribute::UWTable);
  }
  if (opts::sanitize != opts::None) {
    // Set the required sanitizer attribute.
    if (opts::sanitize == opts::AddressSanitizer) {
      func->addFnAttr(LLAttribute::SanitizeAddress);
    }

    if (opts::sanitize == opts::MemorySanitizer) {
      func->addFnAttr(LLAttribute::SanitizeMemory);
    }

    if (opts::sanitize == opts::ThreadSanitizer) {
      func->addFnAttr(LLAttribute::SanitizeThread);
    }
  }

  llvm::BasicBlock *beginbb =
      llvm::BasicBlock::Create(gIR->context(), "", func);

  // assert(gIR->scopes.empty());
  gIR->scopes.push_back(IRScope(beginbb));

  // create alloca point
  // this gets erased when the function is complete, so alignment etc does not
  // matter at all
  llvm::Instruction *allocaPoint =
      new llvm::AllocaInst(LLType::getInt32Ty(gIR->context()),
#if LDC_LLVM_VER >= 500
                           0, // Address space
#endif
                           "alloca_point", beginbb);
  irFunc->allocapoint = allocaPoint;

  // debug info - after all allocas, but before any llvm.dbg.declare etc
  gIR->DBuilder.EmitFuncStart(fd);

  // this hack makes sure the frame pointer elimination optimization is
  // disabled.
  // this this eliminates a bunch of inline asm related issues.
  if (fd->hasReturnExp & 8) // has inline asm
  {
    // emit a call to llvm_eh_unwind_init
    LLFunction *hack = GET_INTRINSIC_DECL(eh_unwind_init);
#if LDC_LLVM_VER >= 307
    gIR->ir->CreateCall(hack, {});
#else
    gIR->ir->CreateCall(hack, "");
#endif
  }

  // give the 'this' argument storage and debug info
  if (irFty.arg_this) {
    LLValue *thisvar = irFunc->thisArg;
    assert(thisvar);

    LLValue *thismem = thisvar;
    if (!irFty.arg_this->byref) {
      thismem = DtoAllocaDump(thisvar, 0, "this");
      irFunc->thisArg = thismem;
    }

    assert(getIrParameter(fd->vthis)->value == thisvar);
    getIrParameter(fd->vthis)->value = thismem;

    gIR->DBuilder.EmitLocalVariable(thismem, fd->vthis, nullptr, true);
  }

  // give the 'nestArg' storage
  if (irFty.arg_nest) {
    irFunc->nestArg = DtoAllocaDump(irFunc->nestArg, 0, "nestedFrame");
  }

  // give arguments storage and debug info
  if (fd->parameters) {
    // Not all arguments are necessarily passed on the LLVM level
    // (e.g. zero-member structs), so we need to keep track of the
    // index in the IrFuncTy args array separately.
    size_t llArgIdx = 0;
    for (size_t i = 0; i < fd->parameters->dim; ++i) {
      Dsymbol *const argsym = (*fd->parameters)[i];
      VarDeclaration *const vd = argsym->isVarDeclaration();
      assert(vd);
      const bool refout = vd->storage_class & (STCref | STCout);

      IrParameter *irparam = getIrParameter(vd);
      Type *debugInfoType = vd->type;
      if (!irparam) {
        // This is a parameter that is not passed on the LLVM level.
        // Create the param here and set it to a "dummy" alloca that
        // we do not store to here.
        irparam = getIrParameter(vd, true);
        irparam->value = DtoAlloca(vd, vd->ident->toChars());
      } else {
        const bool lazy = vd->storage_class & STClazy;
        const bool firstClassVal = !refout && (!irparam->arg->byref || lazy);
        if (firstClassVal) {
          // alloca a stack slot for this first class value arg
          LLValue *mem = DtoAlloca(irparam->arg->type, vd->ident->toChars());

          // let the abi transform the argument back first
          irFty.getParam(vd->type, llArgIdx, irparam->value, mem);

          // set the arg var value to the alloca
          irparam->value = mem;

          debugInfoType = irparam->arg->type;
        }
        ++llArgIdx;
      }

      if (global.params.symdebug &&
          !(isaArgument(irparam->value) &&
            isaArgument(irparam->value)->hasByValAttr())) {
        gIR->DBuilder.EmitLocalVariable(irparam->value, vd, debugInfoType);
      }
    }
  }

  {
    ScopeStack scopeStack(gIR);
    irFunc->scopes = &scopeStack;

    DtoCreateNestedContext(fd);

    if (fd->vresult && !fd->vresult->nestedrefs.dim) // FIXME: not sure here :/
    {
      DtoVarDeclaration(fd->vresult);
    }

    // D varargs: prepare _argptr and _arguments
    if (f->linkage == LINKd && f->varargs == 1) {
      // allocate _argptr (of type core.stdc.stdarg.va_list)
      LLValue *argptrmem = DtoAlloca(Type::tvalist->semantic(fd->loc, fd->scope), "_argptr_mem");
      irFunc->_argptr = argptrmem;

      // initialize _argptr with a call to the va_start intrinsic
      LLValue *vaStartArg = gABI->prepareVaStart(argptrmem);
      llvm::CallInst::Create(GET_INTRINSIC_DECL(vastart), vaStartArg, "",
                             gIR->scopebb());

      // copy _arguments to a memory location
      irFunc->_arguments =
          DtoAllocaDump(irFunc->_arguments, 0, "_arguments_mem");
    }

    // output function body
    Statement_toIR(fd->fbody, gIR);

    irFunc->scopes = nullptr;
  }

  llvm::BasicBlock *bb = gIR->scopebb();
  if (pred_begin(bb) == pred_end(bb) &&
      bb != &bb->getParent()->getEntryBlock()) {
    // This block is trivially unreachable, so just delete it.
    // (This is a common case because it happens when 'return'
    // is the last statement in a function)
    bb->eraseFromParent();
  } else if (!gIR->scopereturned()) {
    // llvm requires all basic blocks to end with a TerminatorInst but DMD does
    // not put a return statement in automatically, so we do it here.

    // pass the previous block into this block
    gIR->DBuilder.EmitStopPoint(fd->endloc);
    if (func->getReturnType() == LLType::getVoidTy(gIR->context())) {
      gIR->ir->CreateRetVoid();
    } else if (!fd->isMain()) {
      CompoundAsmStatement *asmb = fd->fbody->endsWithAsm();
      if (asmb) {
        assert(asmb->abiret);
        gIR->ir->CreateRet(asmb->abiret);
      } else {
        gIR->ir->CreateRet(llvm::UndefValue::get(func->getReturnType()));
      }
    } else {
      gIR->ir->CreateRet(LLConstant::getNullValue(func->getReturnType()));
    }
  }
  gIR->DBuilder.EmitFuncEnd(fd);

  // erase alloca point
  if (allocaPoint->getParent()) {
    allocaPoint->eraseFromParent();
  }
  allocaPoint = nullptr;
  gIR->func()->allocapoint = nullptr;

  gIR->scopes.pop_back();

  gIR->functions.pop_back();
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoArgument(Parameter *fnarg, Expression *argexp) {
  IF_LOG Logger::println("DtoArgument");
  LOG_SCOPE;

  // ref/out arg
  if (fnarg && (fnarg->storageClass & (STCref | STCout))) {
    Loc loc;
    DValue *arg = toElem(argexp, true);
    return new DImValue(argexp->type,
                        arg->isLVal() ? arg->getLVal() : makeLValue(loc, arg));
  }

  DValue *arg = toElem(argexp);

  // lazy arg
  if (fnarg && (fnarg->storageClass & STClazy)) {
    assert(argexp->type->toBasetype()->ty == Tdelegate);
    assert(!arg->isLVal());
    return arg;
  }

  // byval arg, but expr has no storage yet
  if (DtoIsInMemoryOnly(argexp->type) && (arg->isSlice() || arg->isNull())) {
    LLValue *alloc = DtoAlloca(argexp->type, ".tmp_arg");
    auto vv = new DVarValue(argexp->type, alloc);
    DtoAssign(argexp->loc, vv, arg);
    arg = vv;
  }

  return arg;
}

////////////////////////////////////////////////////////////////////////////////

int binary(const char *p, const char **tab, int high) {
  int i = 0, j = high, k, l;
  do {
    k = (i + j) / 2;
    l = strcmp(p, tab[k]);
    if (!l) {
      return k;
    }
    if (l < 0) {
      j = k;
    } else {
      i = k + 1;
    }
  } while (i != j);
  return -1;
}

int isDruntimeArrayOp(FuncDeclaration *fd) {
  /* Some of the array op functions are written as library functions,
   * presumably to optimize them with special CPU vector instructions.
   * List those library functions here, in alpha order.
   */
  static const char *libArrayopFuncs[] = {
      "_arrayExpSliceAddass_a",           "_arrayExpSliceAddass_d",
      "_arrayExpSliceAddass_f", // T[]+=T
      "_arrayExpSliceAddass_g",           "_arrayExpSliceAddass_h",
      "_arrayExpSliceAddass_i",           "_arrayExpSliceAddass_k",
      "_arrayExpSliceAddass_s",           "_arrayExpSliceAddass_t",
      "_arrayExpSliceAddass_u",           "_arrayExpSliceAddass_w",

      "_arrayExpSliceDivass_d", // T[]/=T
      "_arrayExpSliceDivass_f", // T[]/=T

      "_arrayExpSliceMinSliceAssign_a",
      "_arrayExpSliceMinSliceAssign_d", // T[]=T-T[]
      "_arrayExpSliceMinSliceAssign_f", // T[]=T-T[]
      "_arrayExpSliceMinSliceAssign_g",   "_arrayExpSliceMinSliceAssign_h",
      "_arrayExpSliceMinSliceAssign_i",   "_arrayExpSliceMinSliceAssign_k",
      "_arrayExpSliceMinSliceAssign_s",   "_arrayExpSliceMinSliceAssign_t",
      "_arrayExpSliceMinSliceAssign_u",   "_arrayExpSliceMinSliceAssign_w",

      "_arrayExpSliceMinass_a",
      "_arrayExpSliceMinass_d", // T[]-=T
      "_arrayExpSliceMinass_f", // T[]-=T
      "_arrayExpSliceMinass_g",           "_arrayExpSliceMinass_h",
      "_arrayExpSliceMinass_i",           "_arrayExpSliceMinass_k",
      "_arrayExpSliceMinass_s",           "_arrayExpSliceMinass_t",
      "_arrayExpSliceMinass_u",           "_arrayExpSliceMinass_w",

      "_arrayExpSliceMulass_d", // T[]*=T
      "_arrayExpSliceMulass_f", // T[]*=T
      "_arrayExpSliceMulass_i",           "_arrayExpSliceMulass_k",
      "_arrayExpSliceMulass_s",           "_arrayExpSliceMulass_t",
      "_arrayExpSliceMulass_u",           "_arrayExpSliceMulass_w",

      "_arraySliceExpAddSliceAssign_a",
      "_arraySliceExpAddSliceAssign_d", // T[]=T[]+T
      "_arraySliceExpAddSliceAssign_f", // T[]=T[]+T
      "_arraySliceExpAddSliceAssign_g",   "_arraySliceExpAddSliceAssign_h",
      "_arraySliceExpAddSliceAssign_i",   "_arraySliceExpAddSliceAssign_k",
      "_arraySliceExpAddSliceAssign_s",   "_arraySliceExpAddSliceAssign_t",
      "_arraySliceExpAddSliceAssign_u",   "_arraySliceExpAddSliceAssign_w",

      "_arraySliceExpDivSliceAssign_d", // T[]=T[]/T
      "_arraySliceExpDivSliceAssign_f", // T[]=T[]/T

      "_arraySliceExpMinSliceAssign_a",
      "_arraySliceExpMinSliceAssign_d", // T[]=T[]-T
      "_arraySliceExpMinSliceAssign_f", // T[]=T[]-T
      "_arraySliceExpMinSliceAssign_g",   "_arraySliceExpMinSliceAssign_h",
      "_arraySliceExpMinSliceAssign_i",   "_arraySliceExpMinSliceAssign_k",
      "_arraySliceExpMinSliceAssign_s",   "_arraySliceExpMinSliceAssign_t",
      "_arraySliceExpMinSliceAssign_u",   "_arraySliceExpMinSliceAssign_w",

      "_arraySliceExpMulSliceAddass_d", // T[] += T[]*T
      "_arraySliceExpMulSliceAddass_f",   "_arraySliceExpMulSliceAddass_r",

      "_arraySliceExpMulSliceAssign_d", // T[]=T[]*T
      "_arraySliceExpMulSliceAssign_f", // T[]=T[]*T
      "_arraySliceExpMulSliceAssign_i",   "_arraySliceExpMulSliceAssign_k",
      "_arraySliceExpMulSliceAssign_s",   "_arraySliceExpMulSliceAssign_t",
      "_arraySliceExpMulSliceAssign_u",   "_arraySliceExpMulSliceAssign_w",

      "_arraySliceExpMulSliceMinass_d", // T[] -= T[]*T
      "_arraySliceExpMulSliceMinass_f",   "_arraySliceExpMulSliceMinass_r",

      "_arraySliceSliceAddSliceAssign_a",
      "_arraySliceSliceAddSliceAssign_d", // T[]=T[]+T[]
      "_arraySliceSliceAddSliceAssign_f", // T[]=T[]+T[]
      "_arraySliceSliceAddSliceAssign_g", "_arraySliceSliceAddSliceAssign_h",
      "_arraySliceSliceAddSliceAssign_i", "_arraySliceSliceAddSliceAssign_k",
      "_arraySliceSliceAddSliceAssign_r", // T[]=T[]+T[]
      "_arraySliceSliceAddSliceAssign_s", "_arraySliceSliceAddSliceAssign_t",
      "_arraySliceSliceAddSliceAssign_u", "_arraySliceSliceAddSliceAssign_w",

      "_arraySliceSliceAddass_a",
      "_arraySliceSliceAddass_d", // T[]+=T[]
      "_arraySliceSliceAddass_f", // T[]+=T[]
      "_arraySliceSliceAddass_g",         "_arraySliceSliceAddass_h",
      "_arraySliceSliceAddass_i",         "_arraySliceSliceAddass_k",
      "_arraySliceSliceAddass_s",         "_arraySliceSliceAddass_t",
      "_arraySliceSliceAddass_u",         "_arraySliceSliceAddass_w",

      "_arraySliceSliceMinSliceAssign_a",
      "_arraySliceSliceMinSliceAssign_d", // T[]=T[]-T[]
      "_arraySliceSliceMinSliceAssign_f", // T[]=T[]-T[]
      "_arraySliceSliceMinSliceAssign_g", "_arraySliceSliceMinSliceAssign_h",
      "_arraySliceSliceMinSliceAssign_i", "_arraySliceSliceMinSliceAssign_k",
      "_arraySliceSliceMinSliceAssign_r", // T[]=T[]-T[]
      "_arraySliceSliceMinSliceAssign_s", "_arraySliceSliceMinSliceAssign_t",
      "_arraySliceSliceMinSliceAssign_u", "_arraySliceSliceMinSliceAssign_w",

      "_arraySliceSliceMinass_a",
      "_arraySliceSliceMinass_d", // T[]-=T[]
      "_arraySliceSliceMinass_f", // T[]-=T[]
      "_arraySliceSliceMinass_g",         "_arraySliceSliceMinass_h",
      "_arraySliceSliceMinass_i",         "_arraySliceSliceMinass_k",
      "_arraySliceSliceMinass_s",         "_arraySliceSliceMinass_t",
      "_arraySliceSliceMinass_u",         "_arraySliceSliceMinass_w",

      "_arraySliceSliceMulSliceAssign_d", // T[]=T[]*T[]
      "_arraySliceSliceMulSliceAssign_f", // T[]=T[]*T[]
      "_arraySliceSliceMulSliceAssign_i", "_arraySliceSliceMulSliceAssign_k",
      "_arraySliceSliceMulSliceAssign_s", "_arraySliceSliceMulSliceAssign_t",
      "_arraySliceSliceMulSliceAssign_u", "_arraySliceSliceMulSliceAssign_w",

      "_arraySliceSliceMulass_d", // T[]*=T[]
      "_arraySliceSliceMulass_f", // T[]*=T[]
      "_arraySliceSliceMulass_i",         "_arraySliceSliceMulass_k",
      "_arraySliceSliceMulass_s",         "_arraySliceSliceMulass_t",
      "_arraySliceSliceMulass_u",         "_arraySliceSliceMulass_w",
  };
  char *name = fd->ident->toChars();
  int i =
      binary(name, libArrayopFuncs, sizeof(libArrayopFuncs) / sizeof(char *));
  if (i != -1) {
    return 1;
  }

#ifdef DEBUG // Make sure our array is alphabetized
  for (i = 0; i < sizeof(libArrayopFuncs) / sizeof(char *); i++) {
    if (strcmp(name, libArrayopFuncs[i]) == 0)
      assert(0);
  }
#endif
  return 0;
}
