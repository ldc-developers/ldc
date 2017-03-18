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
#include "ldcbindings.h"
#include "module.h"
#include "mtype.h"
#include "statement.h"
#include "template.h"
#include "driver/cl_options.h"
#include "gen/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/dvalue.h"
#include "gen/funcgenstate.h"
#include "gen/function-inlining.h"
#include "gen/inlineir.h"
#include "gen/irstate.h"
#include "gen/linkage.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/mangling.h"
#include "gen/nested.h"
#include "gen/optimizer.h"
#include "gen/pgo.h"
#include "gen/pragma.h"
#include "gen/runtime.h"
#include "gen/scope_exit.h"
#include "gen/tollvm.h"
#include "gen/uda.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/CFG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <iostream>

llvm::FunctionType *DtoFunctionType(Type *type, IrFuncTy &irFty, Type *thistype,
                                    Type *nesttype, bool isMain, bool isCtor,
                                    bool isIntrinsic, bool hasSel) {
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
      if (unsigned alignment = DtoAlignment(rt))
        newIrFty.arg_sret->attrs.addAlignment(alignment);
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

  if (hasSel) {
    // TODO: make arg_objcselector to match dmd type
    newIrFty.arg_objcSelector = new IrFuncTyArg(Type::tvoidptr, false);
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
      auto ltf = TypeFunction::create(nullptr, arg->type, 0, LINKd);
      auto ltd = createTypeDelegate(ltf);
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
  if (irFty.arg_objcSelector) {
    argtypes.push_back(irFty.arg_objcSelector->ltype);
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
  bool hasSel = false;

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

  if (fdecl->linkage == LINKobjc && dthis) {
    if (fdecl->objc.selector) {
      hasSel = true;
    } else if (fdecl->parent->isClassDeclaration()) {
      fdecl->error("Objective-C @selector is missing");
    }
  }

  LLFunctionType *functype = DtoFunctionType(
      fdecl->type, getIrFunc(fdecl, true)->irFty, dthis, dnest, fdecl->isMain(),
      fdecl->isCtorDeclaration(), DtoIsIntrinsic(fdecl), hasSel);

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

  getIrFunc(fdecl)->setLLVMFunc(func);
  return func;
}

////////////////////////////////////////////////////////////////////////////////

void DtoResolveFunction(FuncDeclaration *fdecl) {
  if ((!global.params.useUnitTests || !fdecl->type) &&
      fdecl->isUnitTestDeclaration()) {
    IF_LOG Logger::println("Ignoring unittest %s", fdecl->toPrettyChars());
    return; // ignore declaration completely
  }

  if (fdecl->ir->isResolved()) {
    return;
  }
  fdecl->ir->setResolved();

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
          fdecl->ir->setDefined();
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
          fdecl->ir->setDefined();
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
          fdecl->ir->setDefined();
          return; // this gets mapped to a special inline IR call, no point in
                  // going on.
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

namespace {

void applyParamAttrsToLLFunc(TypeFunction *f, IrFuncTy &irFty,
                             llvm::Function *func) {
  AttrSet newAttrs = AttrSet::extractFunctionAndReturnAttributes(func);
  newAttrs.merge(irFty.getParamAttrs(gABI->passThisBeforeSret(f)));
  func->setAttributes(newAttrs);
}

/// Applies TargetMachine options as function attributes in the IR (options for
/// which attributes exist).
/// This is e.g. needed for LTO: it tells the linker/LTO-codegen what settings
/// to use.
/// It is also needed because "unsafe-fp-math" is not properly reset in LLVM
/// between function definitions, i.e. if a function does not define a value for
/// "unsafe-fp-math" it will be compiled using the value of the previous
/// function. Therefore, each function must explicitly define the value (clang
/// does the same). See https://llvm.org/bugs/show_bug.cgi?id=23172
void applyTargetMachineAttributes(llvm::Function &func,
                                  const llvm::TargetMachine &target) {
  const llvm::TargetOptions &TO = target.Options;

  // TODO: implement commandline switches to change the default values.

  // Target CPU capabilities
  func.addFnAttr("target-cpu", target.getTargetCPU());
  auto featStr = target.getTargetFeatureString();
  if (!featStr.empty())
    func.addFnAttr("target-features", featStr);

  // Floating point settings
  func.addFnAttr("unsafe-fp-math", TO.UnsafeFPMath ? "true" : "false");
  const bool lessPreciseFPMADOption =
#if LDC_LLVM_VER >= 500
      // This option was removed from llvm::TargetOptions in LLVM 5.0.
      // Clang sets this to true when `-cl-mad-enable` is passed (OpenCL only).
      // TODO: implement interface for this option.
      false;
#else
      TO.LessPreciseFPMADOption;
#endif
  func.addFnAttr("less-precise-fpmad",
                 lessPreciseFPMADOption ? "true" : "false");
  func.addFnAttr("no-infs-fp-math", TO.NoInfsFPMath ? "true" : "false");
  func.addFnAttr("no-nans-fp-math", TO.NoNaNsFPMath ? "true" : "false");
#if LDC_LLVM_VER < 307
  func.addFnAttr("use-soft-float", TO.UseSoftFloat ? "true" : "false");
#endif

  // Frame pointer elimination
  func.addFnAttr("no-frame-pointer-elim",
                 opts::disableFpElim ? "true" : "false");
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

void DtoDeclareFunction(FuncDeclaration *fdecl) {
  DtoResolveFunction(fdecl);

  if (fdecl->ir->isDeclared()) {
    return;
  }
  fdecl->ir->setDeclared();

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

  // Check if fdecl should be defined too for cross-module inlining.
  // If true, semantic is fully done for fdecl which is needed for some code
  // below (e.g. code that uses fdecl->vthis).
  const bool defineAtEnd = defineAsExternallyAvailable(*fdecl);
  if (defineAtEnd) {
    IF_LOG Logger::println(
        "Function is an externally_available inline candidate.");
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
  std::string mangledName = getMangledName(fdecl, link);

  // construct function
  LLFunctionType *functype = DtoFunctionType(fdecl);
  LLFunction *func = vafunc ? vafunc : gIR->module.getFunction(mangledName);
  if (!func) {
    // All function declarations are "external" - any other linkage type
    // is set when actually defining the function.
    func = LLFunction::Create(functype, llvm::GlobalValue::ExternalLinkage,
                              mangledName, &gIR->module);
  } else if (func->getFunctionType() != functype) {
    error(fdecl->loc, "Function type does not match previously declared "
                      "function with the same mangled name: %s",
          mangleExact(fdecl));
    fatal();
  }

  func->setCallingConv(gABI->callingConv(func->getFunctionType(), link, fdecl));

  if (global.params.isWindows && fdecl->isExport()) {
    func->setDLLStorageClass(fdecl->isImportedSymbol()
                                 ? LLGlobalValue::DLLImportStorageClass
                                 : LLGlobalValue::DLLExportStorageClass);
  }

  IF_LOG Logger::cout() << "func = " << *func << std::endl;

  // add func to IRFunc
  irFunc->setLLVMFunc(func);

  // parameter attributes
  if (!DtoIsIntrinsic(fdecl)) {
    applyParamAttrsToLLFunc(f, getIrFunc(fdecl)->irFty, func);
    if (global.params.disableRedZone) {
      func->addFnAttr(LLAttribute::NoRedZone);
    }
  }

  // First apply the TargetMachine attributes, such that they can be overridden
  // by UDAs.
  applyTargetMachineAttributes(*func, *gTargetMachine);
  applyFuncDeclUDAs(fdecl, irFunc);

  // main
  if (fdecl->isMain()) {
    // Detect multiple main functions, which is disallowed. DMD checks this
    // in the glue code, so we need to do it here as well.
    if (gIR->mainFunc) {
      error(fdecl->loc, "only one main function allowed");
    }
    gIR->mainFunc = func;
  }

  // Set inlining attribute
  if (fdecl->neverInline) {
    irFunc->setNeverInline();
  } else {
    if (fdecl->inlining == PINLINEalways) {
      irFunc->setAlwaysInline();
    } else if (fdecl->inlining == PINLINEnever) {
      irFunc->setNeverInline();
    }
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
    irFunc->sretArg = &(*iarg);
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

  // TODO: do we need this?
  if (irFty.arg_objcSelector) {
    iarg->setName(".objcSelector_arg");
    irFunc->thisArg = &(*iarg);
    ++iarg;
  }

  if (passThisBeforeSret) {
    iarg->setName(".sret_arg");
    irFunc->sretArg = &(*iarg);
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

    auto *const vd = (*fdecl->parameters)[arg->parametersIdx];
    iarg->setName(vd->ident->toChars() + llvm::Twine("_arg"));

    IrParameter *irParam = getIrParameter(vd, true);
    irParam->arg = arg;
    irParam->value = &(*iarg);
  }

  // Now that this function is declared, also define it if needed.
  if (defineAtEnd) {
    IF_LOG Logger::println(
        "Function is an externally_available inline candidate: define it now.");
    DtoDefineFunction(fdecl, true);
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

// LDC has the same problem with destructors of struct arguments in closures
// as DMD, so we copy the failure detection
void verifyScopedDestructionInClosure(FuncDeclaration *fd) {
  for (size_t i = 0; i < fd->closureVars.dim; i++) {
    VarDeclaration *v = fd->closureVars[i];

    // Hack for the case fail_compilation/fail10666.d, until
    // proper issue https://issues.dlang.org/show_bug.cgi?id=5730 fix will come.
    bool isScopeDtorParam = v->edtor && (v->storage_class & STCparameter);
    if (v->needsScopeDtor() || isScopeDtorParam) {
      // Because the value needs to survive the end of the scope!
      v->error("has scoped destruction, cannot build closure");
    }
    if (v->isargptr) {
      // See https://issues.dlang.org/show_bug.cgi?id=2479
      // This is actually a bug, but better to produce a nice
      // message at compile time rather than memory corruption at runtime
      v->error("cannot reference variadic arguments from closure");
    }
  }
}

namespace {

// Gives all explicit parameters storage and debug info.
// All explicit D parameters are lvalues, just like regular local variables.
void defineParameters(IrFuncTy &irFty, VarDeclarations &parameters) {
  // Not all arguments are necessarily passed on the LLVM level
  // (e.g. zero-member structs), so we need to keep track of the
  // index in the IrFuncTy args array separately.
  size_t llArgIdx = 0;

  for (size_t i = 0; i < parameters.dim; ++i) {
    auto *const vd = parameters[i];
    IrParameter *irparam = getIrParameter(vd);

    // vd->type (parameter) and irparam->arg->type (argument) don't always
    // match.
    // E.g., for a lazy parameter of type T, vd->type is T (with lazy storage
    // class) while irparam->arg->type is the delegate type.
    Type *const paramType = (irparam ? irparam->arg->type : vd->type);

    if (!irparam) {
      // This is a parameter that is not passed on the LLVM level.
      // Create the param here and set it to a "dummy" alloca that
      // we do not store to here.
      irparam = getIrParameter(vd, true);
      irparam->value = DtoAlloca(vd, vd->ident->toChars());
    } else {
      assert(irparam->value);

      if (irparam->arg->byref) {
        // The argument is an appropriate lvalue passed by reference.
        // Use the passed pointer as parameter storage.
        assert(irparam->value->getType() == DtoPtrToType(paramType));
      } else {
        // Let the ABI transform the parameter back to an lvalue.
        irparam->value =
            irFty.getParamLVal(paramType, llArgIdx, irparam->value);
      }

      irparam->value->setName(vd->ident->toChars());

      ++llArgIdx;
    }

    if (global.params.symdebug)
      gIR->DBuilder.EmitLocalVariable(irparam->value, vd, paramType);
  }
}

} // anonymous namespace

void DtoDefineFunction(FuncDeclaration *fd, bool linkageAvailableExternally) {
  IF_LOG Logger::println("DtoDefineFunction(%s): %s", fd->toPrettyChars(),
                         fd->loc.toChars());
  LOG_SCOPE;
  if (linkageAvailableExternally) {
    IF_LOG Logger::println("linkageAvailableExternally = true");
  }

  if (fd->ir->isDefined()) {
    llvm::Function *func = getIrFunc(fd)->getLLVMFunc();
    assert(nullptr != func);
    if (!linkageAvailableExternally &&
        (func->getLinkage() ==
         llvm::GlobalValue::AvailableExternallyLinkage)) {
      // Fix linkage
      const auto lwc = lowerFuncLinkage(fd);
      setLinkage(lwc, func);
    }
    return;
  }

  if ((fd->type && fd->type->ty == Terror) ||
      (fd->type && fd->type->ty == Tfunction &&
       static_cast<TypeFunction *>(fd->type)->next == nullptr) ||
      (fd->type && fd->type->ty == Tfunction &&
       static_cast<TypeFunction *>(fd->type)->next->ty == Terror)) {
    IF_LOG Logger::println(
        "Ignoring; has error type, no return type or returns error type");
    fd->ir->setDefined();
    return;
  }

  if (fd->semanticRun == PASSsemanticdone) {
    // This function failed semantic3() with errors but the errors were gagged.
    // In contrast to DMD we immediately bail out here, since other parts of
    // the codegen expect irFunc to be set for defined functions.
    error(fd->loc, "Internal Compiler Error: function not fully analyzed; "
                   "previous unreported errors compiling %s?",
          fd->toPrettyChars());
    fatal();
  }

  DtoResolveFunction(fd);

  if (fd->isUnitTestDeclaration() && !global.params.useUnitTests) {
    IF_LOG Logger::println("No code generation for unit test declaration %s",
                           fd->toChars());
    fd->ir->setDefined();
    return;
  }

  // Skip array ops implemented in druntime
  if (fd->isArrayOp && !willInline() && isDruntimeArrayOp(fd)) {
    IF_LOG Logger::println(
        "No code generation for array op %s implemented in druntime",
        fd->toChars());
    fd->ir->setDefined();
    return;
  }

  if (!linkageAvailableExternally && !alreadyOrWillBeDefined(*fd)) {
    IF_LOG Logger::println("Skipping '%s'.", fd->toPrettyChars());
    fd->ir->setDefined();
    return;
  }

  DtoDeclareFunction(fd);
  assert(fd->ir->isDeclared());

  // DtoResolveFunction might also set the defined flag for functions we
  // should not touch.
  if (fd->ir->isDefined()) {
    return;
  }
  fd->ir->setDefined();

  // We cannot emit nested functions with parents that have not gone through
  // semantic analysis. This can happen as DMD leaks some template instances
  // from constraints into the module member list. DMD gets away with being
  // sloppy as functions in template contraints obviously never need to access
  // data from the template function itself, but it would still mess up our
  // nested context creation code.
  FuncDeclaration *parent = fd;
  while ((parent = getParentFunc(parent))) {
    if (parent->semanticRun != PASSsemantic3done || parent->semantic3Errors) {
      IF_LOG Logger::println(
          "Ignoring nested function with unanalyzed parent.");
      return;
    }
  }

  if (fd->needsClosure())
    verifyScopedDestructionInClosure(fd);

  assert(fd->ident != Id::empty);

  if (fd->semanticRun != PASSsemantic3done) {
    error(fd->loc, "Internal Compiler Error: function not fully analyzed; "
                   "previous unreported errors compiling %s?",
          fd->toPrettyChars());
    fatal();
  }

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

  // debug info
  irFunc->diSubprogram = gIR->DBuilder.EmitSubProgram(fd);

  if (!fd->fbody) {
    return;
  }

  IF_LOG Logger::println("Doing function body for: %s", fd->toChars());
  gIR->funcGenStates.emplace_back(new FuncGenState(*irFunc, *gIR));
  auto &funcGen = gIR->funcGen();
  SCOPE_EXIT {
    assert(&gIR->funcGen() == &funcGen);
    gIR->funcGenStates.pop_back();
  };

  const auto f = static_cast<TypeFunction *>(fd->type->toBasetype());
  IrFuncTy &irFty = irFunc->irFty;
  llvm::Function *func = irFunc->getLLVMFunc();

  const auto lwc = lowerFuncLinkage(fd);
  if (linkageAvailableExternally) {
    func->setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);
    func->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
    // Assert that we are not overriding a linkage type that disallows inlining
    assert(lwc.first != llvm::GlobalValue::WeakAnyLinkage &&
           lwc.first != llvm::GlobalValue::ExternalWeakLinkage &&
           lwc.first != llvm::GlobalValue::LinkOnceAnyLinkage);
  } else {
    setLinkage(lwc, func);
  }

  assert(!func->hasDLLImportStorageClass());

  // On x86_64, always set 'uwtable' for System V ABI compatibility.
  // TODO: Find a better place for this.
  if (global.params.targetTriple->getArch() == llvm::Triple::x86_64 &&
      !global.params.isWindows) {
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

  gIR->scopes.push_back(IRScope(beginbb));

// Set the FastMath options for this function scope.
#if LDC_LLVM_VER >= 308
  gIR->scopes.back().builder.setFastMathFlags(irFunc->FMF);
#else
  gIR->scopes.back().builder.SetFastMathFlags(irFunc->FMF);
#endif

  // create alloca point
  // this gets erased when the function is complete, so alignment etc does not
  // matter at all
  llvm::Instruction *allocaPoint = new llvm::AllocaInst(
      LLType::getInt32Ty(gIR->context()), "alloca point", beginbb);
  funcGen.allocapoint = allocaPoint;

  // debug info - after all allocas, but before any llvm.dbg.declare etc
  gIR->DBuilder.EmitFuncStart(fd);

  emitInstrumentationFnEnter(fd);

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

  // give the 'this' parameter (an lvalue) storage and debug info
  if (irFty.arg_this) {
    LLValue *thisvar = irFunc->thisArg;
    assert(thisvar);

    LLValue *thismem = thisvar;
    if (!irFty.arg_this->byref) {
      if (fd->interfaceVirtual) {
        // Adjust the 'this' pointer instead of using a thunk
        LLType *targetThisType = thismem->getType();
        thismem = DtoBitCast(thismem, getVoidPtrType());
        auto off = DtoConstInt(-fd->interfaceVirtual->offset);
        thismem = DtoGEP1(thismem, off, true);
        thismem = DtoBitCast(thismem, targetThisType);
      }
      thismem = DtoAllocaDump(thismem, 0, "this");
      irFunc->thisArg = thismem;
    }

    assert(getIrParameter(fd->vthis)->value == thisvar);
    getIrParameter(fd->vthis)->value = thismem;

    gIR->DBuilder.EmitLocalVariable(thismem, fd->vthis, nullptr, true);
  }

  // give the 'nestArg' parameter (an lvalue) storage
  if (irFty.arg_nest) {
    irFunc->nestArg = DtoAllocaDump(irFunc->nestArg, 0, "nestedFrame");
  }

  // define all explicit parameters
  if (fd->parameters)
    defineParameters(irFty, *fd->parameters);

  // Initialize PGO state for this function
  funcGen.pgo.assignRegionCounters(fd, func);

  DtoCreateNestedContext(funcGen);

  if (fd->vresult && !fd->vresult->nestedrefs.dim) // FIXME: not sure here :/
  {
    DtoVarDeclaration(fd->vresult);
  }

  // D varargs: prepare _argptr and _arguments
  if (f->linkage == LINKd && f->varargs == 1) {
    // allocate _argptr (of type core.stdc.stdarg.va_list)
    Type *const argptrType = Type::tvalist->semantic(fd->loc, fd->_scope);
    LLValue *argptrMem = DtoAlloca(argptrType, "_argptr_mem");
    irFunc->_argptr = argptrMem;

    // initialize _argptr with a call to the va_start intrinsic
    DLValue argptrVal(argptrType, argptrMem);
    LLValue *llAp = gABI->prepareVaStart(&argptrVal);
    llvm::CallInst::Create(GET_INTRINSIC_DECL(vastart), llAp, "",
                           gIR->scopebb());

    // copy _arguments to a memory location
    irFunc->_arguments =
        DtoAllocaDump(irFunc->_arguments, 0, "_arguments_mem");

    // Push cleanup block that calls va_end to match the va_start call.
    {
      auto *vaendBB =
          llvm::BasicBlock::Create(gIR->context(), "vaend", gIR->topfunc());
      IRScope saveScope = gIR->scope();
      gIR->scope() = IRScope(vaendBB);
      gIR->ir->CreateCall(GET_INTRINSIC_DECL(vaend), llAp);
      funcGen.scopes.pushCleanup(vaendBB, gIR->scopebb());
      gIR->scope() = saveScope;
    }
  }

  funcGen.pgo.emitCounterIncrement(fd->fbody);
  funcGen.pgo.setCurrentStmt(fd->fbody);

  // output function body
  Statement_toIR(fd->fbody, gIR);

  // D varargs: emit the cleanup block that calls va_end.
  if (f->linkage == LINKd && f->varargs == 1) {
    if (!gIR->scopereturned()) {
      if (!funcGen.retBlock)
        funcGen.retBlock = gIR->insertBB("return");
      funcGen.scopes.runCleanups(0, funcGen.retBlock);
      gIR->scope() = IRScope(funcGen.retBlock);
    }
    funcGen.scopes.popCleanups(0);
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

    emitInstrumentationFnLeave(fd);

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
    funcGen.allocapoint = nullptr;
    allocaPoint->eraseFromParent();
    allocaPoint = nullptr;
  }

  gIR->scopes.pop_back();
}

////////////////////////////////////////////////////////////////////////////////

DValue *DtoArgument(Parameter *fnarg, Expression *argexp) {
  IF_LOG Logger::println("DtoArgument");
  LOG_SCOPE;

  // ref/out arg
  if (fnarg && (fnarg->storageClass & (STCref | STCout))) {
    Loc loc;
    DValue *arg = toElem(argexp, true);
    return new DLValue(argexp->type,
                       arg->isLVal() ? DtoLVal(arg) : makeLValue(loc, arg));
  }

  DValue *arg = toElem(argexp);

  // lazy arg
  if (fnarg && (fnarg->storageClass & STClazy)) {
    assert(argexp->type->toBasetype()->ty == Tdelegate);
    assert(!arg->isLVal());
    return arg;
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
  const char *name = fd->ident->toChars();
  int i =
      binary(name, libArrayopFuncs, sizeof(libArrayopFuncs) / sizeof(char *));
  if (i != -1) {
    return 1;
  }

#ifdef DEBUG // Make sure our array is alphabetized
  for (size_t j = 0; j < sizeof(libArrayopFuncs) / sizeof(char *); j++) {
    if (strcmp(name, libArrayopFuncs[j]) == 0)
      assert(0);
  }
#endif
  return 0;
}
