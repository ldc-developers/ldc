//===-- functions.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/functions.h"

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/init.h"
#include "dmd/mangle.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "dmd/statement.h"
#include "dmd/target.h"
#include "dmd/template.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/cl_options_sanitizers.h"
#include "driver/timetrace.h"
#include "gen/abi/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/dcompute/target.h"
#include "gen/dvalue.h"
#include "gen/dynamiccompile.h"
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
#include "gen/pgo_ASTbased.h"
#include "gen/pragma.h"
#include "gen/runtime.h"
#include "gen/scope_exit.h"
#include "gen/tollvm.h"
#include "gen/to_string.h"
#include "gen/uda.h"
#include "ir/irdsymbol.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/CFG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <iostream>

bool isAnyMainFunction(FuncDeclaration *fd) {
  return fd->isMain() || fd->isCMain();
}

llvm::FunctionType *DtoFunctionType(Type *type, IrFuncTy &irFty, Type *thistype,
                                    Type *nesttype, FuncDeclaration *fd) {
  IF_LOG Logger::println("DtoFunctionType(%s)", type->toChars());
  LOG_SCOPE

  // sanity check
  assert(type->ty == TY::Tfunction);
  TypeFunction *f = static_cast<TypeFunction *>(type);
  assert(f->next && "Encountered function type with invalid return type; "
                    "trying to codegen function ignored by the frontend?");

  // Return cached type if available
  if (irFty.funcType) {
    return irFty.funcType;
  }

  TargetABI *abi = fd && DtoIsIntrinsic(fd) ? TargetABI::getIntrinsic() : gABI;

  // Do not modify irFty yet; this function may be called recursively if any
  // of the argument types refer to this type.
  IrFuncTy newIrFty(f);

  // The index of the next argument on the LLVM level.
  unsigned nextLLArgIdx = 0;

  const bool isMain = fd && isAnyMainFunction(fd);
  if (isMain) {
    // D and C main functions always return i32, even if declared as returning
    // void.
    newIrFty.ret = new IrFuncTyArg(Type::tint32, false);
  } else {
    Type *rt = f->next;
    const bool byref = f->isref() && rt->toBasetype()->ty != TY::Tvoid;
#if LDC_LLVM_VER >= 1400
      llvm::AttrBuilder attrs(getGlobalContext());
#else
    llvm::AttrBuilder attrs;
#endif

    if (abi->returnInArg(f, fd && fd->needThis())) {
      // sret return
#if LDC_LLVM_VER >= 1400
      llvm::AttrBuilder sretAttrs(getGlobalContext());
#else
      llvm::AttrBuilder sretAttrs;
#endif
#if LDC_LLVM_VER >= 1200
      sretAttrs.addStructRetAttr(DtoType(rt));
#else
      sretAttrs.addAttribute(LLAttribute::StructRet);
#endif
      sretAttrs.addAttribute(LLAttribute::NoAlias);
      if (unsigned alignment = DtoAlignment(rt))
        sretAttrs.addAlignmentAttr(alignment);
      newIrFty.arg_sret = new IrFuncTyArg(rt, true, std::move(sretAttrs));
      rt = Type::tvoid;
      ++nextLLArgIdx;
    } else {
      // sext/zext return
      DtoAddExtendAttr(byref ? rt->pointerTo() : rt, attrs);
    }
    newIrFty.ret = new IrFuncTyArg(rt, byref, std::move(attrs));
  }
  ++nextLLArgIdx;

  if (thistype) {
    // Add the this pointer for member functions
#if LDC_LLVM_VER >= 1400
    llvm::AttrBuilder attrs(getGlobalContext());
#else
    llvm::AttrBuilder attrs;
#endif
    if (!opts::fNullPointerIsValid)
      attrs.addAttribute(LLAttribute::NonNull);
    if (fd && fd->isCtorDeclaration()) {
      attrs.addAttribute(LLAttribute::Returned);
    }
    newIrFty.arg_this = new IrFuncTyArg(
        thistype, thistype->toBasetype()->ty == TY::Tstruct, std::move(attrs));
    ++nextLLArgIdx;
  } else if (nesttype) {
    // Add the context pointer for nested functions
#if LDC_LLVM_VER >= 1400
    llvm::AttrBuilder attrs(getGlobalContext());
#else
    llvm::AttrBuilder attrs;
#endif
    if (!opts::fNullPointerIsValid)
      attrs.addAttribute(LLAttribute::NonNull);
    newIrFty.arg_nest = new IrFuncTyArg(nesttype, false, std::move(attrs));
    ++nextLLArgIdx;
  }

  bool hasObjCSelector = false;
  if (fd && fd->_linkage == LINK::objc && thistype) {
    if (fd->objc.selector) {
      hasObjCSelector = true;
    } else if (fd->parent->isClassDeclaration()) {
      fd->error("Objective-C `@selector` is missing");
    }
  }
  if (hasObjCSelector) {
    // TODO: make arg_objcselector to match dmd type
    newIrFty.arg_objcSelector = new IrFuncTyArg(Type::tvoidptr, false);
    ++nextLLArgIdx;
  }

  // Non-typesafe variadics (both C and D styles) are also variadics on the LLVM
  // level.
  const bool isLLVMVariadic = (f->parameterList.varargs == VARARGvariadic ||
                               f->parameterList.varargs == VARARGKRvariadic);
  if (isLLVMVariadic && f->linkage == LINK::d) {
    // Add extra `_arguments` parameter for D-style variadic functions.
    newIrFty.arg_arguments =
        new IrFuncTyArg(getTypeInfoType()->arrayOf(), false);
    ++nextLLArgIdx;
  }

  const size_t numExplicitDArgs = f->parameterList.length();

  // if this _Dmain() doesn't have an argument, we force it to have one
  if (isMain && f->linkage != LINK::c && numExplicitDArgs == 0) {
    Type *mainargs = Type::tchar->arrayOf()->arrayOf();
    newIrFty.args.push_back(new IrFuncTyArg(mainargs, false));
    ++nextLLArgIdx;
  }

  for (size_t i = 0; i < numExplicitDArgs; ++i) {
    Parameter *arg = Parameter::getNth(f->parameterList.parameters, i);

    // Whether the parameter is passed by LLVM value or as a pointer to the
    // alloca/….
    bool passPointer = arg->storageClass & (STCref | STCout);

    Type *loweredDType = arg->type;
#if LDC_LLVM_VER >= 1400
    llvm::AttrBuilder attrs(getGlobalContext());
#else
    llvm::AttrBuilder attrs;
#endif
    if (arg->storageClass & STClazy) {
      // Lazy arguments are lowered to delegates.
      Logger::println("lazy param");
      auto ltf = TypeFunction::create(nullptr, arg->type, VARARGnone, LINK::d);
      auto ltd = TypeDelegate::create(ltf);
      loweredDType = merge(ltd);
    } else if (passPointer) {
      // ref/out
      auto ts = loweredDType->toBasetype()->isTypeStruct();
      if (ts && !ts->sym->members) {
        // opaque struct
        if (!opts::fNullPointerIsValid)
          attrs.addAttribute(LLAttribute::NonNull);
        attrs.addAttribute(LLAttribute::NoUndef);
      } else {
        attrs.addDereferenceableAttr(loweredDType->size());
      }
    } else {
      if (abi->passByVal(f, loweredDType)) {
        // LLVM ByVal parameters are pointers to a copy in the function
        // parameters stack. The caller needs to provide a pointer to the
        // original argument.
#if LDC_LLVM_VER >= 1200
        attrs.addByValAttr(DtoType(loweredDType));
#else
        attrs.addAttribute(LLAttribute::ByVal);
#endif
        if (auto alignment = DtoAlignment(loweredDType))
          attrs.addAlignmentAttr(alignment);
        passPointer = true;
      } else {
        // Add sext/zext as needed.
        DtoAddExtendAttr(loweredDType, attrs);
      }
    }

    newIrFty.args.push_back(new IrFuncTyArg(loweredDType, passPointer, std::move(attrs)));
    newIrFty.args.back()->parametersIdx = i;
    ++nextLLArgIdx;
  }

  // let the ABI rewrite the types as necessary
  abi->rewriteFunctionType(newIrFty);

  // Now we can modify irFty safely.
  irFty = std::move(newIrFty);

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

  const size_t numExplicitLLArgs = irFty.args.size();
  for (size_t i = 0; i < numExplicitLLArgs; i++) {
    argtypes.push_back(irFty.args[i]->ltype);
  }

  irFty.funcType =
      LLFunctionType::get(irFty.ret->ltype, argtypes, isLLVMVariadic);

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
    (void)ad;
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
      fdecl->error("requires a dual-context, which is not yet supported by LDC");
      if (!global.gag)
        fatal();
      return LLFunctionType::get(LLType::getVoidTy(gIR->context()),
                                 /*isVarArg=*/false);
    }
  } else if (fdecl->isNested()) {
    dnest = Type::tvoid->pointerTo();
  }

  LLFunctionType *functype = DtoFunctionType(
      fdecl->type, getIrFunc(fdecl, true)->irFty, dthis, dnest, fdecl);

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

void DtoResolveFunction(FuncDeclaration *fdecl, const bool willDeclare) {
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
  if (type && type->ty == TY::Tfunction) {
    Type *next = static_cast<TypeFunction *>(type)->next;
    if (!next || next->ty == TY::Terror) {
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
                  // no sense
        }
        if (tempdecl->llvmInternal == LLVMva_start) {
          Logger::println("magic va_start found");
          fdecl->llvmInternal = LLVMva_start;
        } else if (tempdecl->llvmInternal == LLVMintrinsic) {
          Logger::println("overloaded intrinsic found");
          assert(fdecl->llvmInternal == LLVMintrinsic);
          assert(fdecl->mangleOverride.length);
        } else if (tempdecl->llvmInternal == LLVMinline_asm) {
          Logger::println("magic inline asm found");
          TypeFunction *tf = static_cast<TypeFunction *>(fdecl->type);
          if (tf->parameterList.varargs != VARARGvariadic ||
              (fdecl->parameters && fdecl->parameters->length != 0)) {
            tempdecl->error("invalid `__asm` declaration, must be a D style "
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
          fdecl->_linkage = LINK::c;
          Type *type = fdecl->type;
          assert(type->ty == TY::Tfunction);
          static_cast<TypeFunction *>(type)->linkage = LINK::c;

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
  if (!willDeclare && (!fdecl->isAbstract() || fdecl->fbody)) {
    DtoDeclareFunction(fdecl);
  }
}

void DtoResolveFunction(FuncDeclaration *fdecl) {
  return DtoResolveFunction(fdecl, false);
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
  const auto dcompute = gIR->dcomputetarget;

  // TODO: (correctly) apply these for NVPTX (but not for SPIRV).
  if (dcompute && dcompute->target == DComputeTarget::ID::OpenCL)
    return;
  const auto cpu = dcompute ? "" : target.getTargetCPU();
  const auto features = dcompute ? "" : target.getTargetFeatureString();

  opts::setFunctionAttributes(cpu, features, func);
  if (opts::fFastMath) // -ffast-math[=true] overrides -enable-unsafe-fp-math
    func.addFnAttr("unsafe-fp-math", "true");
  if (!func.hasFnAttribute("frame-pointer")) // not explicitly set by user
    func.addFnAttr("frame-pointer", isOptimizationEnabled() ? "none" : "all");
}

void applyXRayAttributes(FuncDeclaration &fdecl, llvm::Function &func) {
  if (!opts::fXRayInstrument)
    return;

  if (!fdecl.emitInstrumentation) {
    func.addFnAttr("function-instrument", "xray-never");
  } else {
    func.addFnAttr("xray-instruction-threshold",
                   opts::getXRayInstructionThresholdString());
  }
}

void onlyOneMainCheck(FuncDeclaration *fd) {
  if (!fd->fbody) // multiple *declarations* are fine
    return;

  // We'd actually want all possible main functions to be mutually exclusive.
  // Unfortunately, a D main implies a C main, so only check C mains with
  // -betterC.
  const bool isOSWindows = global.params.targetTriple->isOSWindows();
  if (fd->isMain() || (global.params.betterC && fd->isCMain()) ||
      (isOSWindows && (fd->isWinMain() || fd->isDllMain()))) {
    // global - across all modules compiled in this compiler invocation
    static Loc mainLoc;
    if (!mainLoc.filename()) {
      mainLoc = fd->loc;
      assert(mainLoc.filename());
    } else {
      const char *otherMainNames =
          isOSWindows ? ", `WinMain`, or `DllMain`" : "";
      const char *mainSwitch =
          global.params.addMain ? ", -main switch added another `main()`" : "";
      error(fd->loc,
            "only one `main`%s allowed%s. Previously found `main` at %s",
            otherMainNames, mainSwitch, mainLoc.toChars());
    }
  }
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

void DtoDeclareFunction(FuncDeclaration *fdecl, const bool willDefine) {
  DtoResolveFunction(fdecl, /*willDeclare=*/true);

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
  bool defineAtEnd = false;
  bool defineAsAvailableExternally = false;
  if (willDefine) {
    // will be defined anyway after declaration
  } else if (defineOnDeclare(fdecl, /*isFunction=*/true)) {
    Logger::println("Function is inside a linkonce_odr template, will be "
                    "defined after declaration.");
    if (fdecl->semanticRun < PASS::semantic3done) {
      Logger::println("Function hasn't had sema3 run yet, running it now.");
      const bool semaSuccess = fdecl->functionSemantic3();
      (void)semaSuccess;
      assert(semaSuccess);
      Module::runDeferredSemantic3();
    }
    defineAtEnd = true;
  } else if (defineAsExternallyAvailable(*fdecl)) {
    Logger::println("Function is an externally_available inline candidate, "
                    "will be defined after declaration.");
    defineAtEnd = true;
    defineAsAvailableExternally = true;
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

  // Calling convention.
  //
  // DMD treats _Dmain as having C calling convention and this has been
  // hardcoded into druntime, even if the frontend type has D linkage (Bugzilla
  // issue 9028).
  const bool forceC = vafunc || DtoIsIntrinsic(fdecl) || fdecl->isMain();

  // mangled name
  const auto irMangle = getIRMangledName(fdecl, forceC ? LINK::c : f->linkage);

  // construct function
  LLFunctionType *functype = DtoFunctionType(fdecl);
  LLFunction *func = vafunc ? vafunc : gIR->module.getFunction(irMangle);
  if (!func) {
    // All function declarations are "external" - any other linkage type
    // is set when actually defining the function, except extern_weak.
    auto linkage = llvm::GlobalValue::ExternalLinkage;
    // Apply pragma(LDC_extern_weak)
    if (fdecl->llvmInternal == LLVMextern_weak)
      linkage = llvm::GlobalValue::ExternalWeakLinkage;
    func = LLFunction::Create(functype, linkage, irMangle, &gIR->module);
  } else if (func->getFunctionType() == functype) {
    // IR signature matches existing function
  } else if (fdecl->isCsymbol() &&
             func->getFunctionType() ==
                 LLFunctionType::get(functype->getReturnType(),
                                     functype->params(), false)) {
    // ImportC: a variadic definition replaces a non-variadic declaration; keep
    // existing non-variadic IR function
    assert(func->isDeclaration());
  } else {
    const auto existingTypeString = llvmTypeToString(func->getFunctionType());
    const auto newTypeString = llvmTypeToString(functype);
    error(fdecl->loc,
          "Function type does not match previously declared "
          "function with the same mangled name: `%s`",
          mangleExact(fdecl));
    errorSupplemental(fdecl->loc, "Previous IR type: %s",
                      existingTypeString.c_str());
    errorSupplemental(fdecl->loc, "New IR type:      %s",
                      newTypeString.c_str());
    fatal();
  }

  func->setCallingConv(forceC ? gABI->callingConv(LINK::c)
                              : getCallingConvention(fdecl));

  IF_LOG Logger::cout() << "func = " << *func << std::endl;

  // add func to IRFunc
  irFunc->setLLVMFunc(func);

  // First apply the TargetMachine attributes and NonLazyBind attribute,
  // such that they can be overridden by UDAs.
  applyTargetMachineAttributes(*func, *gTargetMachine);
  if (!fdecl->fbody && opts::noPLT) {
    // Add `NonLazyBind` attribute to function declarations,
    // the codegen options allow skipping PLT.
    func->addFnAttr(LLAttribute::NonLazyBind);
  }
  if (f->next->toBasetype()->ty == TY::Tnoreturn) {
    func->addFnAttr(LLAttribute::NoReturn);
  }
#if LDC_LLVM_VER >= 1300
  if (opts::fWarnStackSize.getNumOccurrences() > 0 &&
      opts::fWarnStackSize < UINT_MAX) {
    // Cache the int->string conversion result.
    static std::string thresholdString = ldc::to_string(opts::fWarnStackSize);

    func->addFnAttr("warn-stack-size", thresholdString);
  }
#endif

  applyFuncDeclUDAs(fdecl, irFunc);

  // parameter attributes
  if (!DtoIsIntrinsic(fdecl)) {
    applyParamAttrsToLLFunc(f, getIrFunc(fdecl)->irFty, func);
    if (global.params.disableRedZone) {
      func->addFnAttr(LLAttribute::NoRedZone);
    }
  }

  if (irFunc->isDynamicCompiled()) {
    declareDynamicCompiledFunction(gIR, irFunc);
  }

  if (irFunc->targetCpuOverridden || irFunc->targetFeaturesOverridden) {
    gIR->targetCpuOrFeaturesOverridden.push_back(irFunc);
  }

  // Detect multiple main function definitions, which is disallowed.
  // DMD checks this in the glue code, so we need to do it here as well.
  onlyOneMainCheck(fdecl);

  // Set inlining attribute
  if (fdecl->neverInline) {
    irFunc->setNeverInline();
  } else {
    if (fdecl->inlining == PINLINE::always) {
      irFunc->setAlwaysInline();
    } else if (fdecl->inlining == PINLINE::never) {
      irFunc->setNeverInline();
    }
  }

  if (fdecl->isCrtCtor()) {
    AppendFunctionToLLVMGlobalCtorsDtors(func, fdecl->priority, true);
  }
  if (fdecl->isCrtDtor()) {
    AppendFunctionToLLVMGlobalCtorsDtors(func, fdecl->priority, false);
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

  if (passThisBeforeSret) {
    iarg->setName(".sret_arg");
    irFunc->sretArg = &(*iarg);
    ++iarg;
  }

  if (irFty.arg_objcSelector) {
    iarg->setName(".objcSelector_arg");
    ++iarg;
  }

  if (irFty.arg_arguments) {
    iarg->setName("._arguments");
    irFunc->_arguments = &(*iarg);
    ++iarg;
  }

  unsigned int k = 0;
  for (; iarg != func->arg_end(); ++iarg) {
    IrFuncTyArg *arg = irFty.args[k++];

    if (!fdecl->parameters || arg->parametersIdx >= fdecl->parameters->length) {
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
    IF_LOG Logger::println("Define function after declaration:");
    DtoDefineFunction(fdecl, defineAsAvailableExternally);
  }
}

void DtoDeclareFunction(FuncDeclaration *fdecl) {
  return DtoDeclareFunction(fdecl, false);
}

////////////////////////////////////////////////////////////////////////////////

static LinkageWithCOMDAT lowerFuncLinkage(FuncDeclaration *fdecl) {
  // Intrinsics are always external.
  if (DtoIsIntrinsic(fdecl)) {
    return LinkageWithCOMDAT(LLGlobalValue::ExternalLinkage, false);
  }

  // A body-less declaration always needs to be marked as external in LLVM
  // (also e.g. naked template functions which would otherwise be weak_odr,
  // but where the definition is in module-level inline asm).
  if (!fdecl->fbody || fdecl->isNaked()) {
    return LinkageWithCOMDAT(LLGlobalValue::ExternalLinkage, false);
  }

  return DtoLinkage(fdecl);
}

// LDC has the same problem with destructors of struct arguments in closures
// as DMD, so we copy the failure detection
void verifyScopedDestructionInClosure(FuncDeclaration *fd) {
  for (VarDeclaration *v : fd->closureVars) {
    // Hack for the case fail_compilation/fail10666.d, until
    // proper issue https://issues.dlang.org/show_bug.cgi?id=5730 fix will come.
    bool isScopeDtorParam = v->edtor && (v->storage_class & STCparameter);
    if (v->needsScopeDtor() || isScopeDtorParam) {
      // Because the value needs to survive the end of the scope!
      v->error("has scoped destruction, cannot build closure");
    }
    if (v->isargptr()) {
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

  for (VarDeclaration *vd : parameters) {
    Type *paramType = vd->type;
    IrParameter *irparam = getIrParameter(vd);

    if (!irparam) {
      // This is a parameter that is not passed on the LLVM level.
      // Create the param here and set it to a "dummy" alloca that
      // we do not store to here.
      irparam = getIrParameter(vd, true);
      irparam->value = DtoAlloca(vd, vd->ident->toChars());
    } else if (!irparam->value) {
      // Captured parameter not passed on the LLVM level.
      assert(irparam->nestedIndex >= 0);
      irparam->value = DtoAlloca(vd, vd->ident->toChars());
    } else {
      // vd->type (parameter) and irparam->arg->type (argument) don't always
      // match. E.g., for a lazy parameter of type T, vd->type is T (with lazy
      // storage class) while irparam->arg->type is the delegate type.
      paramType = irparam->arg->type;

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

    // The debuginfos for captured params are handled later by
    // DtoCreateNestedContext().
    if (global.params.symdebug && vd->nestedrefs.length == 0) {
      // Reference (ref/out) parameters have no storage themselves as they are
      // constant pointers, so pass the reference rvalue to EmitLocalVariable().
      gIR->DBuilder.EmitLocalVariable(irparam->value, vd, paramType, false,
                                      false, /*isRefRVal=*/true);
    }
  }
}

void emitDMDStyleFunctionTrace(IRState &irs, FuncDeclaration *fd,
                               FuncGenState &funcGen) {
  /* DMD-style profiling: wrap the entire function body in:
   *   trace_pro("funcname");
   *   try
   *     body;
   *   finally
   *     _c_trace_epi();
   */

  // Call trace_pro("funcname")
  {
    auto fn = getRuntimeFunction(fd->loc, irs.module, "trace_pro");
    auto funcname = DtoConstString(mangleExact(fd));
    irs.ir->CreateCall(fn, {funcname});
  }

  // Push cleanup block that calls _c_trace_epi at function exit.
  {
    auto traceEpilogBB = irs.insertBB("trace_epi");
    const auto savedInsertPoint = irs.saveInsertPoint();
    irs.ir->SetInsertPoint(traceEpilogBB);
    irs.ir->CreateCall(
        getRuntimeFunction(fd->endloc, irs.module, "_c_trace_epi"));
    funcGen.scopes.pushCleanup(traceEpilogBB, irs.scopebb());
  }
}

// If the specified block is trivially unreachable, erases it and returns true.
// This is a common case because it happens when 'return' is the last statement
// in a function.
bool eraseDummyAfterReturnBB(llvm::BasicBlock *bb) {
  if (pred_begin(bb) == pred_end(bb) &&
      bb != &bb->getParent()->getEntryBlock()) {
    bb->eraseFromParent();
    return true;
  }
  return false;
}

/**
 * LLVM doesn't really support weak linkage for MSVC targets, it just prevents
 * inlining. We can emulate it though, by renaming the defined function, only
 * declaring the original function and embedding a linker directive in the
 * object file, instructing the linker to fall back to the weak implementation
 * if there's no strong definition.
 * The object file still needs to be pulled in by the linker for the directive
 * to be found.
 */
void emulateWeakAnyLinkageForMSVC(IrFunction *irFunc, LINK linkage) {
  LLFunction *func = irFunc->getLLVMFunc();

  const bool isWin32 = global.params.targetTriple->isArch32Bit();

  std::string mangleBuffer;
  llvm::StringRef finalMangle = func->getName();
  if (finalMangle[0] == '\1') {
    finalMangle = finalMangle.substr(1);
  } else if (isWin32) {
    // implicit underscore prefix for Win32
    mangleBuffer = ("_" + finalMangle).str();
    finalMangle = mangleBuffer;
  }

  std::string finalWeakMangle = finalMangle.str();
  if (linkage == LINK::cpp) {
    assert(finalMangle.startswith("?"));
    // prepend `__weak_` to first identifier
    size_t offset = finalMangle.startswith("??$") ? 3 : 1;
    finalWeakMangle.insert(offset, "__weak_");
  } else if (linkage == LINK::d) {
    const size_t offset = isWin32 ? 1 : 0;
    assert(finalMangle.substr(offset).startswith("_D"));
    // prepend a `__weak` package
    finalWeakMangle.insert(offset + 2, "6__weak");
  } else {
    // prepend `__weak_`
    const size_t offset = isWin32 && finalMangle.startswith("_") ? 1 : 0;
    finalWeakMangle.insert(offset, "__weak_");
  }

  const std::string linkerOption =
      ("/ALTERNATENAME:" + finalMangle + "=" + finalWeakMangle).str();
  gIR->addLinkerOption(llvm::StringRef(linkerOption));

  // rename existing function
  const std::string oldName = func->getName().str();
  func->setName("\1" + finalWeakMangle);
  if (func->hasComdat()) {
    func->setComdat(gIR->module.getOrInsertComdat(func->getName()));
  }

  // create a new body-less declaration with the old name
  auto newFunc =
      LLFunction::Create(func->getFunctionType(),
                         LLGlobalValue::ExternalLinkage, oldName, &gIR->module);

  // replace existing and future uses of the old, renamed function with the new
  // declaration
  irFunc->setLLVMFunc(newFunc);
  func->replaceNonMetadataUsesWith(newFunc);
}

} // anonymous namespace

void DtoDefineFunction(FuncDeclaration *fd, bool linkageAvailableExternally) {
  TimeTraceScope timeScope([fd]() {
                             std::string name("Codegen func ");
                             name += fd->toChars();
                             return name;
                           },
                           [fd]() {
                             std::string detail = fd->toPrettyChars();
                             return detail;
                           },
                           fd->loc);

  IF_LOG Logger::println("DtoDefineFunction(%s): %s", fd->toPrettyChars(),
                         fd->loc.toChars());
  LOG_SCOPE;
  if (linkageAvailableExternally) {
    IF_LOG Logger::println("linkageAvailableExternally = true");
  }

  if (fd->ir->isDefined()) {
    llvm::Function *func = getIrFunc(fd)->getLLVMFunc();
    assert(func);
    if (!linkageAvailableExternally &&
        (func->getLinkage() == llvm::GlobalValue::AvailableExternallyLinkage)) {
      // Fix linkage and visibility
      const auto lwc = lowerFuncLinkage(fd);
      setLinkage(lwc, func);
      setVisibility(fd, func);
    }
    return;
  }

  if ((fd->type && fd->type->ty == TY::Terror) ||
      (fd->type && fd->type->ty == TY::Tfunction &&
       static_cast<TypeFunction *>(fd->type)->next == nullptr) ||
      (fd->type && fd->type->ty == TY::Tfunction &&
       static_cast<TypeFunction *>(fd->type)->next->ty == TY::Terror)) {
    IF_LOG Logger::println(
        "Ignoring; has error type, no return type or returns error type");
    fd->ir->setDefined();
    return;
  }

  if (fd->semanticRun == PASS::semanticdone) {
    // This function failed semantic3() with errors but the errors were gagged.
    // In contrast to DMD we immediately bail out here, since other parts of
    // the codegen expect irFunc to be set for defined functions.
    error(fd->loc,
          "Internal Compiler Error: function not fully analyzed; "
          "previous unreported errors compiling `%s`?",
          fd->toPrettyChars());
    fatal();
  }

  DtoDeclareFunction(fd, /*willDefine=*/true);
  assert(fd->ir->isDeclared());

  // DtoDeclareFunction might also set the defined flag for functions we
  // should not touch.
  if (fd->ir->isDefined()) {
    return;
  }
  fd->ir->setDefined();

  if (fd->isUnitTestDeclaration() && !global.params.useUnitTests) {
    IF_LOG Logger::println("No code generation for unit test declaration %s",
                           fd->toChars());
    return;
  }

  if (gIR->dcomputetarget) {
    auto id = fd->ident;
    if (id == Id::xopEquals || id == Id::xopCmp || id == Id::xtoHash) {
      IF_LOG Logger::println(
          "No code generation for typeinfo member %s in @compute code",
          fd->toChars());
      return;
    }
  }

  if (!linkageAvailableExternally && skipCodegen(*fd)) {
    IF_LOG Logger::println("Skipping '%s'.", fd->toPrettyChars());
    return;
  }

  // We cannot emit nested functions with parents that have not gone through
  // semantic analysis. This can happen as DMD leaks some template instances
  // from constraints into the module member list. DMD gets away with being
  // sloppy as functions in template contraints obviously never need to access
  // data from the template function itself, but it would still mess up our
  // nested context creation code.
  FuncDeclaration *parent = fd;
  while ((parent = getParentFunc(parent))) {
    if (parent->semanticRun != PASS::semantic3done ||
        parent->hasSemantic3Errors()) {
      IF_LOG Logger::println(
          "Ignoring nested function with unanalyzed parent.");
      return;
    }
  }

  if (fd->needsClosure())
    verifyScopedDestructionInClosure(fd);

  assert(fd->ident != Id::empty);

  if (fd->semanticRun != PASS::semantic3done) {
    error(fd->loc,
          "Internal Compiler Error: function not fully analyzed; "
          "previous unreported errors compiling `%s`?",
          fd->toPrettyChars());
    fatal();
  }

  if (fd->isUnitTestDeclaration()) {
    // ignore unparsed unittests from non-root modules
    if (fd->fbody)
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

  if (!fd->fbody) {
    return;
  }

  IrFunction *const irFunc = getIrFunc(fd);
  llvm::Function *const func = irFunc->getLLVMFunc();

  if (!func->empty()) {
    warning(fd->loc,
            "skipping definition of function `%s` due to previous definition "
            "for the same mangled name: %s",
            fd->toPrettyChars(), mangleExact(fd));
    return;
  }

  gIR->funcGenStates.emplace_back(new FuncGenState(*irFunc, *gIR));
  auto &funcGen = gIR->funcGen();
  SCOPE_EXIT {
    assert(&gIR->funcGen() == &funcGen);
    gIR->funcGenStates.pop_back();
  };

  // if this function is naked, we take over right away! no standard processing!
  if (fd->isNaked()) {
    DtoDefineNakedFunction(fd);
    return;
  }

  SCOPE_EXIT {
    if (irFunc->isDynamicCompiled()) {
      defineDynamicCompiledFunction(gIR, irFunc);
    }
  };

  // debug info
  gIR->DBuilder.EmitSubProgram(fd);

  IF_LOG Logger::println("Doing function body for: %s", fd->toChars());

  const auto f = static_cast<TypeFunction *>(fd->type->toBasetype());
  IrFuncTy &irFty = irFunc->irFty;

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
    setVisibility(fd, func);
  }

  // function attributes
  if (gABI->needsUnwindTables()) {
#if LDC_LLVM_VER >= 1500
    func->setUWTableKind(llvm::UWTableKind::Default);
#else
    func->addFnAttr(LLAttribute::UWTable);
#endif
  }
  if (opts::isAnySanitizerEnabled() &&
      !opts::functionIsInSanitizerBlacklist(fd)) {
    // Get the @noSanitize mask
    auto noSanitizeMask = getMaskFromNoSanitizeUDA(*fd);

    // Set the required sanitizer attribute.
    if (opts::isSanitizerEnabled(opts::AddressSanitizer & noSanitizeMask)) {
      func->addFnAttr(LLAttribute::SanitizeAddress);
    }

    if (opts::isSanitizerEnabled(opts::MemorySanitizer & noSanitizeMask)) {
      func->addFnAttr(LLAttribute::SanitizeMemory);
    }

    if (opts::isSanitizerEnabled(opts::ThreadSanitizer & noSanitizeMask)) {
      func->addFnAttr(LLAttribute::SanitizeThread);
    }
  }
  applyXRayAttributes(*fd, *func);
  if (opts::fNullPointerIsValid) {
    func->addFnAttr(LLAttribute::NullPointerIsValid);
  }
  if (opts::fSplitStack && !hasNoSplitStackUDA(fd)) {
    func->addFnAttr("split-stack");
  }

  llvm::BasicBlock *beginbb =
      llvm::BasicBlock::Create(gIR->context(), "", func);

  // set up the IRBuilder scope for the function
  const auto savedIRBuilderScope = gIR->setInsertPoint(beginbb);
  gIR->ir->setFastMathFlags(irFunc->FMF);
  gIR->DBuilder.EmitFuncStart(fd);

  // @naked: emit body and return, no prologue/epilogue
  if (func->hasFnAttribute(llvm::Attribute::Naked)) {
    Statement_toIR(fd->fbody, gIR);
    const bool wasDummy = eraseDummyAfterReturnBB(gIR->scopebb());
    if (!wasDummy && !gIR->scopereturned()) {
      // this is what clang does to prevent LLVM complaining about
      // non-terminated function
      gIR->ir->CreateUnreachable();
    }
    return;
  }

  // create alloca point
  // this gets erased when the function is complete, so alignment etc does not
  // matter at all
  llvm::Instruction *allocaPoint =
      new llvm::AllocaInst(LLType::getInt32Ty(gIR->context()),
                           0, // Address space
                           "alloca_point", beginbb);
  funcGen.allocapoint = allocaPoint;

  emitInstrumentationFnEnter(fd);

  if (global.params.trace && fd->emitInstrumentation && !fd->isCMain() &&
      !fd->isNaked()) {
    emitDMDStyleFunctionTrace(*gIR, fd, funcGen);
  }

  // disable frame-pointer-elimination for functions with DMD-style inline asm
  if (fd->hasReturnExp & 32) {
    func->addFnAttr(
        llvm::Attribute::get(gIR->context(), "frame-pointer", "all"));
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
        thismem = DtoGEP1(llvm::Type::getInt8Ty(gIR->context()), thismem, off);
        thismem = DtoBitCast(thismem, targetThisType);
      }
      thismem = DtoAllocaDump(thismem, 0, "this");
      irFunc->thisArg = thismem;
    }

    assert(getIrParameter(fd->vthis)->value == thisvar);
    getIrParameter(fd->vthis)->value = thismem;

    gIR->DBuilder.EmitLocalVariable(thismem, fd->vthis, nullptr, true);
  }

  // define all explicit parameters
  if (fd->parameters)
    defineParameters(irFty, *fd->parameters);

  // Initialize PGO state for this function
  funcGen.pgo.assignRegionCounters(fd, func);

  DtoCreateNestedContext(funcGen);

  // Declare the special __result variable. If it's captured, it has already
  // been allocated by DtoCreateNestedContext().
  if (fd->vresult) {
    DtoVarDeclaration(fd->vresult);
  }

  // D varargs: prepare _argptr and _arguments
  if (f->isDstyleVariadic()) {
    // allocate _argptr (of type core.stdc.stdarg.va_list)
    Type *tvalist = target.va_listType(fd->loc, fd->_scope);
    LLValue *argptrMem = DtoAlloca(tvalist, "_argptr_mem");
    irFunc->_argptr = argptrMem;

    // initialize _argptr with a call to the va_start intrinsic
    DLValue argptrVal(tvalist, argptrMem);
    LLValue *llAp = gABI->prepareVaStart(&argptrVal);
    llvm::CallInst::Create(GET_INTRINSIC_DECL(vastart), llAp, "",
                           gIR->scopebb());

    // copy _arguments to a memory location
    irFunc->_arguments = DtoAllocaDump(irFunc->_arguments, 0, "_arguments_mem");

    // Push cleanup block that calls va_end to match the va_start call.
    {
      auto *vaendBB = llvm::BasicBlock::Create(gIR->context(), "vaend", func);
      const auto savedInsertPoint = gIR->saveInsertPoint();
      gIR->ir->SetInsertPoint(vaendBB);
      gIR->ir->CreateCall(GET_INTRINSIC_DECL(vaend), llAp);
      funcGen.scopes.pushCleanup(vaendBB, gIR->scopebb());
    }
  }

  funcGen.pgo.emitCounterIncrement(fd->fbody);
  funcGen.pgo.setCurrentStmt(fd->fbody);

  // output function body
  Statement_toIR(fd->fbody, gIR);

  // Emit the cleanup blocks (e.g. va_end and function tracing)
  if (!funcGen.scopes.empty()) {
    if (!gIR->scopereturned()) {
      if (!funcGen.retBlock)
        funcGen.retBlock = gIR->insertBB("return");
      funcGen.scopes.runCleanups(0, funcGen.retBlock);
      gIR->ir->SetInsertPoint(funcGen.retBlock);
    }
    funcGen.scopes.popCleanups(0);
  }

  const bool wasDummy = eraseDummyAfterReturnBB(gIR->scopebb());
  if (!wasDummy && !gIR->scopereturned()) {
    // llvm requires all basic blocks to end with a TerminatorInst but DMD does
    // not put a return statement in automatically, so we do it here.

    emitInstrumentationFnLeave(fd);

    // pass the previous block into this block
    gIR->DBuilder.EmitStopPoint(fd->endloc);
    if (func->getReturnType() == LLType::getVoidTy(gIR->context())) {
      gIR->ir->CreateRetVoid();
    } else if (isAnyMainFunction(fd)) {
      gIR->ir->CreateRet(LLConstant::getNullValue(func->getReturnType()));
    } else if (auto asmb = fd->fbody->endsWithAsm()) {
      assert(asmb->abiret);
      gIR->ir->CreateRet(asmb->abiret);
    } else {
      gIR->ir->CreateRet(llvm::UndefValue::get(func->getReturnType()));
    }
  }

  // erase alloca point
  if (allocaPoint->getParent()) {
    funcGen.allocapoint = nullptr;
    allocaPoint->eraseFromParent();
    allocaPoint = nullptr;
  }

  if (gIR->dcomputetarget && hasKernelAttr(fd)) {
    auto fn = gIR->module.getFunction(fd->mangleString);
    gIR->dcomputetarget->addKernelMetadata(fd, fn);
  }

  if (func->getLinkage() == LLGlobalValue::WeakAnyLinkage &&
      !func->hasDLLExportStorageClass() &&
      global.params.targetTriple->isWindowsMSVCEnvironment()) {
    emulateWeakAnyLinkageForMSVC(irFunc, fd->resolvedLinkage());
  }
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
    assert(argexp->type->toBasetype()->ty == TY::Tdelegate);
    assert(!arg->isLVal());
    return arg;
  }

  return arg;
}

////////////////////////////////////////////////////////////////////////////////

/* Gives precedence to user-specified calling convention using ldc.attributes.callingConvention,
 * before querying the ABI.
 */
llvm::CallingConv::ID getCallingConvention(FuncDeclaration *fdecl)
{
  llvm::CallingConv::ID retval;

  // First check if there is an override by a UDA
  // If callconv is MaxID-1, then the "default" calling convention is specified. Behave as if no UDA was specified at all.
  bool userOverride = hasCallingConventionUDA(fdecl, &retval);
  if (userOverride && (retval != llvm::CallingConv::MaxID - 1))
    return retval;

  return gABI->callingConv(fdecl);
}
