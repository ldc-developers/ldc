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
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/init.h"
#include "dmd/ldcbindings.h"
#include "dmd/mangle.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "dmd/statement.h"
#include "dmd/template.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/cl_options_sanitizers.h"
#include "gen/abi.h"
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
#include "gen/uda.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/CFG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <iostream>

static bool isMainFunction(FuncDeclaration *fd) {
  return fd->isMain() || (global.params.betterC && fd->isCMain());
}

llvm::FunctionType *DtoFunctionType(Type *type, IrFuncTy &irFty, Type *thistype,
                                    Type *nesttype, FuncDeclaration *fd) {
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

  TargetABI *abi = fd && DtoIsIntrinsic(fd) ? TargetABI::getIntrinsic() : gABI;

  // Do not modify irFty yet; this function may be called recursively if any
  // of the argument types refer to this type.
  IrFuncTy newIrFty(f);

  // The index of the next argument on the LLVM level.
  unsigned nextLLArgIdx = 0;

  const bool isMain = fd && isMainFunction(fd);
  if (isMain) {
    // D and C main functions always return i32, even if declared as returning
    // void.
    newIrFty.ret = new IrFuncTyArg(Type::tint32, false);
  } else {
    Type *rt = f->next;
    const bool byref = f->isref && rt->toBasetype()->ty != Tvoid;
    llvm::AttrBuilder attrs;

    if (abi->returnInArg(f, fd && fd->needThis())) {
      // sret return
      llvm::AttrBuilder sretAttrs;
      sretAttrs.addAttribute(LLAttribute::StructRet);
      sretAttrs.addAttribute(LLAttribute::NoAlias);
      if (unsigned alignment = DtoAlignment(rt))
        sretAttrs.addAlignmentAttr(alignment);
      newIrFty.arg_sret = new IrFuncTyArg(rt, true, sretAttrs);
      rt = Type::tvoid;
      ++nextLLArgIdx;
    } else {
      // sext/zext return
      DtoAddExtendAttr(byref ? rt->pointerTo() : rt, attrs);
    }
    newIrFty.ret = new IrFuncTyArg(rt, byref, attrs);
  }
  ++nextLLArgIdx;

  if (thistype) {
    // Add the this pointer for member functions
    llvm::AttrBuilder attrs;
    attrs.addAttribute(LLAttribute::NonNull);
    if (fd && fd->isCtorDeclaration()) {
      attrs.addAttribute(LLAttribute::Returned);
    }
    newIrFty.arg_this =
        new IrFuncTyArg(thistype, thistype->toBasetype()->ty == Tstruct, attrs);
    ++nextLLArgIdx;
  } else if (nesttype) {
    // Add the context pointer for nested functions
    llvm::AttrBuilder attrs;
    attrs.addAttribute(LLAttribute::NonNull);
    newIrFty.arg_nest = new IrFuncTyArg(nesttype, false, attrs);
    ++nextLLArgIdx;
  }

  bool hasObjCSelector = false;
  if (fd && fd->linkage == LINKobjc && thistype) {
    if (fd->selector) {
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
  const bool isLLVMVariadic = (f->parameterList.varargs == VARARGvariadic);
  if (isLLVMVariadic && f->linkage == LINKd) {
    // Add extra `_arguments` parameter for D-style variadic functions.
    newIrFty.arg_arguments =
        new IrFuncTyArg(getTypeInfoType()->arrayOf(), false);
    ++nextLLArgIdx;
  }

  const size_t numExplicitDArgs = f->parameterList.length();

  // if this _Dmain() doesn't have an argument, we force it to have one
  if (isMain && f->linkage != LINKc && numExplicitDArgs == 0) {
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
    llvm::AttrBuilder attrs;
    if (arg->storageClass & STClazy) {
      // Lazy arguments are lowered to delegates.
      Logger::println("lazy param");
      auto ltf = TypeFunction::create(nullptr, arg->type, VARARGnone, LINKd);
      auto ltd = createTypeDelegate(ltf);
      loweredDType = ltd;
    } else if (passPointer) {
      // ref/out
      attrs.addDereferenceableAttr(loweredDType->size());
    } else {
      if (abi->passByVal(f, loweredDType)) {
        // LLVM ByVal parameters are pointers to a copy in the function
        // parameters stack. The caller needs to provide a pointer to the
        // original argument.
        attrs.addAttribute(LLAttribute::ByVal);
        if (auto alignment = DtoAlignment(loweredDType))
          attrs.addAlignmentAttr(alignment);
        passPointer = true;
      } else {
        // Add sext/zext as needed.
        DtoAddExtendAttr(loweredDType, attrs);
      }
    }

    newIrFty.args.push_back(new IrFuncTyArg(loweredDType, passPointer, attrs));
    newIrFty.args.back()->parametersIdx = i;
    ++nextLLArgIdx;
  }

  newIrFty.reverseParams = abi->reverseExplicitParams(f);

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
  // TODO: (correctly) apply these for NVPTX (but not for SPIRV).
  if (gIR->dcomputetarget && gIR->dcomputetarget->target == DComputeTarget::OpenCL)
    return;
  if (!gIR->dcomputetarget) {
    // Target CPU capabilities
    func.addFnAttr("target-cpu", target.getTargetCPU());
    auto featStr = target.getTargetFeatureString();
    if (!featStr.empty())
      func.addFnAttr("target-features", featStr);
  }
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

#if LDC_LLVM_VER >= 800
  switch (whichFramePointersToEmit()) {
    case llvm::FramePointer::None:
      func.addFnAttr("frame-pointer", "none");
      break;
    case llvm::FramePointer::NonLeaf:
      func.addFnAttr("frame-pointer", "non-leaf");
      break;
    case llvm::FramePointer::All:
      func.addFnAttr("frame-pointer", "all");
      break;
  }
#else
  func.addFnAttr("no-frame-pointer-elim",
                 willEliminateFramePointer() ? "false" : "true");
#endif
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

  // Calling convention.
  //
  // DMD treats _Dmain as having C calling convention and this has been
  // hardcoded into druntime, even if the frontend type has D linkage (Bugzilla
  // issue 9028).
  const bool forceC = vafunc || DtoIsIntrinsic(fdecl) || fdecl->isMain();
  const auto link = forceC ? LINKc : f->linkage;

  // mangled name
  const auto irMangle = getIRMangledName(fdecl, link);

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
  } else if (func->getFunctionType() != functype) {
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

  func->setCallingConv(gABI->callingConv(link, f, fdecl));

  if (global.params.isWindows && fdecl->isExport()) {
    func->setDLLStorageClass(fdecl->isImportedSymbol()
                                 ? LLGlobalValue::DLLImportStorageClass
                                 : LLGlobalValue::DLLExportStorageClass);
  }

  IF_LOG Logger::cout() << "func = " << *func << std::endl;

  // add func to IRFunc
  irFunc->setLLVMFunc(func);

  // First apply the TargetMachine attributes, such that they can be overridden
  // by UDAs.
  applyTargetMachineAttributes(*func, *gTargetMachine);
  applyFuncDeclUDAs(fdecl, irFunc);

  // parameter attributes
  if (!DtoIsIntrinsic(fdecl)) {
    applyParamAttrsToLLFunc(f, getIrFunc(fdecl)->irFty, func);
    if (global.params.disableRedZone) {
      func->addFnAttr(LLAttribute::NoRedZone);
    }
  }

  if(irFunc->isDynamicCompiled()) {
    declareDynamicCompiledFunction(gIR, irFunc);
  }

  if (irFunc->targetCpuOverridden ||
      irFunc->targetFeaturesOverridden) {
    gIR->targetCpuOrFeaturesOverridden.push_back(irFunc);
  }

  // main
  if (isMainFunction(fdecl) && fdecl->fbody) {
    // Detect multiple main function definitions, which is disallowed.
    // DMD checks this in the glue code, so we need to do it here as well.
    if (gIR->mainFunc) {
      error(fdecl->loc, "only one `main` function allowed");
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

  if (fdecl->isCrtCtorDtor & 1) {
    AppendFunctionToLLVMGlobalCtorsDtors(func, fdecl->priority, true);
  }
  if (fdecl->isCrtCtorDtor & 2) {
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
    size_t llExplicitIdx = irFty.reverseParams ? irFty.args.size() - k - 1 : k;
    ++k;
    IrFuncTyArg *arg = irFty.args[llExplicitIdx];

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
  for (VarDeclaration *v : fd->closureVars) {
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

  for (VarDeclaration *vd : parameters) {
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
    auto saveScope = irs.scope();
    irs.scope() = IRScope(traceEpilogBB);
    irs.ir->CreateCall(
        getRuntimeFunction(fd->endloc, irs.module, "_c_trace_epi"));
    funcGen.scopes.pushCleanup(traceEpilogBB, irs.scopebb());
    irs.scope() = saveScope;
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
        (func->getLinkage() == llvm::GlobalValue::AvailableExternallyLinkage)) {
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
    error(fd->loc,
          "Internal Compiler Error: function not fully analyzed; "
          "previous unreported errors compiling `%s`?",
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

  if (gIR->dcomputetarget) {
    auto id = fd->ident;
    if (id == Id::xopEquals || id == Id::xopCmp || id == Id::xtoHash) {
      IF_LOG Logger::println(
          "No code generation for typeinfo member %s in @compute code",
          fd->toChars());
      fd->ir->setDefined();
      return;
    }
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
    error(fd->loc,
          "Internal Compiler Error: function not fully analyzed; "
          "previous unreported errors compiling `%s`?",
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

  if (opts::defaultToHiddenVisibility && !fd->isExport()) {
    func->setVisibility(LLGlobalValue::HiddenVisibility);
  }

  // if this function is naked, we take over right away! no standard processing!
  if (fd->naked) {
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
  gIR->funcGenStates.emplace_back(new FuncGenState(*irFunc, *gIR));
  auto &funcGen = gIR->funcGen();
  SCOPE_EXIT {
    assert(&gIR->funcGen() == &funcGen);
    gIR->funcGenStates.pop_back();
  };

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
  }

  assert(!func->hasDLLImportStorageClass());

  // function attributes
  if (gABI->needsUnwindTables()) {
    func->addFnAttr(LLAttribute::UWTable);
  }
  if (opts::isAnySanitizerEnabled() &&
      !opts::functionIsInSanitizerBlacklist(fd)) {
    // Set the required sanitizer attribute.
    if (opts::isSanitizerEnabled(opts::AddressSanitizer)) {
      func->addFnAttr(LLAttribute::SanitizeAddress);
    }

    if (opts::isSanitizerEnabled(opts::MemorySanitizer)) {
      func->addFnAttr(LLAttribute::SanitizeMemory);
    }

    if (opts::isSanitizerEnabled(opts::ThreadSanitizer)) {
      func->addFnAttr(LLAttribute::SanitizeThread);
    }
  }
  applyXRayAttributes(*fd, *func);

  llvm::BasicBlock *beginbb =
      llvm::BasicBlock::Create(gIR->context(), "", func);

  gIR->scopes.push_back(IRScope(beginbb));
  SCOPE_EXIT {
    gIR->scopes.pop_back();
  };

  // Set the FastMath options for this function scope.
  gIR->scopes.back().builder.setFastMathFlags(irFunc->FMF);

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
#if LDC_LLVM_VER >= 500
                           0, // Address space
#endif
                           "alloca_point", beginbb);
  funcGen.allocapoint = allocaPoint;

  // debug info - after all allocas, but before any llvm.dbg.declare etc
  gIR->DBuilder.EmitFuncStart(fd);

  emitInstrumentationFnEnter(fd);

  if (global.params.trace && fd->emitInstrumentation && !fd->isCMain() &&
      !fd->naked) {
    emitDMDStyleFunctionTrace(*gIR, fd, funcGen);
  }

  // disable frame-pointer-elimination for functions with inline asm
  if (fd->hasReturnExp & 8) // has inline asm
  {
#if LDC_LLVM_VER >= 800
    func->addFnAttr(
        llvm::Attribute::get(gIR->context(), "frame-pointer", "all"));
#else
    func->addAttribute(
        LLAttributeSet::FunctionIndex,
        llvm::Attribute::get(gIR->context(), "no-frame-pointer-elim", "true"));
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
        thismem = DtoGEP1(thismem, off);
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
    Type *const argptrType = typeSemantic(Type::tvalist, fd->loc, fd->_scope);
    LLValue *argptrMem = DtoAlloca(argptrType, "_argptr_mem");
    irFunc->_argptr = argptrMem;

    // initialize _argptr with a call to the va_start intrinsic
    DLValue argptrVal(argptrType, argptrMem);
    LLValue *llAp = gABI->prepareVaStart(&argptrVal);
    llvm::CallInst::Create(GET_INTRINSIC_DECL(vastart), llAp, "",
                           gIR->scopebb());

    // copy _arguments to a memory location
    irFunc->_arguments = DtoAllocaDump(irFunc->_arguments, 0, "_arguments_mem");

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

  // Emit the cleanup blocks (e.g. va_end and function tracing)
  if (!funcGen.scopes.empty()) {
    if (!gIR->scopereturned()) {
      if (!funcGen.retBlock)
        funcGen.retBlock = gIR->insertBB("return");
      funcGen.scopes.runCleanups(0, funcGen.retBlock);
      gIR->scope() = IRScope(funcGen.retBlock);
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
    } else if (!gIR->isMainFunc(irFunc)) {
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

  if (gIR->dcomputetarget && hasKernelAttr(fd)) {
    auto fn = gIR->module.getFunction(fd->mangleString);
    gIR->dcomputetarget->addKernelMetadata(fd, fn);
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
