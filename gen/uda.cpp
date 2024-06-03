#include "gen/uda.h"

#include "dmd/aggregate.h"
#include "dmd/attrib.h"
#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/module.h"
#include "driver/cl_options_sanitizers.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "ir/irfunction.h"
#include "ir/irvar.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

using namespace dmd;

namespace {

/// Checks whether `moduleDecl` is in the ldc package and it's identifier is
/// `id`.
bool isMagicModule(const ModuleDeclaration *moduleDecl, const Identifier *id) {
  if (!moduleDecl)
    return false;

  if (moduleDecl->id != id) {
    return false;
  }

  if (moduleDecl->packages.length != 1 ||
      moduleDecl->packages.ptr[0] != Id::ldc) {
    return false;
  }
  return true;
}

/// Checks whether the type of `e` is a struct from an ldc recognised module,
/// i.e. ldc.attributes or ldc.dcompute.
bool isFromMagicModule(const StructLiteralExp *e, const Identifier *id) {
  auto moduleDecl = e->sd->getModule()->md;
  return isMagicModule(moduleDecl, id);
}

StructLiteralExp *getLdcAttributesStruct(Expression *attr) {
  // See whether we can evaluate the attribute at compile-time. All the LDC
  // attributes are struct literals that may be constructed using a CTFE
  // function.
  unsigned prevErrors = global.startGagging();
  auto e = ctfeInterpret(attr);
  if (global.endGagging(prevErrors)) {
    return nullptr;
  }

  if (auto sle = e->isStructLiteralExp()) {
    if (isFromMagicModule(sle, Id::attributes))
      return sle;
  }

  return nullptr;
}

void checkStructElems(StructLiteralExp *sle, ArrayParam<Type *> elemTypes) {
  if (sle->elements->length != elemTypes.size()) {
    error(sle->loc,
          "unexpected field count in `%s`; does druntime not match compiler "
          "version?",
          sle->sd->toPrettyChars());
    fatal();
  }

  for (size_t i = 0; i < sle->elements->length; ++i) {
    if ((*sle->elements)[i]->type->toBasetype() != elemTypes[i]) {
      error(sle->loc,
            "invalid field type in `%s`; does druntime not match compiler "
            "version?",
            sle->sd->toPrettyChars());
      fatal();
    }
  }
}

/// Returns the StructLiteralExp magic attribute with identifier `id` from
/// the ldc magic module with identifier `from` (attributes or dcompute)
/// if it is applied to `sym`, otherwise returns nullptr.
StructLiteralExp *getMagicAttribute(Dsymbol *sym, const Identifier *id,
                                    const Identifier *from) {
  if (!sym->userAttribDecl())
    return nullptr;

  // Loop over all UDAs and early return the expression if a match was found.
  Expressions *attrs = getAttributes(sym->userAttribDecl());
  expandTuples(attrs);
  for (auto attr : *attrs) {
    if (auto sle = attr->isStructLiteralExp())
      if (isFromMagicModule(sle, from) && id == sle->sd->ident)
        return sle;
  }

  return nullptr;
}

/// Returns the _last_ StructLiteralExp magic attribute with identifier `id`
/// from the ldc magic module with identifier `from` (attributes or dcompute) if
/// it is applied to `sym`, otherwise returns nullptr.
StructLiteralExp *getLastMagicAttribute(Dsymbol *sym, const Identifier *id,
                                        const Identifier *from) {
  if (!sym->userAttribDecl())
    return nullptr;

  // Loop over all UDAs and find the last match
  StructLiteralExp *lastMatch = nullptr;
  Expressions *attrs = getAttributes(sym->userAttribDecl());
  expandTuples(attrs);
  for (auto attr : *attrs) {
    if (auto sle = attr->isStructLiteralExp())
      if (isFromMagicModule(sle, from) && id == sle->sd->ident)
        lastMatch = sle;
  }

  return lastMatch;
}

/// Calls `action` for each magic attribute with identifier `id` from
/// the ldc magic module with identifier `from` (attributes or dcompute)
/// applied to `sym`.
void callForEachMagicAttribute(Dsymbol &sym, const Identifier *id,
                               const Identifier *from,
                               std::function<void(StructLiteralExp *)> action) {
  if (!sym.userAttribDecl())
    return;

  // Loop over all UDAs and call `action` if a match was found.
  Expressions *attrs = getAttributes(sym.userAttribDecl());
  expandTuples(attrs);
  for (auto attr : *attrs) {
    if (auto sle = attr->isStructLiteralExp())
      if (isFromMagicModule(sle, from) && id == sle->sd->ident)
        action(sle);
  }
}

sinteger_t getIntElem(StructLiteralExp *sle, size_t idx) {
  auto arg = (*sle->elements)[idx];
  return arg->toInteger();
}

llvm::StringRef getStringElem(StructLiteralExp *sle, size_t idx) {
  if (auto arg = (*sle->elements)[idx]) {
    if (auto strexp = arg->isStringExp()) {
      DString str = strexp->peekString();
      return {str.ptr, str.length};
    }
  }
  // Default initialized element (arg->op == TOKnull)
  return {};
}

llvm::StringRef getFirstElemString(StructLiteralExp *sle) {
  return getStringElem(sle, 0);
}

// @allocSize(1)
// @allocSize(0,2)
void applyAttrAllocSize(StructLiteralExp *sle, IrFunction *irFunc) {
  checkStructElems(sle, {Type::tint32, Type::tint32});
  auto sizeArgIdx = getIntElem(sle, 0);
  auto numArgIdx = getIntElem(sle, 1);

  // Get the number of parameters that the user specified (excluding the
  // implicit `this` parameter)
  auto numUserParams = irFunc->irFty.args.size();

  // Verify that the index values are valid
  bool hasErrors = false;
  if (sizeArgIdx + 1 > sinteger_t(numUserParams)) {
    error(sle->loc,
          "`@ldc.attributes.allocSize.sizeArgIdx=%d` too large for "
          "function `%s` with %d arguments.",
          (int)sizeArgIdx, irFunc->decl->toChars(), (int)numUserParams);
    hasErrors = true;
  }
  if (numArgIdx + 1 > sinteger_t(numUserParams)) {
    error(sle->loc,
          "`@ldc.attributes.allocSize.numArgIdx=%d` too large for "
          "function `%s` with %d arguments.",
          (int)numArgIdx, irFunc->decl->toChars(), (int)numUserParams);
    hasErrors = true;
  }
  if (hasErrors)
    return;

  // Get the number of parameters of the function in LLVM IR. This includes
  // the `this` and sret parameters.
  const auto llvmNumParams = irFunc->irFty.funcType->getNumParams();

  // Offset to correct indices for sret and this parameters.
  // These parameters can never be used for allocsize, and the user-specified
  // index does not account for these.
  unsigned offset = llvmNumParams - numUserParams;

  // Calculate the param indices for the function as defined in LLVM IR
  const auto llvmSizeIdx = sizeArgIdx + offset;
  const auto llvmNumIdx = numArgIdx + offset;

  llvm::AttrBuilder builder(getGlobalContext());
  if (numArgIdx >= 0) {
    builder.addAllocSizeAttr(llvmSizeIdx, llvmNumIdx);
  } else {
#if LDC_LLVM_VER < 1600
    builder.addAllocSizeAttr(llvmSizeIdx, llvm::Optional<unsigned>());
#else
    builder.addAllocSizeAttr(llvmSizeIdx, std::optional<unsigned>());
#endif
  }

  llvm::Function *func = irFunc->getLLVMFunc();

  func->addFnAttrs(builder);
}

// @llvmAttr("key", "value")
// @llvmAttr("key")
void applyAttrLLVMAttr(StructLiteralExp *sle, llvm::AttrBuilder &attrs) {
  checkStructElems(sle, {Type::tstring, Type::tstring});
  llvm::StringRef key = getStringElem(sle, 0);
  llvm::StringRef value = getStringElem(sle, 1);
  if (value.empty()) {
    const auto kind = llvm::Attribute::getAttrKindFromName(key);
    if (kind != llvm::Attribute::None) {
      attrs.addAttribute(kind);
    } else {
      attrs.addAttribute(key);
    }
  } else {
    attrs.addAttribute(key, value);
  }
}

// @llvmFastMathFlag("flag")
void applyAttrLLVMFastMathFlag(StructLiteralExp *sle, IrFunction *irFunc) {
  checkStructElems(sle, {Type::tstring});
  llvm::StringRef value = getStringElem(sle, 0);

  if (value == "clear") {
    irFunc->FMF.clear();
  } else if (value == "fast") {
    irFunc->FMF.setFast();
  } else if (value == "contract") {
    irFunc->FMF.setAllowContract(true);
  } else if (value == "nnan") {
    irFunc->FMF.setNoNaNs();
  } else if (value == "ninf") {
    irFunc->FMF.setNoInfs();
  } else if (value == "nsz") {
    irFunc->FMF.setNoSignedZeros();
  } else if (value == "arcp") {
    irFunc->FMF.setAllowReciprocal();
  } else {
    warning(sle->loc,
        "ignoring unrecognized flag parameter `%.*s` for `@ldc.attributes.%s`",
        static_cast<int>(value.size()), value.data(),
        sle->sd->ident->toChars());
  }
}

void applyAttrOptStrategy(StructLiteralExp *sle, IrFunction *irFunc) {
  checkStructElems(sle, {Type::tstring});
  llvm::StringRef value = getStringElem(sle, 0);

  llvm::Function *func = irFunc->getLLVMFunc();
  if (value == "none") {
    if (irFunc->decl->inlining == PINLINE::always) {
      error(sle->loc,
            "cannot combine `@ldc.attributes.%s(\"none\")` with "
            "`pragma(inline, true)`",
            sle->sd->ident->toChars());
      return;
    }
    irFunc->decl->inlining = PINLINE::never;
    func->addFnAttr(llvm::Attribute::OptimizeNone);
  } else if (value == "optsize") {
    func->addFnAttr(llvm::Attribute::OptimizeForSize);
  } else if (value == "minsize") {
    func->addFnAttr(llvm::Attribute::MinSize);
  } else {
    warning(sle->loc,
        "ignoring unrecognized parameter `%.*s` for `@ldc.attributes.%s`",
        static_cast<int>(value.size()), value.data(),
        sle->sd->ident->toChars());
  }
}

void applyAttrSection(StructLiteralExp *sle, llvm::GlobalObject *globj) {
  checkStructElems(sle, {Type::tstring});
  globj->setSection(getFirstElemString(sle));
}

void applyAttrTarget(StructLiteralExp *sle, llvm::Function *func,
                     IrFunction *irFunc) {
  // TODO: this is a rudimentary implementation for @target. Many more
  // target-related attributes could be applied to functions (not just for
  // @target): clang applies many attributes that LDC does not.
  // The current implementation here does not do any checking of the specified
  // string and simply passes all to llvm.

#if LDC_LLVM_VER >= 1800
  #define startswith starts_with
#endif

  checkStructElems(sle, {Type::tstring});
  llvm::StringRef targetspec = getFirstElemString(sle);

  if (targetspec.empty() || targetspec == "default")
    return;

  llvm::StringRef CPU;
  std::vector<std::string> features;

  // Preserve the order of the features as they appear in the source
  // code. `hasFnAttribute` returns all the features accumulated
  // so far and they should remain at the beginning of the result.
  if (func->hasFnAttribute("target-features")) {
    auto attr = func->getFnAttribute("target-features");
    features.push_back(std::string(attr.getValueAsString()));
  }

  llvm::SmallVector<llvm::StringRef, 4> fragments;
  llvm::SplitString(targetspec, fragments, ",");
  // special strings: "arch=<cpu>", "tune=<...>", "fpmath=<...>"
  // if string starts with "no-", strip "no"
  // otherwise add "+"
  for (auto s : fragments) {
    s = s.trim();
    if (s.empty())
      continue;

    if (s.startswith("arch=")) {
      // TODO: be smarter than overwriting the previous arch= setting
      CPU = s.drop_front(5);
      continue;
    }
    if (s.startswith("tune=")) {
      // clang 3.8 ignores tune= too
      continue;
    }
    if (s.startswith("fpmath=")) {
      // TODO: implementation; clang 3.8 ignores fpmath= too
      continue;
    }
    if (s.startswith("no-")) {
      std::string f = (std::string("-") + s.drop_front(3)).str();
      features.emplace_back(std::move(f));
      continue;
    }
    std::string f = (std::string("+") + s).str();
    features.emplace_back(std::move(f));
  }

  if (!CPU.empty()) {
    func->addFnAttr("target-cpu", CPU);
    irFunc->targetCpuOverridden = true;
  }

  if (!features.empty()) {
    func->addFnAttr("target-features",
                    llvm::join(features.begin(), features.end(), ","));
    irFunc->targetFeaturesOverridden = true;
  }

#if LDC_LLVM_VER >= 1800
  #undef startswith
#endif
}

void applyAttrAssumeUsed(IRState &irs, StructLiteralExp *sle,
                         llvm::Constant *symbol) {
  checkStructElems(sle, {});
  irs.usedArray.push_back(symbol);
}

/// Tries to recognize the calling convention,
/// TODO: Add support "cc <n>"
/// Returns true if succesful, with the calling convention in 'callconv'.
/// Returns false if unsuccesful.
bool parseCallingConvention(llvm::StringRef name,
                            llvm::CallingConv::ID *callconv) {

  llvm::CallingConv::ID conv_id =
      llvm::StringSwitch<llvm::CallingConv::ID>(name)
          // Names recognized by Clang (see Clang's
          // CodeGenTypes::ClangCallConvToLLVMCallConv):
          .Case("stdcall", llvm::CallingConv::X86_StdCall)
          .Case("fastcall", llvm::CallingConv::X86_FastCall)
          .Case("regcall", llvm::CallingConv::X86_RegCall)
          .Case("thiscall", llvm::CallingConv::X86_ThisCall)
          .Case("ms_abi", llvm::CallingConv::Win64)
          .Case("sysv_abi", llvm::CallingConv::X86_64_SysV)
          .Case("pcs(\"aapcs\")", llvm::CallingConv::ARM_AAPCS)
          .Case("pcs(\"aapcs-vfp\")", llvm::CallingConv::ARM_AAPCS_VFP)
          .Case("intel_ocl_bicc", llvm::CallingConv::Intel_OCL_BI)
          .Case("pascal",
                llvm::CallingConv::C) // Check Clang, this may change in future
          .Case("vectorcall", llvm::CallingConv::X86_VectorCall)
          .Case("aarch64_vector_pcs", llvm::CallingConv::AArch64_VectorCall)
          .Case("preserve_most", llvm::CallingConv::PreserveMost)
          .Case("preserve_all", llvm::CallingConv::PreserveAll)
          .Case("swiftcall", llvm::CallingConv::Swift)
          .Case("swiftasynccall", llvm::CallingConv::SwiftTail)

          // Names recognized in LLVM IR (see LLVM's
          // LLParser::parseOptionalCallingConv):
          .Case("ccc", llvm::CallingConv::C)
          .Case("fastcc", llvm::CallingConv::Fast)
          .Case("coldcc", llvm::CallingConv::Cold)
          .Case("cfguard_checkcc", llvm::CallingConv::CFGuard_Check)
          .Case("x86_stdcallcc", llvm::CallingConv::X86_StdCall)
          .Case("x86_fastcallcc", llvm::CallingConv::X86_FastCall)
          .Case("x86_regcallcc", llvm::CallingConv::X86_RegCall)
          .Case("x86_thiscallcc", llvm::CallingConv::X86_ThisCall)
          .Case("x86_vectorcallcc", llvm::CallingConv::X86_VectorCall)
          .Case("arm_apcscc", llvm::CallingConv::ARM_APCS)
          .Case("arm_aapcscc", llvm::CallingConv::ARM_AAPCS)
          .Case("arm_aapcs_vfpcc", llvm::CallingConv::ARM_AAPCS_VFP)
          .Case("aarch64_vector_pcs", llvm::CallingConv::AArch64_VectorCall)
          .Case("aarch64_sve_vector_pcs",
                llvm::CallingConv::AArch64_SVE_VectorCall)
          .Case("msp430_intrcc", llvm::CallingConv::MSP430_INTR)
          .Case("avr_intrcc", llvm::CallingConv::AVR_INTR)
          .Case("avr_signalcc", llvm::CallingConv::AVR_SIGNAL)
          .Case("ptx_kernel", llvm::CallingConv::PTX_Kernel)
          .Case("ptx_device", llvm::CallingConv::PTX_Device)
          .Case("spir_kernel", llvm::CallingConv::SPIR_KERNEL)
          .Case("spir_func", llvm::CallingConv::SPIR_FUNC)
          .Case("intel_ocl_bicc", llvm::CallingConv::Intel_OCL_BI)
          .Case("x86_64_sysvcc", llvm::CallingConv::X86_64_SysV)
          .Case("win64cc", llvm::CallingConv::Win64)
#if LDC_LLVM_VER >= 1800
          .Case("webkit_jscc", llvm::CallingConv::WASM_EmscriptenInvoke)
#else
          .Case("webkit_jscc", llvm::CallingConv::WebKit_JS)
#endif
          .Case("anyregcc", llvm::CallingConv::AnyReg)
          .Case("preserve_mostcc", llvm::CallingConv::PreserveMost)
          .Case("preserve_allcc", llvm::CallingConv::PreserveAll)
          .Case("ghccc", llvm::CallingConv::GHC)
          .Case("swiftcc", llvm::CallingConv::Swift)
          .Case("swifttailcc", llvm::CallingConv::SwiftTail)
          .Case("x86_intrcc", llvm::CallingConv::X86_INTR)
#if LDC_LLVM_VER >= 1700
          .Case("hhvmcc", llvm::CallingConv::DUMMY_HHVM)
          .Case("hhvm_ccc", llvm::CallingConv::DUMMY_HHVM_C)
#else
          .Case("hhvmcc", llvm::CallingConv::HHVM)
          .Case("hhvm_ccc", llvm::CallingConv::HHVM_C)
#endif
          .Case("cxx_fast_tlscc", llvm::CallingConv::CXX_FAST_TLS)
          .Case("amdgpu_vs", llvm::CallingConv::AMDGPU_VS)
          .Case("amdgpu_gfx", llvm::CallingConv::AMDGPU_Gfx)
          .Case("amdgpu_ls", llvm::CallingConv::AMDGPU_LS)
          .Case("amdgpu_hs", llvm::CallingConv::AMDGPU_HS)
          .Case("amdgpu_es", llvm::CallingConv::AMDGPU_ES)
          .Case("amdgpu_gs", llvm::CallingConv::AMDGPU_GS)
          .Case("amdgpu_ps", llvm::CallingConv::AMDGPU_PS)
          .Case("amdgpu_cs", llvm::CallingConv::AMDGPU_CS)
          .Case("amdgpu_kernel", llvm::CallingConv::AMDGPU_KERNEL)
          .Case("tailcc", llvm::CallingConv::Tail)

          .Case("default", llvm::CallingConv::MaxID - 1)
          .Default(llvm::CallingConv::MaxID);

  bool success = (conv_id != llvm::CallingConv::MaxID);
  if (success && callconv)
    *callconv = conv_id;
  return success;
}

} // anonymous namespace

void applyVarDeclUDAs(VarDeclaration *decl, llvm::GlobalVariable *gvar) {
  if (!decl->userAttribDecl())
    return;

  Expressions *attrs = getAttributes(decl->userAttribDecl());
  expandTuples(attrs);
  for (auto &attr : *attrs) {
    auto sle = getLdcAttributesStruct(attr);
    if (!sle)
      continue;

    auto ident = sle->sd->ident;
    if (ident == Id::udaSection) {
      applyAttrSection(sle, gvar);
    } else if (ident == Id::udaHidden) {
      if (!decl->isExport()) // export visibility is stronger
        gvar->setVisibility(LLGlobalValue::HiddenVisibility);
    } else if (ident == Id::udaOptStrategy || ident == Id::udaTarget) {
      error(sle->loc,
            "special attribute `ldc.attributes.%s` is only valid for functions",
            ident->toChars());
    } else if (ident == Id::udaAssumeUsed) {
      applyAttrAssumeUsed(*gIR, sle, gvar);
    } else if (ident == Id::udaWeak) {
      // @weak is applied elsewhere
    } else if (ident == Id::udaDynamicCompile ||
               ident == Id::udaDynamicCompileEmit ||
               ident == Id::udaCallingConvention) {
      error(sle->loc,
            "special attribute `ldc.attributes.%s` is only valid for functions",
            ident->toChars());
    } else if (ident == Id::udaDynamicCompileConst) {
      getIrGlobal(decl)->dynamicCompileConst = true;
    } else {
      warning(sle->loc,
          "ignoring unrecognized special attribute `ldc.attributes.%s`",
          ident->toChars());
    }
  }
}

void applyFuncDeclUDAs(FuncDeclaration *decl, IrFunction *irFunc) {
  // function UDAs
  if (decl->userAttribDecl()) {
    llvm::Function *func = irFunc->getLLVMFunc();
    assert(func);

    Expressions *attrs = getAttributes(decl->userAttribDecl());
    expandTuples(attrs);
    for (auto &attr : *attrs) {
      auto sle = getLdcAttributesStruct(attr);
      if (!sle)
        continue;

      auto ident = sle->sd->ident;
      if (ident == Id::udaAllocSize) {
        applyAttrAllocSize(sle, irFunc);
      } else if (ident == Id::udaLLVMAttr) {
        llvm::AttrBuilder attrs(getGlobalContext());
        applyAttrLLVMAttr(sle, attrs);
        func->addFnAttrs(attrs);
      } else if (ident == Id::udaHidden) {
        if (!decl->isExport()) // export visibility is stronger
          func->setVisibility(LLGlobalValue::HiddenVisibility);
      } else if (ident == Id::udaLLVMFastMathFlag) {
        applyAttrLLVMFastMathFlag(sle, irFunc);
      } else if (ident == Id::udaOptStrategy) {
        applyAttrOptStrategy(sle, irFunc);
      } else if (ident == Id::udaSection) {
        applyAttrSection(sle, func);
      } else if (ident == Id::udaTarget) {
        applyAttrTarget(sle, func, irFunc);
      } else if (ident == Id::udaAssumeUsed) {
        applyAttrAssumeUsed(*gIR, sle, func);
      } else if (ident == Id::udaWeak || ident == Id::udaKernel ||
                 ident == Id::udaNoSanitize ||
                 ident == Id::udaCallingConvention ||
                 ident == Id::udaNoSplitStack) {
        // These UDAs are applied elsewhere, thus should silently be ignored here.
      } else if (ident == Id::udaDynamicCompile) {
        irFunc->dynamicCompile = true;
      } else if (ident == Id::udaDynamicCompileEmit) {
        irFunc->dynamicCompileEmit = true;
      } else if (ident == Id::udaDynamicCompileConst) {
        error(
            sle->loc,
            "special attribute `ldc.attributes.%s` is only valid for variables",
            ident->toChars());
      } else {
        warning(sle->loc,
            "ignoring unrecognized special attribute `ldc.attributes.%s`",
            ident->toChars());
      }
    }
  }

  // parameter UDAs
  auto parameterList = irFunc->type->parameterList;
  for (auto arg : irFunc->irFty.args) {
    if (arg->parametersIdx >= parameterList.length())
      continue;

    auto param =
        Parameter::getNth(parameterList.parameters, arg->parametersIdx);
    if (!param->userAttribDecl)
      continue;

    Expressions *attrs = getAttributes(param->userAttribDecl);
    expandTuples(attrs);
    for (auto &attr : *attrs) {
      auto sle = getLdcAttributesStruct(attr);
      if (!sle)
        continue;

      auto ident = sle->sd->ident;
      if (ident == Id::udaLLVMAttr) {
        applyAttrLLVMAttr(sle, arg->attrs);
      } else {
        warning(sle->loc,
                "ignoring unrecognized special parameter attribute "
                "`ldc.attributes.%s`",
                ident->toChars());
      }
    }
  }
}

/// Checks whether 'fd' has the @ldc.attributes.callingConvention("...") UDA applied.
/// If so, it returns the calling convention in 'callconv'.
bool hasCallingConventionUDA(FuncDeclaration *fd,
                             llvm::CallingConv::ID *callconv) {
  auto sle =
      getLastMagicAttribute(fd, Id::udaCallingConvention, Id::attributes);
  if (!sle)
    return false;

  checkStructElems(sle, {Type::tstring});
  auto name = getFirstElemString(sle);
  bool success = parseCallingConvention(name, callconv);
  if (!success)
    warning(sle->loc, "ignoring unrecognized calling convention name '%s' for "
                 "`@ldc.attributes.callingConvention`",
                 name.str().c_str());
  return success;
}

/// Checks whether 'sym' has the @ldc.attributes._weak() UDA applied.
bool hasWeakUDA(Dsymbol *sym) {
  auto sle = getMagicAttribute(sym, Id::udaWeak, Id::attributes);
  if (!sle)
    return false;

  checkStructElems(sle, {});
  auto vd = sym->isVarDeclaration();
  if (!(vd && vd->isDataseg()) && !sym->isFuncDeclaration())
    error(sym->loc,
          "`@ldc.attributes.weak` can only be applied to functions or "
          "global variables");
  return true;
}

/// Returns 0 if 'sym' does not have the @ldc.dcompute.compute() UDA applied.
/// Returns 1 + n if 'sym' does and is @compute(n).
extern "C" DComputeCompileFor hasComputeAttr(Dsymbol *sym) {

  auto sle = getMagicAttribute(sym, Id::udaCompute, Id::dcompute);
  if (!sle)
    return DComputeCompileFor::hostOnly;

  checkStructElems(sle, {Type::tint32});

  return static_cast<DComputeCompileFor>(1 + (*sle->elements)[0]->toInteger());
}

/// Checks whether 'sym' has the @ldc.dcompute._kernel() UDA applied.
bool hasKernelAttr(Dsymbol *sym) {
  auto sle = getMagicAttribute(sym, Id::udaKernel, Id::dcompute);
  if (!sle)
    return false;

  checkStructElems(sle, {});

  if (!sym->isFuncDeclaration() &&
      hasComputeAttr(sym->getModule()) != DComputeCompileFor::hostOnly) {
    error(sym->loc, "`@ldc.dcompute.kernel` can only be applied to functions"
                    " in modules marked `@ldc.dcompute.compute`");
  }

  return true;
}

/// Check whether `fd` has the `@ldc.attributes.noSplitStack` UDA applied.
bool hasNoSplitStackUDA(FuncDeclaration *fd) {
  auto sle = getMagicAttribute(fd, Id::udaNoSplitStack, Id::attributes);
  return sle != nullptr;
}

/// Creates a mask (for &) of @ldc.attributes.noSanitize UDA applied to the
/// function.
/// If a bit is set in the mask, then the sanitizer is enabled.
/// If a bit is not set in the mask, then the sanitizer is explicitly disabled
/// by @noSanitize.
unsigned getMaskFromNoSanitizeUDA(FuncDeclaration &fd) {
  opts::SanitizerBits inverse_mask = opts::NoneSanitizer;

  callForEachMagicAttribute(fd, Id::udaNoSanitize, Id::attributes,
                            [&inverse_mask](StructLiteralExp *sle) {
    checkStructElems(sle, {Type::tstring});
    auto name = getFirstElemString(sle);
    inverse_mask |= opts::parseSanitizerName(name, [&] {
      warning(sle->loc,
          "unrecognized sanitizer name '%s' for `@ldc.attributes.noSanitize`.",
          name.str().c_str());
    });
  });

  return ~inverse_mask;
}
