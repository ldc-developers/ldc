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

#if LDC_LLVM_VER < 1100
namespace llvm {
// Auto-generate:
// Attribute::AttrKind getAttrKindFromName(StringRef AttrName) { ... }
#define GET_ATTR_KIND_FROM_NAME
#if LDC_LLVM_VER < 700
#include "llvm/IR/Attributes.gen"
#else
#include "llvm/IR/Attributes.inc"
#endif
}
#endif

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
  auto e = attr->ctfeInterpret();
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
    sle->error("unexpected field count in `ldc.%s.%s`; does druntime not "
               "match compiler version?",
               sle->sd->getModule()->md->id->toChars(),
               sle->sd->ident->toChars());
    fatal();
  }

  for (size_t i = 0; i < sle->elements->length; ++i) {
    if ((*sle->elements)[i]->type->toBasetype() != elemTypes[i]) {
      sle->error("invalid field type in `ldc.%s.%s`; does druntime not "
                 "match compiler version?",
                 sle->sd->getModule()->md->id->toChars(),
                 sle->sd->ident->toChars());
      fatal();
    }
  }
}

/// Returns the StructLiteralExp magic attribute with identifier `id` from
/// the ldc magic module with identifier `from` (attributes or dcompute)
/// if it is applied to `sym`, otherwise returns nullptr.
StructLiteralExp *getMagicAttribute(Dsymbol *sym, const Identifier *id,
                                    const Identifier *from) {
  if (!sym->userAttribDecl)
    return nullptr;

  // Loop over all UDAs and early return the expression if a match was found.
  Expressions *attrs = sym->userAttribDecl->getAttributes();
  expandTuples(attrs);
  for (auto attr : *attrs) {
    if (auto sle = attr->isStructLiteralExp())
      if (isFromMagicModule(sle, from) && id == sle->sd->ident)
        return sle;
  }

  return nullptr;
}

/// Calls `action` for each magic attribute with identifier `id` from
/// the ldc magic module with identifier `from` (attributes or dcompute)
/// applied to `sym`.
void callForEachMagicAttribute(Dsymbol &sym, const Identifier *id,
                               const Identifier *from,
                               std::function<void(StructLiteralExp *)> action) {
  if (!sym.userAttribDecl)
    return;

  // Loop over all UDAs and call `action` if a match was found.
  Expressions *attrs = sym.userAttribDecl->getAttributes();
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
  bool error = false;
  if (sizeArgIdx + 1 > sinteger_t(numUserParams)) {
    sle->error("`@ldc.attributes.allocSize.sizeArgIdx=%d` too large for "
               "function `%s` with %d arguments.",
               (int)sizeArgIdx, irFunc->decl->toChars(), (int)numUserParams);
    error = true;
  }
  if (numArgIdx + 1 > sinteger_t(numUserParams)) {
    sle->error("`@ldc.attributes.allocSize.numArgIdx=%d` too large for "
               "function `%s` with %d arguments.",
               (int)numArgIdx, irFunc->decl->toChars(), (int)numUserParams);
    error = true;
  }
  if (error)
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

#if LDC_LLVM_VER >= 1400
  llvm::AttrBuilder builder(getGlobalContext());
#else
  llvm::AttrBuilder builder;
#endif
  if (numArgIdx >= 0) {
    builder.addAllocSizeAttr(llvmSizeIdx, llvmNumIdx);
  } else {
    builder.addAllocSizeAttr(llvmSizeIdx, llvm::Optional<unsigned>());
  }

  llvm::Function *func = irFunc->getLLVMFunc();

#if LDC_LLVM_VER >= 1400
  func->addFnAttrs(builder);
#else
  func->addAttributes(LLAttributeList::FunctionIndex, builder);
#endif
}

// @llvmAttr("key", "value")
// @llvmAttr("key")
void applyAttrLLVMAttr(StructLiteralExp *sle, llvm::AttrBuilder &attrs) {
  checkStructElems(sle, {Type::tstring, Type::tstring});
  llvm::StringRef key = getStringElem(sle, 0);
  llvm::StringRef value = getStringElem(sle, 1);
  if (value.empty()) {
#if LDC_LLVM_VER >= 1100
    const auto kind = llvm::Attribute::getAttrKindFromName(key);
#else
    const auto kind = llvm::getAttrKindFromName(key);
#endif
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
    sle->warning(
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
      sle->error("cannot combine `@ldc.attributes.%s(\"none\")` with "
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
    sle->warning(
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

  checkStructElems(sle, {Type::tstring});
  llvm::StringRef targetspec = getFirstElemString(sle);

  if (targetspec.empty() || targetspec == "default")
    return;

  llvm::StringRef CPU;
  std::vector<std::string> features;

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
    // Sorting the features puts negative features ("-") after positive features
    // ("+"). This provides the desired behavior of negative features overriding
    // positive features regardless of their order in the source code.
    sort(features.begin(), features.end());
    func->addFnAttr("target-features",
                    llvm::join(features.begin(), features.end(), ","));
    irFunc->targetFeaturesOverridden = true;
  }
}

void applyAttrAssumeUsed(IRState &irs, StructLiteralExp *sle,
                         llvm::Constant *symbol) {
  checkStructElems(sle, {});
  irs.usedArray.push_back(symbol);
}

} // anonymous namespace

void applyVarDeclUDAs(VarDeclaration *decl, llvm::GlobalVariable *gvar) {
  if (!decl->userAttribDecl)
    return;

  Expressions *attrs = decl->userAttribDecl->getAttributes();
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
      sle->error(
          "Special attribute `ldc.attributes.%s` is only valid for functions",
          ident->toChars());
    } else if (ident == Id::udaAssumeUsed) {
      applyAttrAssumeUsed(*gIR, sle, gvar);
    } else if (ident == Id::udaWeak) {
      // @weak is applied elsewhere
    } else if (ident == Id::udaDynamicCompile ||
               ident == Id::udaDynamicCompileEmit) {
      sle->error(
          "Special attribute `ldc.attributes.%s` is only valid for functions",
          ident->toChars());
    } else if (ident == Id::udaDynamicCompileConst) {
      getIrGlobal(decl)->dynamicCompileConst = true;
    } else {
      sle->warning(
          "Ignoring unrecognized special attribute `ldc.attributes.%s`",
          ident->toChars());
    }
  }
}

void applyFuncDeclUDAs(FuncDeclaration *decl, IrFunction *irFunc) {
  // function UDAs
  if (decl->userAttribDecl) {
    llvm::Function *func = irFunc->getLLVMFunc();
    assert(func);

    Expressions *attrs = decl->userAttribDecl->getAttributes();
    expandTuples(attrs);
    for (auto &attr : *attrs) {
      auto sle = getLdcAttributesStruct(attr);
      if (!sle)
        continue;

      auto ident = sle->sd->ident;
      if (ident == Id::udaAllocSize) {
        applyAttrAllocSize(sle, irFunc);
      } else if (ident == Id::udaLLVMAttr) {
#if LDC_LLVM_VER >= 1400
        llvm::AttrBuilder attrs(getGlobalContext());
#else
        llvm::AttrBuilder attrs;
#endif
        applyAttrLLVMAttr(sle, attrs);
#if LDC_LLVM_VER >= 1400
        func->addFnAttrs(attrs);
#else
        func->addAttributes(LLAttributeList::FunctionIndex, attrs);
#endif
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
                 ident == Id::udaNoSanitize) {
        // These UDAs are applied elsewhere, thus should silently be ignored here.
      } else if (ident == Id::udaDynamicCompile) {
        irFunc->dynamicCompile = true;
      } else if (ident == Id::udaDynamicCompileEmit) {
        irFunc->dynamicCompileEmit = true;
      } else if (ident == Id::udaDynamicCompileConst) {
        sle->error(
            "Special attribute `ldc.attributes.%s` is only valid for variables",
            ident->toChars());
      } else {
        sle->warning(
            "Ignoring unrecognized special attribute `ldc.attributes.%s`",
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

    Expressions *attrs = param->userAttribDecl->getAttributes();
    expandTuples(attrs);
    for (auto &attr : *attrs) {
      auto sle = getLdcAttributesStruct(attr);
      if (!sle)
        continue;

      auto ident = sle->sd->ident;
      if (ident == Id::udaLLVMAttr) {
        applyAttrLLVMAttr(sle, arg->attrs);
      } else {
        sle->warning("Ignoring unrecognized special parameter attribute "
                     "`ldc.attributes.%s`",
                     ident->toChars());
      }
    }
  }
}

/// Checks whether 'sym' has the @ldc.attributes._weak() UDA applied.
bool hasWeakUDA(Dsymbol *sym) {
  auto sle = getMagicAttribute(sym, Id::udaWeak, Id::attributes);
  if (!sle)
    return false;

  checkStructElems(sle, {});
  auto vd = sym->isVarDeclaration();
  if (!(vd && vd->isDataseg()) && !sym->isFuncDeclaration())
    sym->error("`@ldc.attributes.weak` can only be applied to functions or "
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
    sym->error("`@ldc.dcompute.kernel` can only be applied to functions"
               " in modules marked `@ldc.dcompute.compute`");
  }

  return true;
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
      sle->warning(
          "Unrecognized sanitizer name '%s' for `@ldc.attributes.noSanitize`.",
          name.str().c_str());
    });
  });

  return ~inverse_mask;
}
