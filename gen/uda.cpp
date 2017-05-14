#include "gen/uda.h"

#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "aggregate.h"
#include "attrib.h"
#include "declaration.h"
#include "expression.h"
#include "ir/irfunction.h"
#include "module.h"
#include "id.h"

#include "llvm/ADT/StringExtras.h"

namespace {

/// Checks whether `moduleDecl` is the ldc.attributes module.
bool isMagicModule(const ModuleDeclaration *moduleDecl, const Identifier* id) {
  if (!moduleDecl)
    return false;

  if (moduleDecl->id != id) {
    return false;
  }

  if (moduleDecl->packages->dim != 1 ||
      (*moduleDecl->packages)[0] != Id::ldc) {
    return false;
  }
  return true;
}

/// Checks whether the type of `e` is a struct from the ldc.attributes module.
bool isFromMagicModule(const StructLiteralExp *e,const Identifier* id) {
  auto moduleDecl = e->sd->getModule()->md;
  return isMagicModule(moduleDecl,id);
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

  if (e->op != TOKstructliteral) {
    return nullptr;
  }

  auto sle = static_cast<StructLiteralExp *>(attr);
  if (isFromMagicModule(sle,Id::attributes)) {
    return sle;
  }

  return nullptr;
}

void checkStructElems(StructLiteralExp *sle, ArrayParam<Type *> elemTypes) {
  if (sle->elements->dim != elemTypes.size()) {
    sle->error(
        "unexpected field count in 'ldc.attributes.%s'; does druntime not "
        "match compiler version?",
        sle->sd->ident->toChars());
    fatal();
  }

  for (size_t i = 0; i < sle->elements->dim; ++i) {
    if ((*sle->elements)[i]->type->toBasetype() != elemTypes[i]) {
      sle->error("invalid field type in 'ldc.attributes.%s'; does druntime not "
                 "match compiler version?",
                 sle->sd->ident->toChars());
      fatal();
    }
  }
}

/// Returns the StructLiteralExp magic attribute with identifier `id` if it is
/// applied to `sym`, otherwise returns nullptr.
StructLiteralExp *getMagicAttribute(Dsymbol *sym, const Identifier* id,
                                    const Identifier* from) {
  if (!sym->userAttribDecl)
    return nullptr;

  // Loop over all UDAs and early return the expression if a match was found.
  Expressions *attrs = sym->userAttribDecl->getAttributes();
  expandTuples(attrs);
  for (auto &attr : *attrs) {
    if (attr->op != TOKstructliteral)
        continue;
    auto sle = static_cast<StructLiteralExp *>(e);
    if (!isFromMagicModule(sle,from))
        continue;

    if (id == sle->sd->ident) {
      return sle;
    }
  }

  return nullptr;
}

sinteger_t getIntElem(StructLiteralExp *sle, size_t idx) {
  auto arg = (*sle->elements)[idx];
  return arg->toInteger();
}

/// Returns a null-terminated string
const char *getStringElem(StructLiteralExp *sle, size_t idx) {
  auto arg = (*sle->elements)[idx];
  if (arg && arg->op == TOKstring) {
    auto strexp = static_cast<StringExp *>(arg);
    assert(strexp->sz == 1);
    return strexp->toStringz();
  }
  // Default initialized element (arg->op == TOKnull)
  return "";
}

/// Returns a null-terminated string
const char *getFirstElemString(StructLiteralExp *sle) {
  return getStringElem(sle, 0);
}

// @allocSize(1)
// @allocSize(0,2)
void applyAttrAllocSize(StructLiteralExp *sle, IrFunction *irFunc) {
  llvm::Function *func = irFunc->getLLVMFunc();

  checkStructElems(sle, {Type::tint32, Type::tint32});
  auto sizeArgIdx = getIntElem(sle, 0);
  auto numArgIdx = getIntElem(sle, 1);

  // Get the number of parameters that the user specified (excluding the
  // implicit `this` parameter)
  auto numUserParams = irFunc->irFty.args.size();

  // Get the number of parameters of the function in LLVM IR. This includes
  // the `this` and sret parameters.
  auto llvmNumParams = irFunc->irFty.funcType->getNumParams();

  // Verify that the index values are valid
  bool error = false;
  if (sizeArgIdx + 1 > sinteger_t(numUserParams)) {
    sle->error("@ldc.attributes.allocSize.sizeArgIdx=%d too large for function "
               "`%s` with %d arguments.",
               (int)sizeArgIdx, irFunc->decl->toChars(), (int)numUserParams);
    error = true;
  }
  if (numArgIdx + 1 > sinteger_t(numUserParams)) {
    sle->error("@ldc.attributes.allocSize.numArgIdx=%d too large for function "
               "`%s` with %d arguments.",
               (int)numArgIdx, irFunc->decl->toChars(), (int)numUserParams);
    error = true;
  }
  if (error)
    return;

// The allocSize attribute is only effective for LLVM >= 3.9.
#if LDC_LLVM_VER >= 309
  // Offset to correct indices for sret and this parameters.
  // These parameters can never be used for allocsize, and the user-specified
  // index does not account for these.
  unsigned offset = llvmNumParams - numUserParams;

  // Calculate the param indices for the function as defined in LLVM IR
  auto llvmSizeIdx =
      irFunc->irFty.reverseParams ? numUserParams - sizeArgIdx - 1 : sizeArgIdx;
  auto llvmNumIdx =
      irFunc->irFty.reverseParams ? numUserParams - numArgIdx - 1 : numArgIdx;
  llvmSizeIdx += offset;
  llvmNumIdx += offset;

  llvm::AttrBuilder builder;
  if (numArgIdx >= 0) {
    builder.addAllocSizeAttr(llvmSizeIdx, llvmNumIdx);
  } else {
    builder.addAllocSizeAttr(llvmSizeIdx, llvm::Optional<unsigned>());
  }
  func->addAttributes(LLAttributeSet::FunctionIndex,
                      LLAttributeSet::get(func->getContext(),
                                          LLAttributeSet::FunctionIndex,
                                          builder));
#endif
}

// @llvmAttr("key", "value")
// @llvmAttr("key")
void applyAttrLLVMAttr(StructLiteralExp *sle, llvm::Function *func) {
  checkStructElems(sle, {Type::tstring, Type::tstring});
  llvm::StringRef key = getStringElem(sle, 0);
  llvm::StringRef value = getStringElem(sle, 1);
  if (value.empty()) {
    func->addFnAttr(key);
  } else {
    func->addFnAttr(key, value);
  }
}

// @llvmFastMathFlag("flag")
void applyAttrLLVMFastMathFlag(StructLiteralExp *sle, IrFunction *irFunc) {
  checkStructElems(sle, {Type::tstring});
  llvm::StringRef value = getStringElem(sle, 0);

  if (value == "clear") {
    irFunc->FMF.clear();
  } else if (value == "fast") {
    irFunc->FMF.setUnsafeAlgebra();
  } else if (value == "contract") {
#if LDC_LLVM_VER >= 500
    irFunc->FMF.setAllowContract(true);
#else
    sle->warning("ignoring parameter \"contract\" for @ldc.attributes.%s: "
                 "LDC needs to be built against LLVM 5.0+ for support",
                 sle->sd->ident->toChars());
#endif
  } else if (value == "nnan") {
    irFunc->FMF.setNoNaNs();
  } else if (value == "ninf") {
    irFunc->FMF.setNoInfs();
  } else if (value == "nsz") {
    irFunc->FMF.setNoSignedZeros();
  } else if (value == "arcp") {
    irFunc->FMF.setAllowReciprocal();
  } else {
    // `value` is a null-terminated returned from getStringElem so can be passed
    // to warning("... %s ...").
    sle->warning(
        "ignoring unrecognized flag parameter '%s' for '@ldc.attributes.%s'",
        value.data(), sle->sd->ident->toChars());
  }
}

void applyAttrOptStrategy(StructLiteralExp *sle, IrFunction *irFunc) {
  checkStructElems(sle, {Type::tstring});
  llvm::StringRef value = getStringElem(sle, 0);

  llvm::Function *func = irFunc->getLLVMFunc();
  if (value == "none") {
    if (irFunc->decl->inlining == PINLINEalways) {
      sle->error("cannot combine '@ldc.attributes.%s(\"none\")' with "
                 "'pragma(inline, true)'",
                 sle->sd->ident->toChars());
      return;
    }
    irFunc->decl->inlining = PINLINEnever;
    func->addFnAttr(llvm::Attribute::OptimizeNone);
  } else if (value == "optsize") {
    func->addFnAttr(llvm::Attribute::OptimizeForSize);
  } else if (value == "minsize") {
    func->addFnAttr(llvm::Attribute::MinSize);
  } else {
    sle->warning(
        "ignoring unrecognized parameter '%s' for '@ldc.attributes.%s'",
        value.data(), sle->sd->ident->toChars());
  }
}

void applyAttrSection(StructLiteralExp *sle, llvm::GlobalObject *globj) {
  checkStructElems(sle, {Type::tstring});
  globj->setSection(getFirstElemString(sle));
}

void applyAttrTarget(StructLiteralExp *sle, llvm::Function *func) {
  // TODO: this is a rudimentary implementation for @target. Many more
  // target-related attributes could be applied to functions (not just for
  // @target): clang applies many attributes that LDC does not.
  // The current implementation here does not do any checking of the specified
  // string and simply passes all to llvm.

  checkStructElems(sle, {Type::tstring});
  std::string targetspec = getFirstElemString(sle);

  if (targetspec.empty() || targetspec == "default")
    return;

  llvm::StringRef CPU;
  std::vector<std::string> features;

  if (func->hasFnAttribute("target-features")) {
    auto attr = func->getFnAttribute("target-features");
    features.push_back(attr.getValueAsString());
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

  if (!CPU.empty())
    func->addFnAttr("target-cpu", CPU);
  if (!features.empty()) {
    // Sorting the features puts negative features ("-") after positive features
    // ("+"). This provides the desired behavior of negative features overriding
    // positive features regardless of their order in the source code.
    sort(features.begin(), features.end());
    func->addFnAttr("target-features",
                    llvm::join(features.begin(), features.end(), ","));
  }
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
    } else if (ident == Id::udaOptStrategy || ident == Id::udaTarget) {
      sle->error(
          "Special attribute 'ldc.attributes.%s' is only valid for functions",
          ident->toChars());
    } else if (ident == Id::udaWeak) {
      // @weak is applied elsewhere
    } else {
      sle->warning(
          "Ignoring unrecognized special attribute 'ldc.attributes.%s'", ident->toChars());
    }
  }
}

void applyFuncDeclUDAs(FuncDeclaration *decl, IrFunction *irFunc) {
  if (!decl->userAttribDecl)
    return;

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
    } else if (ident == Id::udaLlvmAttr) {
      applyAttrLLVMAttr(sle, func);
    } else if (ident == Id::udaLlvmFastMathFlag) {
      applyAttrLLVMFastMathFlag(sle, irFunc);
    } else if (ident == Id::udaOptStrategy) {
      applyAttrOptStrategy(sle, irFunc);
    } else if (ident == Id::udaSection) {
      applyAttrSection(sle, func);
    } else if (ident == Id::udaTarget) {
      applyAttrTarget(sle, func);
    } else if (ident == Id::udaWeak || ident == Id::udaKernel) {
      // @weak and @kernel are applied elsewhere
    } else {
      sle->warning(
          "Ignoring unrecognized special attribute 'ldc.attributes.%s'", ident->toChars());
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
    sym->error("@ldc.attributes.weak can only be applied to functions or "
               "global variables");
  return true;
}

/// Returns 0 if 'sym' does not have the @ldc.attributes.compute() UDA applied.
/// Returns 1 if 'sym' does but is @compute(0), meaning generate only for
///     compute.
/// Returns 2 if 'sym' does and is @compute(1), meaning generate for compute
///     and for host.
int hasComputeAttr(Dsymbol *sym) {

  auto sle = getMagicAttribute(sym, Id::udaCompute, Id::dcompute);
  if (!sle)
    return 0;

  checkStructElems(sle, {Type::tint32});

  return 1 + (*sle->elements)[0]->toInteger();
}

/// Checks whether 'sym' has the @ldc.attributes._kernel() UDA applied.
bool hasKernelAttr(Dsymbol *sym) {
  auto sle = getMagicAttribute(sym, Id::udaKernel, Id::dcompute);
  if (!sle)
    return false;

  checkStructElems(sle, {});

  if (!sym->isFuncDeclaration() && !hasComputeAttr(sym->getModule()))
    sym->error("@ldc.dcompute.kernel can only be applied to functions"
               " in modules marked @lcd.dcompute.compute");

  return true;
}

