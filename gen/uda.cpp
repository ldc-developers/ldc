#include "gen/uda.h"

#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "aggregate.h"
#include "attrib.h"
#include "declaration.h"
#include "expression.h"
#include "ir/irfunction.h"
#include "module.h"

#include "llvm/ADT/StringExtras.h"

namespace {

/// Names of the attribute structs we recognize.
namespace attr {
const std::string llvmAttr = "llvmAttr";
const std::string llvmFastMathFlag = "llvmFastMathFlag";
const std::string optStrategy = "optStrategy";
const std::string section = "section";
const std::string target = "target";
const std::string weak = "_weak";
}

/// Checks whether `moduleDecl` is the ldc.attributes module.
bool isLdcAttibutes(const ModuleDeclaration *moduleDecl) {
  if (!moduleDecl)
    return false;

  if (strcmp("attributes", moduleDecl->id->string)) {
    return false;
  }

  if (moduleDecl->packages->dim != 1 ||
      strcmp("ldc", (*moduleDecl->packages)[0]->string)) {
    return false;
  }
  return true;
}

/// Checks whether the type of `e` is a struct from the ldc.attributes module.
bool isFromLdcAttibutes(const StructLiteralExp *e) {
  auto moduleDecl = e->sd->getModule()->md;
  return isLdcAttibutes(moduleDecl);
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

  auto sle = static_cast<StructLiteralExp *>(e);
  if (isFromLdcAttibutes(sle)) {
    return sle;
  }

  return nullptr;
}

void checkStructElems(StructLiteralExp *sle, ArrayParam<Type *> elemTypes) {
  if (sle->elements->dim != elemTypes.size()) {
    sle->error(
        "unexpected field count in 'ldc.attributes.%s'; does druntime not "
        "match compiler version?",
        sle->sd->ident->string);
    fatal();
  }

  for (size_t i = 0; i < sle->elements->dim; ++i) {
    if ((*sle->elements)[i]->type != elemTypes[i]) {
      sle->error("invalid field type in 'ldc.attributes.%s'; does druntime not "
                 "match compiler version?",
                 sle->sd->ident->string);
      fatal();
    }
  }
}

/// Returns the StructLiteralExp magic attribute with name `name` if it is
/// applied to `sym`, otherwise returns nullptr.
StructLiteralExp *getMagicAttribute(Dsymbol *sym, std::string name) {
  if (!sym->userAttribDecl)
    return nullptr;

  // Loop over all UDAs and early return the expression if a match was found.
  Expressions *attrs = sym->userAttribDecl->getAttributes();
  expandTuples(attrs);
  for (auto &attr : *attrs) {
    auto sle = getLdcAttributesStruct(attr);
    if (!sle)
      continue;

    if (name == sle->sd->ident->string) {
      return sle;
    }
  }

  return nullptr;
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
        value.data(), sle->sd->ident->string);
  }
}

void applyAttrOptStrategy(StructLiteralExp *sle, IrFunction *irFunc) {
  checkStructElems(sle, {Type::tstring});
  llvm::StringRef value = getStringElem(sle, 0);

  if (value == "none") {
    if (irFunc->decl->inlining == PINLINEalways) {
      sle->error("cannot combine '@ldc.attributes.%s(\"none\")' with "
                 "'pragma(inline, true)'",
                 sle->sd->ident->string);
      return;
    }
    irFunc->decl->inlining = PINLINEnever;
    irFunc->func->addFnAttr(llvm::Attribute::OptimizeNone);
  } else if (value == "optsize") {
    irFunc->func->addFnAttr(llvm::Attribute::OptimizeForSize);
  } else if (value == "minsize") {
    irFunc->func->addFnAttr(llvm::Attribute::MinSize);
  } else {
    sle->warning(
        "ignoring unrecognized parameter '%s' for '@ldc.attributes.%s'",
        value.data(), sle->sd->ident->string);
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

    auto name = sle->sd->ident->string;
    if (name == attr::section) {
      applyAttrSection(sle, gvar);
    } else if (name == attr::optStrategy || name == attr::target) {
      sle->error(
          "Special attribute 'ldc.attributes.%s' is only valid for functions",
          name);
    } else if (name == attr::weak) {
      // @weak is applied elsewhere
    } else {
      sle->warning(
          "Ignoring unrecognized special attribute 'ldc.attributes.%s'", name);
    }
  }
}

void applyFuncDeclUDAs(FuncDeclaration *decl, IrFunction *irFunc) {
  if (!decl->userAttribDecl)
    return;

  llvm::Function *func = irFunc->func;
  assert(func);

  Expressions *attrs = decl->userAttribDecl->getAttributes();
  expandTuples(attrs);
  for (auto &attr : *attrs) {
    auto sle = getLdcAttributesStruct(attr);
    if (!sle)
      continue;

    auto name = sle->sd->ident->string;
    if (name == attr::llvmAttr) {
      applyAttrLLVMAttr(sle, func);
    } else if (name == attr::llvmFastMathFlag) {
      applyAttrLLVMFastMathFlag(sle, irFunc);
    } else if (name == attr::optStrategy) {
      applyAttrOptStrategy(sle, irFunc);
    } else if (name == attr::section) {
      applyAttrSection(sle, func);
    } else if (name == attr::target) {
      applyAttrTarget(sle, func);
    } else if (name == attr::weak) {
      // @weak is applied elsewhere
    } else {
      sle->warning(
          "Ignoring unrecognized special attribute 'ldc.attributes.%s'", name);
    }
  }
}

/// Checks whether 'sym' has the @ldc.attributes._weak() UDA applied.
bool hasWeakUDA(Dsymbol *sym) {
  auto sle = getMagicAttribute(sym, attr::weak);
  if (!sle)
    return false;

  checkStructElems(sle, {});
  auto vd = sym->isVarDeclaration();
  if (!(vd && vd->isDataseg()) && !sym->isFuncDeclaration())
    sym->error("@ldc.attributes.weak can only be applied to functions or "
               "global variables");
  return true;
}
