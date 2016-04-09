#include "gen/uda.h"

#include "gen/llvm.h"
#include "aggregate.h"
#include "attrib.h"
#include "declaration.h"
#include "expression.h"
#include "module.h"

#include "llvm/ADT/StringExtras.h"

namespace {

/// Names of the attribute structs we recognize.
namespace attr {
const std::string section = "section";
const std::string target  = "target";
const std::string weak    = "_weak";
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

void checkStructElems(StructLiteralExp *sle, llvm::ArrayRef<Type *> elemTypes) {
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

const char *getFirstElemString(StructLiteralExp *sle) {
  auto arg = (*sle->elements)[0];
  assert(arg->op == TOKstring);
  auto strexp = static_cast<StringExp *>(arg);
  assert(strexp->sz == 1);
  return strexp->toStringz();
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
    } else if (name == attr::target) {
      sle->error("Special attribute 'ldc.attributes.target' is only valid for "
                 "functions");
    } else if (name == attr::weak) {
      // @weak is applied elsewhere
    } else {
      sle->warning(
          "Ignoring unrecognized special attribute 'ldc.attributes.%s'",
          sle->sd->ident->string);
    }
  }
}

void applyFuncDeclUDAs(FuncDeclaration *decl, llvm::Function *func) {
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
      applyAttrSection(sle, func);
    } else if (name == attr::target) {
      applyAttrTarget(sle, func);
    } else if (name == attr::weak) {
      // @weak is applied elsewhere
    } else {
      sle->warning(
          "ignoring unrecognized special attribute 'ldc.attributes.%s'",
          sle->sd->ident->string);
    }
  }
}

/// Checks whether 'sym' has the @ldc.attributes._weak() UDA applied.
bool hasWeakUDA(Dsymbol *sym) {
  if (!sym->userAttribDecl)
    return false;

  // Loop over all UDAs and early return true if @weak was found.
  Expressions *attrs = sym->userAttribDecl->getAttributes();
  expandTuples(attrs);
  for (auto &attr : *attrs) {
    auto sle = getLdcAttributesStruct(attr);
    if (!sle)
      continue;

    auto name = sle->sd->ident->string;
    if (name == attr::weak) {
        // Check whether @weak can be applied to this symbol.
        // Because hasWeakUDA is currently only called for global symbols, this check never errors.
        auto vd = sym->isVarDeclaration();
        if (!(vd && vd->isDataseg()) && !sym->isFuncDeclaration()) {
          sym->error("@ldc.attributes.weak can only be applied to functions or global variables");
          return false;
        }

      return true;
    }
  }

  return false;
}
