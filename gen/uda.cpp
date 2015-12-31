#include "gen/uda.h"

#include "gen/llvm.h"
#include "aggregate.h"
#include "attrib.h"
#include "declaration.h"
#include "expression.h"
#include "module.h"

namespace {

/// Names of the attribute structs we recognize.
namespace attr {
const std::string section = "section";
}

bool isFromLdcAttibutes(StructLiteralExp *e) {
  auto moduleDecl = e->sd->getModule()->md;
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
  checkStructElems(sle, {Type::tstring});
  auto arg = (*sle->elements)[0];
  assert(arg->op == TOKstring);
  auto strexp = static_cast<StringExp *>(arg);
  assert(strexp->sz == 1);
  return static_cast<const char *>(strexp->string);
}

void applyAttrSection(StructLiteralExp *sle, llvm::GlobalObject *globj) {
  globj->setSection(getFirstElemString(sle));
}

void applyDeclUDAs(Declaration *decl, llvm::GlobalObject *globj) {
  if (!decl->userAttribDecl)
    return;

  Expressions *attrs = decl->userAttribDecl->getAttributes();
  expandTuples(attrs);
  for (auto &attr : *attrs) {
    // See whether we can evaluate the attribute at compile-time. All the LDC
    // attributes are struct literals that may be constructed using a CTFE
    // function.
    unsigned prevErrors = global.startGagging();
    auto e = ctfeInterpret(attr);
    if (global.endGagging(prevErrors)) {
      continue;
    }

    if (e->op != TOKstructliteral) {
      continue;
    }

    auto sle = static_cast<StructLiteralExp *>(e);

    if (!isFromLdcAttibutes(sle)) {
      continue;
    }

    auto name = sle->sd->ident->string;
    if (name == attr::section) {
      applyAttrSection(sle, globj);
    }
  }
}

} // anonymous namespace

void applyVarDeclUDAs(VarDeclaration *decl, llvm::GlobalVariable *gvar) {
  applyDeclUDAs(decl, gvar);
}

void applyFuncDeclUDAs(FuncDeclaration *decl, llvm::Function *func) {
  applyDeclUDAs(decl, func);
}
