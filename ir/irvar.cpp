//===-- irvar.cpp ---------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irvar.h"

#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/init.h"
#include "gen/dynamiccompile.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/mangling.h"
#include "gen/pragma.h"
#include "gen/uda.h"
#include "ir/irdsymbol.h"

//////////////////////////////////////////////////////////////////////////////

LLValue *IrGlobal::getValue(bool define) {
  if (!value) {
    declare();

    if (!define && DtoIsTemplateInstance(V))
      define = true;
  }

  if (define && !(V->storage_class & STCextern)) {
    auto gvar = llvm::dyn_cast<LLGlobalVariable>(value);
    const bool isDefined = !gvar || gvar->hasInitializer();
    if (!isDefined)
      this->define();
  }

  return value;
}

void IrGlobal::declare() {
  Logger::println("Declaring global: %s", V->toChars());
  LOG_SCOPE

  if (V->parent) {
    Logger::println("parent: %s (%s)", V->parent->toChars(), V->parent->kind());
  } else {
    Logger::println("parent: null");
  }

  assert(!value);

  // If a const/immutable value has a proper initializer (not "= void"),
  // it cannot be assigned again in a static constructor. Thus, we can
  // emit it as read-only data.
  // We also do so for forward-declared (extern) globals, just like clang.
  const bool isLLConst = (V->isConst() || V->isImmutable()) &&
                         ((V->_init && !V->_init->isVoidInitializer()) ||
                          (V->storage_class & STCextern));

  const auto irMangle = getIRMangledName(V);

  // Since the type of a global must exactly match the type of its
  // initializer, we cannot know the type until after we have emitted the
  // latter (e.g. in case of unions, …). However, it is legal for the
  // initializer to refer to the address of the variable. Thus, we first
  // create a global with the generic type (note the assignment to
  // vd->ir->irGlobal->value!), and in case we also do an initializer
  // with a different type later, swap it out and replace any existing
  // uses with bitcasts to the previous type.

  LLGlobalVariable *gvar =
      declareGlobal(V->loc, gIR->module, DtoMemType(V->type), irMangle,
                    isLLConst, V->isThreadlocal());
  value = gvar;

  if (V->llvmInternal == LLVMextern_weak)
    gvar->setLinkage(llvm::GlobalValue::ExternalWeakLinkage);

  // Set the alignment (it is important not to use type->alignsize because
  // VarDeclarations can have an align() attribute independent of the type
  // as well).
  gvar->setAlignment(LLMaybeAlign(DtoAlignment(V)));

  // Windows: initialize DLL storage class with `dllimport` for `export`ed
  // symbols
  if (global.params.isWindows && V->isExport()) {
    gvar->setDLLStorageClass(LLGlobalValue::DLLImportStorageClass);
  }

  applyVarDeclUDAs(V, gvar);

  if (dynamicCompileConst)
    addDynamicCompiledVar(gIR, this);
}

void IrGlobal::define() {
  Logger::println("Defining global: %s", V->toChars());

  if (global.params.vtls && V->isThreadlocal() &&
      !(V->storage_class & STCtemp)) {
    message("%s: `%s` is thread local", V->loc.toChars(), V->toChars());
  }

  LLConstant *initVal = DtoConstInitializer(V->loc, V->type, V->_init);

  // Set the initializer, swapping out the variable if the types do not
  // match.
  auto gvar = llvm::cast<LLGlobalVariable>(value);
  value = gIR->setGlobalVarInitializer(gvar, initVal, V);

  // Finalize DLL storage class.
  if (gvar->hasDLLImportStorageClass()) {
    gvar->setDLLStorageClass(LLGlobalValue::DLLExportStorageClass);
  }

  // If this global is used from a naked function, we need to create an
  // artificial "use" for it, or it could be removed by the optimizer if
  // the only reference to it is in inline asm.
  if (nakedUse) {
    gIR->usedArray.push_back(gvar);
  }

  // Also set up the debug info.
  gIR->DBuilder.EmitGlobalVariable(gvar, V);
}

//////////////////////////////////////////////////////////////////////////////

IrVar *getIrVar(VarDeclaration *decl) {
  assert(isIrVarCreated(decl));
  assert(decl->ir->irVar != NULL);
  return decl->ir->irVar;
}

llvm::Value *getIrValue(VarDeclaration *decl) { return getIrVar(decl)->value; }

bool isIrVarCreated(VarDeclaration *decl) {
  int t = decl->ir->type();
  bool isIrVar = t == IrDsymbol::GlobalType || t == IrDsymbol::LocalType ||
                 t == IrDsymbol::ParamterType || t == IrDsymbol::FieldType;
  assert(isIrVar || t == IrDsymbol::NotSet);
  return isIrVar;
}

//////////////////////////////////////////////////////////////////////////////

IrGlobal *getIrGlobal(VarDeclaration *decl, bool create) {
  if (!isIrGlobalCreated(decl) && create) {
    assert(decl->ir->irGlobal == NULL);
    decl->ir->irGlobal = new IrGlobal(decl);
    decl->ir->m_type = IrDsymbol::GlobalType;
  }
  assert(decl->ir->irGlobal != NULL);
  return decl->ir->irGlobal;
}

bool isIrGlobalCreated(VarDeclaration *decl) {
  int t = decl->ir->type();
  assert(t == IrDsymbol::GlobalType || t == IrDsymbol::NotSet);
  return t == IrDsymbol::GlobalType;
}

//////////////////////////////////////////////////////////////////////////////

IrLocal *getIrLocal(VarDeclaration *decl, bool create) {
  if (!isIrLocalCreated(decl) && create) {
    assert(decl->ir->irLocal == NULL);
    decl->ir->irLocal = new IrLocal(decl);
    decl->ir->m_type = IrDsymbol::LocalType;
  }
  assert(decl->ir->irLocal != NULL);
  return decl->ir->irLocal;
}

bool isIrLocalCreated(VarDeclaration *decl) {
  int t = decl->ir->type();
  assert(t == IrDsymbol::LocalType || t == IrDsymbol::ParamterType ||
         t == IrDsymbol::NotSet);
  return t == IrDsymbol::LocalType || t == IrDsymbol::ParamterType;
}

//////////////////////////////////////////////////////////////////////////////

IrParameter *getIrParameter(VarDeclaration *decl, bool create) {
  if (!isIrParameterCreated(decl) && create) {
    assert(decl->ir->irParam == NULL);
    decl->ir->irParam = new IrParameter(decl);
    decl->ir->m_type = IrDsymbol::ParamterType;
  }
  return decl->ir->irParam;
}

bool isIrParameterCreated(VarDeclaration *decl) {
  int t = decl->ir->type();
  assert(t == IrDsymbol::ParamterType || t == IrDsymbol::NotSet);
  return t == IrDsymbol::ParamterType;
}

//////////////////////////////////////////////////////////////////////////////

IrField *getIrField(VarDeclaration *decl, bool create) {
  if (!isIrFieldCreated(decl) && create) {
    assert(decl->ir->irField == NULL);
    decl->ir->irField = new IrField(decl);
    decl->ir->m_type = IrDsymbol::FieldType;
  }
  assert(decl->ir->irField != NULL);
  return decl->ir->irField;
}

bool isIrFieldCreated(VarDeclaration *decl) {
  int t = decl->ir->type();
  assert(t == IrDsymbol::FieldType || t == IrDsymbol::NotSet);
  return t == IrDsymbol::FieldType;
}
