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

    if (!define)
      define = defineOnDeclare(V, /*isFunction=*/false);
  }

  if (define) {
    if (V->storage_class & STCextern) {
      // external
    } else if (!gIR->funcGenStates.empty() &&
               gIR->topfunc()->getLinkage() ==
                   LLGlobalValue::AvailableExternallyLinkage) {
      // don't define globals while codegen'ing available_externally functions
    } else {
      auto gvar = llvm::dyn_cast<LLGlobalVariable>(value);
      const bool isDefined = !gvar // bitcast pointer to a helper global
                             || gvar->hasInitializer();
      if (!isDefined)
        this->define();
    }
  }

  return value;
}

llvm::Type *IrGlobal::getType() {
  return llvm::dyn_cast<llvm::GlobalVariable>(value)->getValueType();
}
void IrGlobal::declare() {
  Logger::println("Declaring global: %s", V->toChars());
  LOG_SCOPE

  IF_LOG {
    if (V->parent) {
      Logger::println("parent: %s (%s)", V->parent->toChars(),
                      V->parent->kind());
    } else {
      Logger::println("parent: null");
    }
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

  // Windows: for globals with `export` visibility, initialize the DLL storage
  // class with dllimport unless the variable is defined in a root module
  // (=> no extra indirection for other root modules, assuming *all* root
  // modules will be linked together to one or more binaries).
  // [Defining a global overrides its DLL storage class.]
  bool useDLLImport = false;
  if (global.params.targetTriple->isOSWindows()) {
    // dllimport isn't supported for thread-local globals (MSVC++ neither)
    if (!V->isThreadlocal()) {
      // implicitly include extern(D) globals with -dllimport
      useDLLImport =
          (V->isExport() || V->_linkage == LINK::d) && dllimportDataSymbol(V);
    }
  }

  // Since the type of a global must exactly match the type of its
  // initializer, we cannot know the type until after we have emitted the
  // latter (e.g. in case of unions, …). However, it is legal for the
  // initializer to refer to the address of the variable. Thus, we first
  // create a global with the generic type (note the assignment to
  // value!), and in case we also do an initializer with a different type
  // later, swap it out and replace any existing uses with bitcasts to the
  // previous type.

  LLGlobalVariable *gvar =
      declareGlobal(V->loc, gIR->module, DtoMemType(V->type), irMangle,
                    isLLConst, V->isThreadlocal(), useDLLImport);
  value = gvar;

  if (V->llvmInternal == LLVMextern_weak)
    gvar->setLinkage(llvm::GlobalValue::ExternalWeakLinkage);

  // Set the alignment (it is important not to use type->alignsize because
  // VarDeclarations can have an align() attribute independent of the type
  // as well).
  gvar->setAlignment(llvm::MaybeAlign(DtoAlignment(V)));

  applyVarDeclUDAs(V, gvar);

  if (dynamicCompileConst)
    addDynamicCompiledVar(gIR, this);

  IF_LOG Logger::cout() << *gvar << '\n';
}

void IrGlobal::define() {
  Logger::println("Defining global: %s", V->toChars());
  LOG_SCOPE

  if (global.params.vtls && V->isThreadlocal() &&
      !(V->storage_class & STCtemp)) {
    message("%s: `%s` is thread local", V->loc.toChars(), V->toChars());
  }

  LLConstant *initVal =
      DtoConstInitializer(V->loc, V->type, V->_init, V->isCsymbol());

  // Set the initializer, swapping out the variable if the types do not
  // match.
  auto gvar = llvm::cast<LLGlobalVariable>(value);
  value = gIR->setGlobalVarInitializer(gvar, initVal, V);

  // dllexport isn't supported for thread-local globals (MSVC++ neither);
  // don't let LLVM create a useless /EXPORT directive (yields the same linker
  // error anyway when trying to dllimport).
  if (gvar->hasDLLExportStorageClass() && V->isThreadlocal())
    gvar->setDLLStorageClass(LLGlobalValue::DefaultStorageClass);

  // If this global is used from a naked function, we need to create an
  // artificial "use" for it, or it could be removed by the optimizer if
  // the only reference to it is in inline asm.
  // Also prevent linker-level dead-symbol-elimination from stripping
  // special `rt_*` druntime symbol overrides (e.g., from executables linked
  // against *shared* druntime; required at least for Apple's ld64 linker).
  const auto name = gvar->getName();
  if (nakedUse || name == "rt_options" || name == "rt_envvars_enabled" ||
      name == "rt_cmdline_enabled") {
    gIR->usedArray.push_back(gvar);
  }

  // Also set up the debug info.
  gIR->DBuilder.EmitGlobalVariable(gvar, V);

  IF_LOG Logger::cout() << *gvar << '\n';
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
