//===-- irfunction.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irfunction.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "ir/irdsymbol.h"

IrFunction::IrFunction(FuncDeclaration *fd) : FMF() {
  decl = fd;

  Type *t = fd->type->toBasetype();
  assert(t->ty == Tfunction);
  type = static_cast<TypeFunction *>(t);
}

void IrFunction::setNeverInline() {
  assert(!func->getAttributes().hasAttribute(llvm::AttributeSet::FunctionIndex,
                                             llvm::Attribute::AlwaysInline) &&
         "function can't be never- and always-inline at the same time");
  func->addFnAttr(llvm::Attribute::NoInline);
}

void IrFunction::setAlwaysInline() {
  assert(!func->getAttributes().hasAttribute(llvm::AttributeSet::FunctionIndex,
                                             llvm::Attribute::NoInline) &&
         "function can't be never- and always-inline at the same time");
  func->addFnAttr(llvm::Attribute::AlwaysInline);
}

IrFunction *getIrFunc(FuncDeclaration *decl, bool create) {
  if (!isIrFuncCreated(decl) && create) {
    assert(decl->ir->irFunc == NULL);
    decl->ir->irFunc = new IrFunction(decl);
    decl->ir->m_type = IrDsymbol::FuncType;
  }
  assert(decl->ir->irFunc != NULL);
  return decl->ir->irFunc;
}

bool isIrFuncCreated(FuncDeclaration *decl) {
  assert(decl);
  assert(decl->ir);
  IrDsymbol::Type t = decl->ir->type();
  assert(t == IrDsymbol::FuncType || t == IrDsymbol::NotSet);
  return t == IrDsymbol::FuncType;
}
