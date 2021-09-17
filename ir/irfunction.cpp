//===-- irfunction.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irfunction.h"

#include "driver/cl_options.h"
#include "gen/functions.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "ir/irdsymbol.h"

IrFunction::IrFunction(FuncDeclaration *fd)
    : irFty(nullptr /*set immediately below*/), FMF(opts::defaultFMF) {
  decl = fd;

  Type *t = fd->type->toBasetype();
  assert(t->ty == TY::Tfunction);
  type = static_cast<TypeFunction *>(t);

  irFty.type = type;
}

void IrFunction::setNeverInline() {
  assert(!func->hasFnAttribute(llvm::Attribute::AlwaysInline) &&
         "function can't be never- and always-inline at the same time");
  func->addFnAttr(llvm::Attribute::NoInline);
}

void IrFunction::setAlwaysInline() {
  assert(!func->hasFnAttribute(llvm::Attribute::NoInline) &&
         "function can't be never- and always-inline at the same time");
  func->addFnAttr(llvm::Attribute::AlwaysInline);
}

void IrFunction::setLLVMFunc(llvm::Function *function) {
  assert(function != nullptr);
  func = function;
}

llvm::Function *IrFunction::getLLVMFunc() const {
  return func;
}

llvm::CallingConv::ID IrFunction::getCallingConv() const {
  assert(func != nullptr);
  return func->getCallingConv();
}

llvm::FunctionType *IrFunction::getLLVMFuncType() const {
  assert(func != nullptr);
  return func->getFunctionType();
}

bool IrFunction::hasLLVMPersonalityFn() const {
  assert(func != nullptr);
  return func->hasPersonalityFn();
}

void IrFunction::setLLVMPersonalityFn(llvm::Constant *personality) {
  assert(func != nullptr);
  func->setPersonalityFn(personality);
}

llvm::StringRef IrFunction::getLLVMFuncName() const {
  assert(func != nullptr);
  return func->getName();
}

llvm::Function *IrFunction::getLLVMCallee() const {
  assert(func != nullptr);
  return rtCompileFunc != nullptr ? rtCompileFunc : func;
}

bool IrFunction::isDynamicCompiled() const {
  return dynamicCompile || dynamicCompileEmit;
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

llvm::Function *DtoCallee(FuncDeclaration *decl, bool create) {
  assert(decl != nullptr);
  if (create) {
    DtoDeclareFunction(decl);
  }
  return getIrFunc(decl)->getLLVMCallee();
}
