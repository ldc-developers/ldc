//===-- irstate.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/irstate.h"
#include "declaration.h"
#include "mtype.h"
#include "statement.h"
#include "gen/funcgenstate.h"
#include "gen/llvm.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include <cstdarg>

IRState *gIR = nullptr;
llvm::TargetMachine *gTargetMachine = nullptr;
const llvm::DataLayout *gDataLayout = nullptr;
TargetABI *gABI = nullptr;

////////////////////////////////////////////////////////////////////////////////
IRScope::IRScope() : builder(gIR->context()) { begin = nullptr; }

IRScope::IRScope(llvm::BasicBlock *b) : begin(b), builder(b) {}

IRScope &IRScope::operator=(const IRScope &rhs) {
  begin = rhs.begin;
  builder.SetInsertPoint(begin);
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
IRState::IRState(const char *name, llvm::LLVMContext &context)
    : module(name, context), DBuilder(this) {
  mutexType = nullptr;
  moduleRefType = nullptr;

  dmodule = nullptr;
  mainFunc = nullptr;
  ir.state = this;
  asmBlock = nullptr;
}

IRState::~IRState() {}

FuncGenState &IRState::funcGen() {
  assert(!funcGenStates.empty() && "Function stack is empty!");
  return *funcGenStates.back();
}

IrFunction *IRState::func() {
  return &funcGen().irFunc;
}

llvm::Function *IRState::topfunc() {
  return func()->func;
}

llvm::Instruction *IRState::topallocapoint() {
  return funcGen().allocapoint;
}

IRScope &IRState::scope() {
  assert(!scopes.empty());
  return scopes.back();
}

llvm::BasicBlock *IRState::scopebb() {
  IRScope &s = scope();
  assert(s.begin);
  return s.begin;
}

bool IRState::scopereturned() {
  // return scope().returned;
  return !scopebb()->empty() && scopebb()->back().isTerminator();
}

llvm::BasicBlock *IRState::insertBBBefore(llvm::BasicBlock *successor,
                                          const llvm::Twine &name) {
  return llvm::BasicBlock::Create(context(), name, topfunc(), successor);
}

llvm::BasicBlock *IRState::insertBBAfter(llvm::BasicBlock *predecessor,
                                         const llvm::Twine &name) {
  auto bb = llvm::BasicBlock::Create(context(), name, topfunc());
  bb->moveAfter(predecessor);
  return bb;
}

llvm::BasicBlock *IRState::insertBB(const llvm::Twine &name) {
  return insertBBAfter(scopebb(), name);
}

LLCallSite IRState::CreateCallOrInvoke(LLValue *Callee, const char *Name) {
  LLSmallVector<LLValue *, 1> args;
  return funcGen().callOrInvoke(Callee, args, Name);
}

LLCallSite IRState::CreateCallOrInvoke(LLValue *Callee, LLValue *Arg1,
                                       const char *Name) {
  LLValue *args[] = {Arg1};
  return funcGen().callOrInvoke(Callee, args, Name);
}

LLCallSite IRState::CreateCallOrInvoke(LLValue *Callee, LLValue *Arg1,
                                       LLValue *Arg2, const char *Name) {
  LLValue *args[] = {Arg1, Arg2};
  return funcGen().callOrInvoke(Callee, args, Name);
}

LLCallSite IRState::CreateCallOrInvoke(LLValue *Callee, LLValue *Arg1,
                                       LLValue *Arg2, LLValue *Arg3,
                                       const char *Name) {
  LLValue *args[] = {Arg1, Arg2, Arg3};
  return funcGen().callOrInvoke(Callee, args, Name);
}

LLCallSite IRState::CreateCallOrInvoke(LLValue *Callee, LLValue *Arg1,
                                       LLValue *Arg2, LLValue *Arg3,
                                       LLValue *Arg4, const char *Name) {
  LLValue *args[] = {Arg1, Arg2, Arg3, Arg4};
  return funcGen().callOrInvoke(Callee, args, Name);
}

bool IRState::emitArrayBoundsChecks() {
  if (global.params.useArrayBounds != BOUNDSCHECKsafeonly) {
    return global.params.useArrayBounds == BOUNDSCHECKon;
  }

  // Safe functions only.
  if (funcGenStates.empty()) {
    return false;
  }

  Type *t = func()->decl->type;
  return t->ty == Tfunction && ((TypeFunction *)t)->trust == TRUSTsafe;
}

////////////////////////////////////////////////////////////////////////////////

IRBuilder<> *IRBuilderHelper::operator->() {
  IRBuilder<> &b = state->scope().builder;
  assert(b.GetInsertBlock() != NULL);
  return &b;
}

////////////////////////////////////////////////////////////////////////////////

bool useMSVCEH() {
#if LDC_LLVM_VER >= 308
  return global.params.targetTriple->isWindowsMSVCEnvironment();
#else
  return false;
#endif
}
