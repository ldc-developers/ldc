//===-- irstate.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/irstate.h"

#include "dmd/declaration.h"
#include "dmd/expression.h"
#include "dmd/identifier.h"
#include "dmd/mtype.h"
#include "dmd/statement.h"
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
    : module(name, context), objc(module), DBuilder(this) {
  ir.state = this;
  mem.addRange(&inlineAsmLocs, sizeof(inlineAsmLocs));
}

IRState::~IRState() { mem.removeRange(&inlineAsmLocs); }

FuncGenState &IRState::funcGen() {
  assert(!funcGenStates.empty() && "Function stack is empty!");
  return *funcGenStates.back();
}

IrFunction *IRState::func() { return &funcGen().irFunc; }

llvm::Function *IRState::topfunc() { return func()->getLLVMFunc(); }

llvm::Instruction *IRState::topallocapoint() { return funcGen().allocapoint; }

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
  return funcGen().callOrInvoke(Callee, {}, Name);
}

LLCallSite IRState::CreateCallOrInvoke(LLValue *Callee,
                                       llvm::ArrayRef<LLValue *> Args,
                                       const char *Name, bool isNothrow) {
  return funcGen().callOrInvoke(Callee, Args, Name, isNothrow);
}

LLCallSite IRState::CreateCallOrInvoke(LLValue *Callee, LLValue *Arg1,
                                       const char *Name) {
  return funcGen().callOrInvoke(Callee, {Arg1}, Name);
}

LLCallSite IRState::CreateCallOrInvoke(LLValue *Callee, LLValue *Arg1,
                                       LLValue *Arg2, const char *Name) {
  return CreateCallOrInvoke(Callee, {Arg1, Arg2}, Name);
}

LLCallSite IRState::CreateCallOrInvoke(LLValue *Callee, LLValue *Arg1,
                                       LLValue *Arg2, LLValue *Arg3,
                                       const char *Name) {
  return CreateCallOrInvoke(Callee, {Arg1, Arg2, Arg3}, Name);
}

LLCallSite IRState::CreateCallOrInvoke(LLValue *Callee, LLValue *Arg1,
                                       LLValue *Arg2, LLValue *Arg3,
                                       LLValue *Arg4, const char *Name) {
  return CreateCallOrInvoke(Callee, {Arg1, Arg2, Arg3, Arg4}, Name);
}

bool IRState::isMainFunc(const IrFunction *func) const {
  assert(func != nullptr);
  return func->getLLVMFunc() == mainFunc;
}

bool IRState::emitArrayBoundsChecks() {
  if (global.params.useArrayBounds != CHECKENABLEsafeonly) {
    return global.params.useArrayBounds == CHECKENABLEon;
  }

  // Safe functions only.
  if (funcGenStates.empty()) {
    return false;
  }

  Type *t = func()->decl->type;
  return t->ty == Tfunction && ((TypeFunction *)t)->trust == TRUSTsafe;
}

LLConstant *IRState::setGlobalVarInitializer(LLGlobalVariable *&globalVar,
                                             LLConstant *initializer) {
  if (initializer->getType() == globalVar->getType()->getContainedType(0)) {
    globalVar->setInitializer(initializer);
    return globalVar;
  }

  // Create the global helper variable matching the initializer type.
  // It inherits most properties from the existing globalVar.
  auto globalHelperVar = new LLGlobalVariable(
      module, initializer->getType(), globalVar->isConstant(),
      globalVar->getLinkage(), initializer, "", nullptr,
      globalVar->getThreadLocalMode());
  globalHelperVar->setAlignment(LLMaybeAlign(globalVar->getAlignment()));
  globalHelperVar->setComdat(globalVar->getComdat());
  globalHelperVar->setDLLStorageClass(globalVar->getDLLStorageClass());
  globalHelperVar->setSection(globalVar->getSection());
  globalHelperVar->takeName(globalVar);

  // Replace all existing uses of globalVar by the bitcast pointer.
  auto castHelperVar = DtoBitCast(globalHelperVar, globalVar->getType());
  globalVar->replaceAllUsesWith(castHelperVar);

  // Register replacement for later occurrences of the original globalVar.
  globalsToReplace.emplace_back(globalVar, castHelperVar);

  // Reset globalVar to the helper variable.
  globalVar = globalHelperVar;

  return castHelperVar;
}

void IRState::replaceGlobals() {
  for (const auto &pair : globalsToReplace) {
    pair.first->replaceAllUsesWith(pair.second);
    pair.first->eraseFromParent();
  }

  globalsToReplace.resize(0);
}

////////////////////////////////////////////////////////////////////////////////

LLConstant *IRState::getStructLiteralConstant(StructLiteralExp *sle) const {
  return static_cast<LLConstant *>(structLiteralConstants.lookup(sle->origin));
}

void IRState::setStructLiteralConstant(StructLiteralExp *sle,
                                       LLConstant *constant) {
  structLiteralConstants[sle->origin] = constant;
}

////////////////////////////////////////////////////////////////////////////////

void IRState::addLinkerOption(llvm::ArrayRef<llvm::StringRef> options) {
  llvm::SmallVector<llvm::Metadata *, 2> mdStrings;
  mdStrings.reserve(options.size());
  for (const auto &s : options)
    mdStrings.push_back(llvm::MDString::get(context(), s));
  linkerOptions.push_back(llvm::MDNode::get(context(), mdStrings));
}

void IRState::addLinkerDependentLib(llvm::StringRef libraryName) {
  auto n = llvm::MDString::get(context(), libraryName);
  linkerDependentLibs.push_back(llvm::MDNode::get(context(), n));
}

////////////////////////////////////////////////////////////////////////////////

void IRState::addInlineAsmSrcLoc(const Loc &loc,
                                 llvm::CallInst *inlineAsmCall) {
  // Simply use a stack of Loc* per IR module, and use index+1 as 32-bit
  // cookie to be mapped back by the InlineAsmDiagnosticHandler.
  // 0 is not a valid cookie.
  inlineAsmLocs.push_back(loc);
  auto srcLocCookie = static_cast<unsigned>(inlineAsmLocs.size());

  auto constant =
      LLConstantInt::get(LLType::getInt32Ty(context()), srcLocCookie);
  inlineAsmCall->setMetadata(
      "srcloc",
      llvm::MDNode::get(context(), llvm::ConstantAsMetadata::get(constant)));
}

const Loc &IRState::getInlineAsmSrcLoc(unsigned srcLocCookie) const {
  assert(srcLocCookie > 0 && srcLocCookie <= inlineAsmLocs.size());
  return inlineAsmLocs[srcLocCookie - 1];
}

////////////////////////////////////////////////////////////////////////////////

IRBuilder<> *IRBuilderHelper::operator->() {
  IRBuilder<> &b = state->scope().builder;
  assert(b.GetInsertBlock() != NULL);
  return &b;
}

////////////////////////////////////////////////////////////////////////////////

bool useMSVCEH() {
  return global.params.targetTriple->isWindowsMSVCEnvironment();
}
