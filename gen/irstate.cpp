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
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "llvm/IR/InlineAsm.h"
#include <cstdarg>

IRState *gIR = nullptr;
llvm::TargetMachine *gTargetMachine = nullptr;
const llvm::DataLayout *gDataLayout = nullptr;
TargetABI *gABI = nullptr;

////////////////////////////////////////////////////////////////////////////////
IRState::IRState(const char *name, llvm::LLVMContext &context)
    : builder(context), module(name, context), objc(module), DBuilder(this) {
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

std::unique_ptr<IRBuilderScope> IRState::setInsertPoint(llvm::BasicBlock *bb) {
  auto savedScope = std::make_unique<IRBuilderScope>(builder);
  builder.SetInsertPoint(bb);
  return savedScope;
}

std::unique_ptr<llvm::IRBuilderBase::InsertPointGuard>
IRState::saveInsertPoint() {
  return std::make_unique<llvm::IRBuilderBase::InsertPointGuard>(builder);
}

bool IRState::scopereturned() {
  auto bb = scopebb();
  return !bb->empty() && bb->back().isTerminator();
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

llvm::Instruction *IRState::CreateCallOrInvoke(LLFunction *Callee,
                                               const char *Name) {
  return CreateCallOrInvoke(Callee, {}, Name);
}

llvm::Instruction *IRState::CreateCallOrInvoke(LLFunction *Callee,
                                               llvm::ArrayRef<LLValue *> Args,
                                               const char *Name,
                                               bool isNothrow) {
  return funcGen().callOrInvoke(Callee, Callee->getFunctionType(), Args, Name,
                                isNothrow);
}

llvm::Instruction *IRState::CreateCallOrInvoke(LLFunction *Callee,
                                               LLValue *Arg1,
                                               const char *Name) {
  return CreateCallOrInvoke(Callee, llvm::ArrayRef<LLValue *>(Arg1), Name);
}

llvm::Instruction *IRState::CreateCallOrInvoke(LLFunction *Callee,
                                               LLValue *Arg1, LLValue *Arg2,
                                               const char *Name) {
  return CreateCallOrInvoke(Callee, {Arg1, Arg2}, Name);
}

llvm::Instruction *IRState::CreateCallOrInvoke(LLFunction *Callee,
                                               LLValue *Arg1, LLValue *Arg2,
                                               LLValue *Arg3,
                                               const char *Name) {
  return CreateCallOrInvoke(Callee, {Arg1, Arg2, Arg3}, Name);
}

llvm::Instruction *IRState::CreateCallOrInvoke(LLFunction *Callee,
                                               LLValue *Arg1, LLValue *Arg2,
                                               LLValue *Arg3, LLValue *Arg4,
                                               const char *Name) {
  return CreateCallOrInvoke(Callee, {Arg1, Arg2, Arg3, Arg4}, Name);
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
  return t->ty == TY::Tfunction && ((TypeFunction *)t)->trust == TRUST::safe;
}

LLConstant *
IRState::setGlobalVarInitializer(LLGlobalVariable *&globalVar,
                                 LLConstant *initializer,
                                 Dsymbol *symbolForLinkageAndVisibility) {
  if (initializer->getType() == globalVar->getValueType()) {
    defineGlobal(globalVar, initializer, symbolForLinkageAndVisibility);
    return globalVar;
  }

  // Create the global helper variable matching the initializer type.
  // It inherits most properties from the existing globalVar.
  auto globalHelperVar = new LLGlobalVariable(
      module, initializer->getType(), globalVar->isConstant(),
      globalVar->getLinkage(), nullptr, "", nullptr,
      globalVar->getThreadLocalMode());
  globalHelperVar->setAlignment(llvm::MaybeAlign(globalVar->getAlignment()));
  globalHelperVar->setComdat(globalVar->getComdat());
  globalHelperVar->setDLLStorageClass(globalVar->getDLLStorageClass());
  globalHelperVar->setSection(globalVar->getSection());
  globalHelperVar->takeName(globalVar);

  defineGlobal(globalHelperVar, initializer, symbolForLinkageAndVisibility);

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

namespace {
template <typename F>
LLGlobalVariable *
getCachedStringLiteralImpl(llvm::Module &module,
                           llvm::StringMap<LLGlobalVariable *> &cache,
                           llvm::StringRef key, F initFactory) {
  auto iter = cache.find(key);
  if (iter != cache.end()) {
    return iter->second;
  }

  LLConstant *constant = initFactory();

  auto gvar =
      new LLGlobalVariable(module, constant->getType(), true,
                           LLGlobalValue::PrivateLinkage, constant, ".str");
  gvar->setUnnamedAddr(LLGlobalValue::UnnamedAddr::Global);

  cache[key] = gvar;

  return gvar;
}
}

LLGlobalVariable *IRState::getCachedStringLiteral(StringExp *se) {
  llvm::StringMap<LLGlobalVariable *> *cache;
  switch (se->sz) {
  default:
    llvm_unreachable("Unknown char type");
  case 1:
    cache = &cachedStringLiterals;
    break;
  case 2:
    cache = &cachedWstringLiterals;
    break;
  case 4:
    cache = &cachedDstringLiterals;
    break;
  }

  const DArray<const unsigned char> keyData = se->peekData();
  const llvm::StringRef key(reinterpret_cast<const char *>(keyData.ptr),
                            keyData.length);

  return getCachedStringLiteralImpl(module, *cache, key, [se]() {
    // null-terminate
    return buildStringLiteralConstant(se, se->numberOfCodeUnits() + 1);
  });
}

LLGlobalVariable *IRState::getCachedStringLiteral(llvm::StringRef s) {
  return getCachedStringLiteralImpl(module, cachedStringLiterals, s, [&]() {
    return llvm::ConstantDataArray::getString(context(), s, true);
  });
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

llvm::CallInst *
IRState::createInlineAsmCall(const Loc &loc, llvm::InlineAsm *ia,
                             llvm::ArrayRef<llvm::Value *> args,
                             llvm::ArrayRef<llvm::Type *> indirectTypes) {
  llvm::CallInst *call = ir->CreateCall(ia, args);
  addInlineAsmSrcLoc(loc, call);

#if LDC_LLVM_VER >= 1400
  // a non-indirect output constraint (=> return value of call) shifts the
  // constraint/argument index mapping
  ptrdiff_t i = call->getType()->isVoidTy() ? 0 : -1;
  size_t indirectIdx = 0;
    
  for (const auto &constraintInfo : ia->ParseConstraints()) {
    if (constraintInfo.isIndirect) {
      call->addParamAttr(i, llvm::Attribute::get(
                                context(),
                                llvm::Attribute::ElementType,
                                indirectTypes[indirectIdx]));
      ++indirectIdx;
    }
    ++i;
  }
#endif

  return call;
}

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
  IRBuilder<> &b = state->builder;
  assert(b.GetInsertBlock());
  return &b;
}

////////////////////////////////////////////////////////////////////////////////

bool useMSVCEH() {
  return global.params.targetTriple->isWindowsMSVCEnvironment();
}
