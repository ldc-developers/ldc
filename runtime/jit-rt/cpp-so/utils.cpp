//===-- utils.cpp ---------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "utils.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include "context.h"

void fatal(const Context &context, const std::string &reason) {
  if (nullptr != context.fatalHandler) {
    context.fatalHandler(context.fatalHandlerData, reason.c_str());
  } else {
    fprintf(stderr, "Runtime compiler fatal: %s\n", reason.c_str());
    fflush(stderr);
    abort();
  }
}

void interruptPoint(const Context &context, const char *desc,
                    const char *object) {
  assert(nullptr != desc);
  if (nullptr != context.interruptPointHandler) {
    context.interruptPointHandler(context.interruptPointHandlerData, desc,
                                  object);
  }
}

void verifyModule(const Context &context, llvm::Module &module) {
  std::string err;
  llvm::raw_string_ostream errstream(err);
  if (llvm::verifyModule(module, &errstream)) {
    std::string desc =
        std::string("module verification failed:") + errstream.str();
    fatal(context, desc);
  }
}

void createModuleCtorsWrapper(const Context &context, llvm::Module &module,
                              const std::string &wrapperName) {
  assert(!wrapperName.empty());
  auto ctorsVar = module.getGlobalVariable("llvm.global_ctors");
  if (ctorsVar == nullptr || ctorsVar->isDeclaration()) {
    return;
  }

  auto &llcontext = module.getContext();
  auto funcType = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(llcontext), false);
  auto func = llvm::Function::Create(funcType,
                                     llvm::GlobalValue::ExternalLinkage,
                                     wrapperName, &module);

  auto bb = llvm::BasicBlock::Create(llcontext, "", func);
  llvm::IRBuilder<> builder(llcontext);
  builder.SetInsertPoint(bb);

  // Should be an array of '{ i32, void ()* }' structs.  The first value is
  // the init priority, which we ignore.
  auto InitList = llvm::dyn_cast<llvm::ConstantArray>(ctorsVar->getInitializer());
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i) {
    auto CS = llvm::dyn_cast<llvm::ConstantStruct>(InitList->getOperand(i));
    if (CS == nullptr) continue;

    auto FP = CS->getOperand(1);
    if (FP->isNullValue()) {
      continue;  // Found a sentinal value, ignore.
    }

    // Strip off constant expression casts.
    if (auto CE = llvm::dyn_cast<llvm::ConstantExpr>(FP)) {
      if (CE->isCast()) {
        FP = CE->getOperand(0);
      }
    }

    // Execute the ctor/dtor function!
    if (auto F = llvm::dyn_cast<llvm::Function>(FP)) {
      builder.CreateCall(F);
    }
  }
  builder.CreateRetVoid();

  ctorsVar->eraseFromParent();
}
