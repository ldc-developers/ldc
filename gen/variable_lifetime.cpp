//===-- gen/variable_lifetime.cpp - -----------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Codegen for local variable lifetime: llvm.lifetime.start abd
// llvm.lifetime.end.
//
//===----------------------------------------------------------------------===//

#include "gen/variable_lifetime.h"

#include "driver/cl_options.h"
#include "gen/irstate.h"

#include <vector>
#include <utility>

// TODO: make this option depend on -O and -fsanitize settings.
static llvm::cl::opt<bool> fEmitLocalVarLifetime(
    "femit-local-var-lifetime",
    llvm::cl::desc(
        "Emit local variable lifetime, enabling more optimizations."),
    llvm::cl::Hidden, llvm::cl::ZeroOrMore);

LocalVariableLifetimeAnnotator::LocalVariableLifetimeAnnotator(IRState &irs)
    : irs(irs) {
  allocaType =
      llvm::Type::getInt8Ty(irs.context())
          ->getPointerTo(irs.module.getDataLayout().getAllocaAddrSpace());
}

void LocalVariableLifetimeAnnotator::pushScope() { scopes.emplace_back(); }

void LocalVariableLifetimeAnnotator::addLocalVariable(llvm::Value *address,
                                                      llvm::Value *size) {
  assert(address);
  assert(size);

  if (!fEmitLocalVarLifetime)
    return;

  if (scopes.empty())
    return;

  // Push to scopes
  scopes.back().variables.emplace_back(size, address);

  // Emit lifetime start
  address = irs.ir->CreateBitCast(address, allocaType);
  irs.CreateCallOrInvoke(getLLVMLifetimeStartFn(), {size, address}, "",
                         true /*nothrow*/);
}

// Emits end-of-lifetime annotation for all variables in current scope.
void LocalVariableLifetimeAnnotator::popScope() {
  if (scopes.empty())
    return;

  for (const auto &var : scopes.back().variables) {
    auto size = var.first;
    auto address = var.second;

    address = irs.ir->CreateBitCast(address, allocaType);
    assert(address);

    irs.CreateCallOrInvoke(getLLVMLifetimeEndFn(), {size, address}, "",
                           true /*nothrow*/);
  }
  scopes.pop_back();
}

/// Lazily declare the @llvm.lifetime.start intrinsic.
llvm::Function *LocalVariableLifetimeAnnotator::getLLVMLifetimeStartFn() {
  if (lifetimeStartFunction)
    return lifetimeStartFunction;

  lifetimeStartFunction = llvm::Intrinsic::getDeclaration(
      &irs.module, llvm::Intrinsic::lifetime_start, allocaType);
  assert(lifetimeStartFunction);
  return lifetimeStartFunction;
}

/// Lazily declare the @llvm.lifetime.end intrinsic.
llvm::Function *LocalVariableLifetimeAnnotator::getLLVMLifetimeEndFn() {
  if (lifetimeEndFunction)
    return lifetimeEndFunction;

  lifetimeEndFunction = llvm::Intrinsic::getDeclaration(
      &irs.module, llvm::Intrinsic::lifetime_end, allocaType);
  assert(lifetimeEndFunction);
  return lifetimeEndFunction;
}
