//===-- gen/variable_lifetime.h - -------------------------------*- C++ -*-===//
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

#pragma once

#include <vector>
#include <utility>

namespace llvm {
class Function;
class Type;
class Value;
}
struct IRState;

struct LocalVariableLifetimeAnnotator {
  struct LocalVariableScope {
    std::vector<std::pair<llvm::Value *, llvm::Value *>> variables;
  };
  /// Stack of scopes, each scope can have multiple variables.
  std::vector<LocalVariableScope> scopes;

  /// Cache the llvm types and intrinsics used for codegen.
  llvm::Function *lifetimeStartFunction = nullptr;
  llvm::Function *lifetimeEndFunction = nullptr;
  llvm::Type *allocaType = nullptr;

  llvm::Function *getLLVMLifetimeStartFn();
  llvm::Function *getLLVMLifetimeEndFn();

  IRState &irs;

public:
  LocalVariableLifetimeAnnotator(IRState &irs);

  /// Opens a new scope.
  void pushScope();

  /// Closes current scope and emits end-of-lifetime annotation for all
  /// variables in current scope.
  void popScope();

  /// Register a new local variable for lifetime annotation.
  void addLocalVariable(llvm::Value *address, llvm::Value *size);
};
