//===-- ir/irmodule.h - Codegen state for top-level D modules ---*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Represents the state of a D module on its way through code generation.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <list>

class FuncDeclaration;
class VarDeclaration;
class Module;
namespace llvm {
class GlobalVariable;
class Function;
class DIModule;
}

struct IrModule {
  IrModule(Module *module);
  virtual ~IrModule() = default;

  Module *const M = nullptr;

  llvm::GlobalVariable *moduleInfoSymbol();

  // static ctors/dtors/unittests
  using FuncDeclList = std::list<FuncDeclaration *>;
  using GatesList = std::list<VarDeclaration *>;
  FuncDeclList ctors;
  FuncDeclList dtors;
  FuncDeclList sharedCtors;
  FuncDeclList standaloneSharedCtors;
  FuncDeclList sharedDtors;
  GatesList gates;
  GatesList sharedGates;
  FuncDeclList unitTests;
  llvm::Function *coverageCtor = nullptr;

  llvm::DIModule *diModule = nullptr;

private:
  llvm::GlobalVariable *moduleInfoVar = nullptr;
};

IrModule *getIrModule(Module *m);
