//===-- gen/objcgen.cpp -----------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Functions for generating Objective-C method calls.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <vector>
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Type.h"

struct ObjcSelector;
namespace llvm {
class Constant;
class GlobalVariable;
class Module;
class Triple;
}

bool objc_isSupported(const llvm::Triple &triple);

// Objective-C state tied to an LLVM module (object file).
class ObjCState {
public:
  ObjCState(llvm::Module &module);

  llvm::GlobalVariable *getMethVarRef(const ObjcSelector &sel);
  void finalize();

private:
  llvm::Module &module;

  llvm::Type* _class_t;
  llvm::Type* _objc_cache;
  llvm::Type* _class_ro_t;
  llvm::Type* __method_list_t;
  llvm::Type* _objc_method;
  llvm::Type* _objc_protocol_list;
  llvm::Type* _protocol_t;
  llvm::Type* _ivar_list_t;
  llvm::Type* _ivar_t;
  llvm::Type* _prop_list_t;
  llvm::Type* _prop_t;

  // symbols that shouldn't be optimized away
  std::vector<llvm::Constant *> retainedSymbols;

  llvm::StringMap<llvm::GlobalVariable *> methVarNameMap;
  llvm::StringMap<llvm::GlobalVariable *> methVarRefMap;

  llvm::GlobalVariable *getCStringVar(const char *symbol,
                                      const llvm::StringRef &str,
                                      const char *section);
  llvm::GlobalVariable *getMethVarName(const llvm::StringRef &name);
  void retain(llvm::Constant *sym);

  void genImageInfo();
  void retainSymbols();
};
