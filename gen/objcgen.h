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

struct ObjcSelector;
namespace llvm {
class Constant;
class GlobalVariable;
class Module;
class Triple;
}

// Forward decl.
class ClassDeclaration;
class FuncDeclaration;
class InterfaceDeclaration;
class VarDeclaration;

typedef llvm::StringMap<llvm::GlobalVariable *> SymbolCache;

bool objc_isSupported(const llvm::Triple &triple);

// Objective-C state tied to an LLVM module (object file).
class ObjCState {
public:
  ObjCState(llvm::Module &module) : module(module) {}

  // Classes
  llvm::GlobalVariable *getClassReference(const ClassDeclaration& cd);

  // Interface variables
  llvm::GlobalVariable *getIVarOffset(const ClassDeclaration& cd, const VarDeclaration& bar, bool outputSymbol);

  // Methods
  llvm::GlobalVariable *getMethodVarRef(const ObjcSelector &sel);
  llvm::GlobalVariable *getMethodVarName(const llvm::StringRef& name);
  llvm::GlobalVariable *getMethodVarType(const llvm::StringRef& ty);
  llvm::GlobalVariable *getMethodVarType(const FuncDeclaration& ty);

  // Protocols
  llvm::GlobalVariable *getProtocolSymbol(const InterfaceDeclaration& id);

  void finalize();

private:
  llvm::Module &module;

  // Symbols that shouldn't be optimized away
  std::vector<llvm::Constant *> retainedSymbols;

  /// Cache for `__OBJC_METACLASS_$_`/`__OBJC_CLASS_$_` symbols.
  SymbolCache classNameTable;
  SymbolCache classNameRoTable;

  /// Cache for `L_OBJC_CLASSLIST_REFERENCES_$_` symbols.
  SymbolCache classReferenceTable;

  /// Cache for `__OBJC_PROTOCOL_$_` symbols.
  SymbolCache protocolTable;

  // Cache for methods.
  SymbolCache methodVarNameTable;
  SymbolCache methodVarRefTable;
  SymbolCache methodVarTypeTable;

  // Cache for instance variables.
  SymbolCache ivarOffsetTable;

  llvm::GlobalVariable *getCStringVar(const char *symbol,
                                      const llvm::StringRef &str,
                                      const char *section);
  llvm::GlobalVariable *getClassName(const ClassDeclaration& cd, bool isMeta);

  void retain(llvm::Constant *sym);

  void genImageInfo();
  void retainSymbols();
};
