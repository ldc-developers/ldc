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
#include "ir/irfunction.h"

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
  llvm::GlobalVariable *getClassSymbol(const ClassDeclaration& cd, bool meta);
  llvm::GlobalVariable *getClassRoSymbol(const ClassDeclaration& cd, bool meta);
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
  llvm::GlobalVariable *getProtocolReference(const InterfaceDeclaration& id);

  void finalize();

private:
  llvm::Module &module;

  // Symbols that shouldn't be optimized away
  std::vector<llvm::Constant *> retainedSymbols;

  // Store the classes and protocols.
  std::vector<ClassDeclaration *> classes;
  std::vector<InterfaceDeclaration *> protocols;

  /// Cache for `__OBJC_METACLASS_$_`/`__OBJC_CLASS_$_` symbols.
  SymbolCache classNameTable;
  SymbolCache classNameRoTable;

  /// Cache for `_OBJC_CLASS_$_` symbols stored in `__objc_stubs`.
  /// NOTE: Stub classes have a different layout from normal classes
  /// And need to be instantiated with a call to the objective-c runtime.
  SymbolCache stubClassNameTable;

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

  // Gets Objective-C type tag
  const char *getObjcType(Type *t);

  llvm::Constant *constU32(uint32_t value);
  llvm::Constant *constU64(uint64_t value);
  llvm::Constant *constSizeT(size_t value);

  llvm::GlobalVariable *getProtocoList(const InterfaceDeclaration& iface);
  llvm::GlobalVariable *getProtocoList(const ClassDeclaration& cd);
  llvm::GlobalVariable *getMethodListFor(const ClassDeclaration& cd, bool meta);

  llvm::GlobalVariable *getCStringVar(const char *symbol,
                                      const llvm::StringRef &str,
                                      const char *section);

  void retain(llvm::Constant *sym);

  void genImageInfo();
  void retainSymbols();
};
