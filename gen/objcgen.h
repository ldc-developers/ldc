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
#include <unordered_map>
#include "llvm/ADT/StringMap.h"
#include "gen/tollvm.h"
#include "dmd/mtype.h"
#include "dmd/errors.h"

struct ObjcSelector;
namespace llvm {
class Constant;
class GlobalVariable;
class Module;
class Triple;
}

// Forward decl.
class Declaration;
class ClassDeclaration;
class FuncDeclaration;
class InterfaceDeclaration;
class VarDeclaration;
class Identifier;
class Type;

// Fwd declaration.
class ObjCState;

// class is a metaclass
#define RO_META               (1<<0)

// class is a root class
#define RO_ROOT               (1<<1)

// Section Names
#define OBJC_SECNAME_CLASSNAME             "__TEXT,__objc_classname, cstring_literals"
#define OBJC_SECNAME_METHNAME              "__TEXT,__objc_methname, cstring_literals"
#define OBJC_SECNAME_METHTYPE              "__TEXT,__objc_methtype, cstring_literals"
#define OBJC_SECNAME_SELREFS               "__DATA,__objc_selrefs, literal_pointers, no_dead_strip"
#define OBJC_SECNAME_IMAGEINFO             "__DATA,__objc_imageinfo, regular, no_dead_strip"
#define OBJC_SECNAME_CLASSREFS             "__DATA,__objc_classrefs, regular, no_dead_strip"
#define OBJC_SECNAME_CLASSLIST             "__DATA,__objc_classlist, regular, no_dead_strip"
#define OBJC_SECNAME_STUBS                 "__DATA,__objc_stubs, regular, no_dead_strip"
#define OBJC_SECNAME_CATLIST               "__DATA,__objc_catlist, regular, no_dead_strip"
#define OBJC_SECNAME_PROTOLIST             "__DATA,__objc_protolist, coalesced, no_dead_strip"
#define OBJC_SECNAME_PROTOREFS             "__DATA,__objc_protorefs, regular"
#define OBJC_SECNAME_CONST                 "__DATA,__objc_const"
#define OBJC_SECNAME_DATA                  "__DATA,__objc_data"
#define OBJC_SECNAME_IVAR                  "__DATA,__objc_ivar"

// Names of Objective-C runtime structs
#define OBJC_STRUCTNAME_CLASSRO   "class_ro_t"
#define OBJC_STRUCTNAME_CLASS     "class_t"
#define OBJC_STRUCTNAME_STUBCLASS "stub_class_t"
#define OBJC_STRUCTNAME_PROTO     "protocol_t"
#define OBJC_STRUCTNAME_IVAR      "ivar_t"
#define OBJC_STRUCTNAME_METHOD    "objc_method"

#define ObjcList std::vector
#define ObjcMap std::unordered_map

// Gets whether Objective-C is supported.
bool objc_isSupported(const llvm::Triple &triple);

// Generate name strings
std::string objcGetClassRoSymbol(const char *name, bool meta);
std::string objcGetClassSymbol(const char *name, bool meta);
std::string objcGetClassLabelSymbol(const char *name);
std::string objcGetClassMethodListSymbol(const char *className, bool meta);
std::string objcGetIvarListSymbol(const char *className);
std::string objcGetIvarSymbol(const char *className, const char *varName);
std::string objcGetProtoMethodListSymbol(const char *className, bool meta, bool optional);
std::string objcGetProtoSymbol(const char *name);
std::string objcGetProtoListSymbol(const char *name);
std::string objcGetSymbolName(const char *dsymPrefix, const char *dsymName);

// Utility which fetches the appropriate Objective-C
// name for a declaration.
const char *objcResolveName(Dsymbol *decl);

// Gets the Objective-C type encoding for D type t 
std::string objcGetTypeEncoding(Type *t);

// class_t
LLStructType *objcGetClassType(const llvm::Module& module);

// class_ro_t
LLStructType *objcGetClassRoType(const llvm::Module& module);

// stub_class_t
LLStructType *objcGetStubClassType(const llvm::Module& module);

// objc_method
LLStructType *objcGetMethodType(const llvm::Module& module);

// ivar_t
LLStructType *objcGetIvarType(const llvm::Module& module);

// protocol_t
LLStructType *objcGetProtocolType(const llvm::Module& module);

// xyz_list_t (count-only)
LLConstant *objcEmitList(llvm::Module &module, LLConstantList objects, bool alignSizeT = false, bool countOnly = false);

struct ObjcClassInfo {
  ClassDeclaration *decl;
  LLGlobalVariable *ref;

  LLGlobalVariable *name;
  LLGlobalVariable *table;
};

struct ObjcProtocolInfo {
  InterfaceDeclaration *decl;
  LLGlobalVariable *ref;

  LLGlobalVariable *name;
  LLGlobalVariable *table;
};

struct ObjcMethodInfo {
  FuncDeclaration *decl;
  
  LLGlobalVariable *name;
  LLGlobalVariable *type;
  LLGlobalVariable *selector;
  LLConstant *llfunction;
};

struct ObjcIvarInfo {
  VarDeclaration *decl;

  LLGlobalVariable *name;
  LLGlobalVariable *type;
  LLGlobalVariable *offset;
};

// Objective-C state tied to an LLVM module (object file).
class ObjCState {
public:

  ObjCState(llvm::Module &module) : module(module) { }

  ObjcClassInfo *getClass(ClassDeclaration *decl);
  ObjcProtocolInfo *getProtocol(InterfaceDeclaration *decl);
  ObjcMethodInfo *getMethod(FuncDeclaration *decl);
  ObjcIvarInfo *getIvar(VarDeclaration *decl);

  LLValue *deref(ClassDeclaration *decl, LLType *as);

  void finalize();

private:
  llvm::Module &module;

  // Creates an ivar_t struct which can be
  // used in ivar lists.
  ObjcMap<VarDeclaration *, ObjcIvarInfo> ivars;
  LLConstant *createIvarInfo(VarDeclaration *decl);
  LLConstant *createIvarList(ClassDeclaration *decl);

  // Creates an objc_method struct which can be
  // used in method lists.
  ObjcMap<FuncDeclaration *, ObjcMethodInfo> methods;
  LLConstant *createMethodInfo(FuncDeclaration *decl);
  LLConstant *createMethodList(ClassDeclaration *decl, bool optional = false);

  // class_t and class_ro_t generation.
  ObjcMap<ClassDeclaration *, ObjcClassInfo> classes;
  ObjcMap<ClassDeclaration *, LLGlobalVariable *> classTables;
  ObjcMap<ClassDeclaration *, LLGlobalVariable *> classRoTables;
  LLConstant *getClassRoTable(ClassDeclaration *decl);
  LLConstant *getClassTable(ClassDeclaration *decl);

  // Class names and refs need to be replicated
  // for RO structs, as such we store
  // then seperately.
  llvm::StringMap<LLGlobalVariable *> classNames;
  llvm::StringMap<LLGlobalVariable *> classRefs;
  LLConstant *getClassName(ClassDeclaration *decl);
  LLConstant *getClassRef(ClassDeclaration *decl);

  // protocol_t generation.
  ObjcMap<InterfaceDeclaration *, ObjcProtocolInfo> protocols;
  LLConstant *createProtocolTable(InterfaceDeclaration *decl);
  LLConstant *createProtocolList(ClassDeclaration *decl);

  // Private helpers
  ObjcList<FuncDeclaration *> getMethodsForType(ClassDeclaration *decl, bool optional = false);

  ObjcList<LLConstant *> retainedSymbols;
  void retain(LLConstant *symbol);
  void genImageInfo();
  void retainSymbols();
};
