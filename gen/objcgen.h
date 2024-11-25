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

// Gets the Objective-C type encoding for D type t 
std::string getTypeEncoding(Type *t);

// Gets whether Objective-C is supported.
bool objc_isSupported(const llvm::Triple &triple);

// Generate name strings
std::string getObjcClassRoSymbol(const char *name, bool meta);
std::string getObjcClassSymbol(const char *name, bool meta);
std::string getObjcClassMethodListSymbol(const char *className, bool meta);
std::string getObjcIvarListSymbol(const char *className);
std::string getObjcIvarSymbol(const char *className, const char *varName);
std::string getObjcProtoMethodListSymbol(const char *className, bool meta, bool optional);
std::string getObjcProtoSymbol(const char *name);
std::string getObjcProtoListSymbol(const char *name);
std::string getObjcSymbolName(const char *dsymPrefix, const char *dsymName);


// Base class for Objective-C definitions in a
// LLVM module.
class ObjcObject {
public:
  ObjcObject(llvm::Module &module, ObjCState &objc) : module(module), objc(objc) { }

  // Whether the object is used.
  bool isUsed;

  // Gets a reference to the object in the module.
  virtual LLConstant *get() { return nullptr; }

  // Gets the name of the object.
  virtual const char *getName() { return nullptr; }

  // Emits a new list for the specified objects as a constant.
  static LLConstant *emitList(llvm::Module &module, LLConstantList objects, bool isCountPtrSized = false);

protected:

  // Gets a global variable or creates it.
  LLGlobalVariable *getOrCreate(LLStringRef name, LLType* type, LLStringRef section, bool extInitializer=false) {
    auto global = module.getGlobalVariable(name, true);
    if (global)
      return global;

    return makeGlobal(name, type, section, true, extInitializer);
  }

  // Gets a global variable or creates it.
  LLGlobalVariable *getOrCreateWeak(LLStringRef name, LLType* type, LLStringRef section, bool extInitializer=false) {
    auto global = module.getGlobalVariable(name, true);
    if (global)
      return global;

    global = makeGlobal(name, type, section, false, extInitializer);
    global->setLinkage(llvm::GlobalValue::LinkageTypes::WeakAnyLinkage);
    global->setVisibility(llvm::GlobalValue::VisibilityTypes::HiddenVisibility);
    return global;
  }

  // The module the object resides in.
  llvm::Module &module;

  // The irstate.
  ObjCState &objc;

  // Called to emit the data for the type.
  virtual LLConstant *emit() { return nullptr; }

  // Retains a symbol.
  void retain(LLGlobalVariable *toRetain);
};

// objc_method
class ObjcMethod : public ObjcObject {
public:
  FuncDeclaration *decl;

  ObjcMethod(llvm::Module &module, ObjCState &objc, FuncDeclaration *decl) : 
    ObjcObject(module, objc), decl(decl) { }

  // Gets the main reference to the object.
  LLConstant *get() override;

  // Gets the type of an Objective-C objc_method struct
  static LLStructType *getObjcMethodType(const llvm::Module& module) {
    auto methType = LLStructType::getTypeByName(module.getContext(), OBJC_STRUCTNAME_METHOD);
    if (methType)
      return methType;

    return LLStructType::create(
      module.getContext(),
      {
        getOpaquePtrType(), // SEL name
        getOpaquePtrType(), // const char *types
        getOpaquePtrType(), // IMP imp
      },
      OBJC_STRUCTNAME_METHOD
    );
  }

  // Emits the constant struct containing the method
  // information.
  LLConstant *info();

  // Gets the selector for the function
  LLStringRef getSelector() {
    auto selector = decl->objc.selector;
    return LLStringRef(selector->stringvalue, selector->stringlen);
  }

  // Gets whether the function is optional.
  bool isOptional() {
    return decl->objc.isOptional;
  }

protected:

  // Called to emit the object.
  LLConstant *emit() override;

private:
  LLGlobalVariable *selref;
  LLGlobalVariable *name;
  LLGlobalVariable *type;
};

// ivar_t
class ObjcIvar : public ObjcObject {
public:
  VarDeclaration *decl;

  ObjcIvar(llvm::Module &module, ObjCState &objc, VarDeclaration *decl) : 
    ObjcObject(module, objc), decl(decl) { }

  // Gets the type for an Objective-C ivar_t struct. 
  static LLStructType *getObjcIvarType(const llvm::Module& module) {
    auto ivarType = LLStructType::getTypeByName(module.getContext(), OBJC_STRUCTNAME_IVAR);
    if (ivarType)
      return ivarType;

    ivarType = LLStructType::create(
      module.getContext(),
      {
        getOpaquePtrType(), // int32_t *offset
        getOpaquePtrType(), // const char *name
        getOpaquePtrType(), // const char *type
        getI32Type(),       // uint32_t alignment_raw
        getI32Type(),       // uint32_t size
      },
      OBJC_STRUCTNAME_IVAR
    );
    return ivarType;
  }

  LLConstant *getOffset() {
    isUsed = true;
    if (!name)
      emit();
    
    return offset;
  }

  // Gets the main reference to the object.
  LLConstant *get() override {
    isUsed = true;
    if (!name)
      emit();
    
    return name;
  }

  // Emits the constant struct containing the method
  // information.
  LLConstant *info();

protected:

  // Called to emit the object.
  LLConstant *emit() override;

private:
  LLGlobalVariable *offset;
  LLGlobalVariable *name;
  LLGlobalVariable *type;
};

// Base type for class-like objective-c types.
class ObjcClasslike : public ObjcObject {
public:
  ClassDeclaration *decl;

  ObjcClasslike(llvm::Module &module, ObjCState &objc, ClassDeclaration *decl) : 
    ObjcObject(module, objc), decl(decl) { }
  
  const char *getName() override;

  virtual ObjcMethod *getMethod(FuncDeclaration *fd) {
    for(auto it : instanceMethods) {
      if (it->decl == fd) {
        return it;
      }
    }

    for(auto it : classMethods) {
      if (it->decl == fd) {
        return it;
      }
    }

    return nullptr;
  }

  // Scans the class-like object and fills out internal information
  // about functions, ivars, etc.
  void scan() {
    if (!hasScanned) {
      onScan(true);
      onScan(false);
    }
    
    hasScanned = true;
  }

protected:
  LLGlobalVariable *emitName();
  
  // Implement this to modify scanning behaviour.
  virtual void onScan(bool meta);

  // Emits a method list as a constant.
  LLConstant *emitMethodList(std::vector<ObjcMethod *> &methods, bool optionalMethods=false);

  // Emits a method list as a constant.
  LLConstant *emitProtocolList();

  ObjcList<ObjcMethod *> instanceMethods;
  ObjcList<ObjcMethod *> classMethods;
private:

  LLGlobalVariable *className;
  bool hasScanned;
};

// objc_protocol_t
class ObjcProtocol : public ObjcClasslike {
public:
  ObjcProtocol(llvm::Module &module, ObjCState &objc, ClassDeclaration *decl) : 
    ObjcClasslike(module, objc, decl) { }

  // Gets the type of an Objective-C class_t struct
  static LLStructType *getObjcProtocolType(const llvm::Module& module) {
    auto protoType = LLStructType::getTypeByName(module.getContext(), OBJC_STRUCTNAME_PROTO);
    if (protoType)
      return protoType;

    protoType = LLStructType::create(
      module.getContext(),
      {
        getOpaquePtrType(), // objc_object* isa
        getOpaquePtrType(), // protocol_list_t* protocols
        getOpaquePtrType(), // const char *mangledName
        getOpaquePtrType(), // method_list_t* instanceMethods
        getOpaquePtrType(), // method_list_t* classMethods
        getOpaquePtrType(), // method_list_t* optionalInstanceMethods
        getOpaquePtrType(), // method_list_t* optionalClassMethods
        getOpaquePtrType(), // property_list_t* instanceProperties
        getI32Type(),       // uint32_t size
        getI32Type(),       // uint32_t flags

        // Further fields follow but are optional and are fixed up at
        // runtime.
      },
      OBJC_STRUCTNAME_PROTO
    );
    return protoType;
  }
  
  // Gets a protocol by its name
  static LLGlobalVariable *get(const llvm::Module& module, LLStringRef name, bool meta = false) {
    return module.getGlobalVariable(getObjcProtoSymbol(name.data()), true);
  }

  // Gets the main reference to the object.
  LLConstant *get() override {
    isUsed = true;
    if (!protocolTable)
      emit();
    
    return protocolTable;
  }

  // Gets the protocol ref.
  LLConstant *ref() { return protoref; }

protected:

  // Called to emit the object.
  LLConstant *emit() override;

  // Emits the protocol table.
  void emitTable(LLGlobalVariable *table);

private:
  LLGlobalVariable *protoref;
  LLGlobalVariable *protocolTable;
};

// class_t, class_ro_t, stub_class_t
class ObjcClass : public ObjcClasslike {
public:
  ObjcClass(llvm::Module &module, ObjCState &objc, ClassDeclaration *decl) : 
    ObjcClasslike(module, objc, decl) { }

  // Gets objective-c the flags for the class declaration
  static size_t getClassFlags(const ClassDeclaration& decl);

  // Gets the type of an Objective-C class_t struct
  static LLStructType *getObjcClassType(const llvm::Module& module) {
    auto classType = LLStructType::getTypeByName(module.getContext(), OBJC_STRUCTNAME_CLASS);
    if (classType)
      return classType;

    classType = LLStructType::create(
      module.getContext(),
      {
        getOpaquePtrType(), // objc_object* isa
        getOpaquePtrType(), // objc_object* superclass
        getOpaquePtrType(), // cache_t* cache
        getOpaquePtrType(), // void* vtbl; (unused, set to null)
        getOpaquePtrType(), // class_ro_t* ro
      },
      OBJC_STRUCTNAME_CLASS
    );
    return classType;
  }

  // Gets the type of an Objective-C class_ro_t struct.
  static LLStructType *getObjcClassRoType(const llvm::Module& module) {
    auto classRoType = LLStructType::getTypeByName(module.getContext(), OBJC_STRUCTNAME_CLASSRO);
    if (classRoType)
      return classRoType;

    classRoType = LLStructType::create(
      module.getContext(),
      {
          getI32Type(), // uint32_t flags
          getI32Type(), // uint32_t instanceStart
          getI32Type(), // uint32_t instanceSize
          getOpaquePtrType(), // void* layoutOrNonMetaClass
          getOpaquePtrType(), // const char* name
          getOpaquePtrType(), // method_list_t* baseMethods
          getOpaquePtrType(), // protocol_list_t* baseProtocols
          getOpaquePtrType(), // ivar_list_t* ivars
          getOpaquePtrType(), // const uint8_t* weakIvarLayout
          getOpaquePtrType(), // property_list_t* baseProperties
      },
      OBJC_STRUCTNAME_CLASSRO
    );
    return classRoType;
  }

  // Gets the type for an Swift stub class
  static LLStructType *getObjcStubClassType(const llvm::Module& module) {
    auto stubClassType = LLStructType::getTypeByName(module.getContext(), OBJC_STRUCTNAME_STUBCLASS);
    if (stubClassType)
      return stubClassType;

    stubClassType = LLStructType::create(
      module.getContext(),
      {
        getOpaquePtrType(), // objc_object* isa
        getOpaquePtrType(), // function pointer.
      },
      OBJC_STRUCTNAME_STUBCLASS
    );
    return stubClassType;
  }

  // Gets a class by its name
  static LLGlobalVariable *get(const llvm::Module& module, LLStringRef name, bool meta = false) {
    return module.getGlobalVariable(getObjcClassSymbol(name.data(), meta), true);
  }

  ObjcIvar *get(VarDeclaration *vd) {
    for(size_t i = 0; i < ivars.size(); i++) {
      if (ivars[i]->decl == vd) {
        return ivars[i];
      }
    }

    return nullptr;
  }

  LLGlobalVariable *getIVarOffset(VarDeclaration *vd);

  // Gets a reference to the class.
  LLValue *ref();

  // Gets the main reference to the object.
  LLConstant *get() override;

  // Gets the superclass of this class.
  LLConstant *getSuper(bool meta);

  // Gets the root metaclass of this class.
  LLConstant *getRootMetaClass();

protected:

  // Called to emit the object.
  LLConstant *emit() override;

  void onScan(bool meta) override;

private:
  ObjcList<ObjcIvar *> ivars;

  // Core data
  ptrdiff_t getInstanceStart(bool meta);
  size_t getInstanceSize(bool meta);

  void emitTable(LLGlobalVariable *table, LLConstant *super, LLConstant *meta, LLConstant *roTable);
  void emitRoTable(LLGlobalVariable *table, bool meta);

  // instance variables
  LLConstant *emitIvarList();

  LLValue *deref(LLValue *classptr);

  // Gets the empty cache variable, and creates a reference to it
  // if needed.
  LLGlobalVariable *getEmptyCache() {
    static LLGlobalVariable *objcCache;
    if(!objcCache)
      objcCache = makeGlobal("_objc_empty_cache", nullptr, "", true, false);
    return objcCache;
  }
  
  LLGlobalVariable *classTable;
  LLGlobalVariable *classRoTable;
  LLGlobalVariable *metaClassTable;
  LLGlobalVariable *metaClassRoTable;
};

// Objective-C state tied to an LLVM module (object file).
class ObjCState {
friend ObjcObject;
public:

  ObjCState(llvm::Module &module) : module(module) { }
  
  void               emit(ClassDeclaration *cd);
  ObjcClass         *getClassRef(ClassDeclaration *cd);
  ObjcProtocol      *getProtocolRef(InterfaceDeclaration *id);
  ObjcMethod        *getMethodRef(ClassDeclaration *cd, FuncDeclaration *fd);
  ObjcMethod        *getMethodRef(FuncDeclaration *fd);
  ObjcIvar          *getIVarRef(ClassDeclaration *cd, VarDeclaration *vd);
  LLGlobalVariable  *getIVarOffset(ClassDeclaration *cd, VarDeclaration *vd);

  void finalize();

private:
  llvm::Module &module;
  ObjcList<LLConstant *> retained;

  ObjcList<ObjcProtocol *> protocols;
  ObjcList<ObjcClass *> classes;

  void genImageInfo();
  void retainSymbols();
};
