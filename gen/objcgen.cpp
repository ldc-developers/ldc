//===-- objcgen.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
// Support limited to Objective-C on Darwin (OS X, iOS, tvOS, watchOS)
//
//===----------------------------------------------------------------------===//

#include "gen/objcgen.h"
#include "dmd/objc.h"
#include "dmd/expression.h"
#include "dmd/declaration.h"
#include "dmd/identifier.h"
#include "gen/irstate.h"
#include "gen/runtime.h"
#include "ir/irfunction.h"

bool objc_isSupported(const llvm::Triple &triple) {
  if (triple.isOSDarwin()) {

    // Objective-C only supported on Darwin at this time
    // Additionally only Objective-C 2 is supported.
    switch (triple.getArch()) {
    case llvm::Triple::aarch64: // arm64 iOS, tvOS, macOS, watchOS, visionOS
    case llvm::Triple::x86_64:  // OSX, iOS, tvOS sim
      return true;
    default:
      return false;
    }
  }
  return false;
}

// TYPE ENCODINGS
std::string objcGetTypeEncoding(Type *t) {
  std::string tmp;
  switch (t->ty) {
    case TY::Tclass: {
      if (auto klass = t->isTypeClass()) {
        return klass->sym->classKind == ClassKind::objc ? "@" : "?";
      }
      return "?";
    }
    case TY::Tfunction: {
      tmp = objcGetTypeEncoding(t->nextOf());
      tmp.append("@:");

      if (auto func = t->isTypeFunction()) {
        for (size_t i = 0; i < func->parameterList.length(); i++)
          tmp.append(objcGetTypeEncoding(func->parameterList[i]->type));
      }
      return tmp;
    }
    case TY::Tpointer: {

      // C string (char*)
      if (t->nextOf()->ty == TY::Tchar)
        return "*";

      tmp.append("^");
      tmp.append(objcGetTypeEncoding(t->nextOf()));
      return tmp;
    }
    case TY::Tsarray: {

      // Static arrays are encoded in the form of:
      // [<element count><element type>]
      auto typ = t->isTypeSArray();
      uinteger_t count = typ->dim->toUInteger();
      tmp.append("[");
      tmp.append(std::to_string(count));
      tmp.append(objcGetTypeEncoding(typ->next));
      tmp.append("]");
      return tmp;
    }
    case TY::Tstruct: {

      // Structs are encoded in the form of:
      //   {<name>=<element types>}
      // Unions are encoded as
      //   (<name>=<element types>)
      auto sym = t->isTypeStruct()->sym;
      bool isUnion = sym->isUnionDeclaration();

      tmp.append(isUnion ? "(" : "{");
      tmp.append(t->toChars());
      tmp.append("=");
      
      for(unsigned int i = 0; i < sym->numArgTypes(); i++) {
        tmp.append(objcGetTypeEncoding(sym->argType(i)));
      }

      tmp.append(isUnion ? ")" : "}");
      return tmp;
    }
    case TY::Tvoid: return "v";
    case TY::Tbool: return "B";
    case TY::Tint8: return "c";
    case TY::Tuns8: return "C";
    case TY::Tchar: return "C";
    case TY::Tint16: return "s";
    case TY::Tuns16: return "S";
    case TY::Twchar: return "S";
    case TY::Tint32: return "i";
    case TY::Tuns32: return "I";
    case TY::Tdchar: return "I";
    case TY::Tint64: return "q";
    case TY::Tuns64: return "Q";
    case TY::Tfloat32: return "f";
    case TY::Tcomplex32: return "jf";
    case TY::Tfloat64: return "d";
    case TY::Tcomplex64: return "jd";
    case TY::Tfloat80: return "D";
    case TY::Tcomplex80: return "jD";
    default: return "?"; // unknown
  }
}

//
//      STRING HELPERS
//

std::string objcGetClassRoSymbol(const char *name, bool meta) {
  return objcGetSymbolName(meta ? "_OBJC_METACLASS_RO_$_" : "_OBJC_CLASS_RO_$_", name);
}

std::string objcGetClassSymbol(const char *name, bool meta) {
  return objcGetSymbolName(meta ? "OBJC_METACLASS_$_" : "OBJC_CLASS_$_", name);
}

std::string objcGetClassLabelSymbol(const char *name) {
  return objcGetSymbolName("OBJC_LABEL_CLASS_$_", name);
}

std::string objcGetClassMethodListSymbol(const char *className, bool meta) {
  return objcGetSymbolName(meta ? "_OBJC_$_CLASS_METHODS_" : "_OBJC_$_INSTANCE_METHODS_", className);
}

std::string objcGetProtoMethodListSymbol(const char *className, bool meta, bool optional) {  
  return optional ?
    objcGetSymbolName(meta ? "_OBJC_$_PROTOCOL_CLASS_METHODS_OPT_" : "_OBJC_$_PROTOCOL_INSTANCE_METHODS_OPT_", className) :
    objcGetSymbolName(meta ? "_OBJC_$_PROTOCOL_CLASS_METHODS_" : "_OBJC_$_PROTOCOL_INSTANCE_METHODS_", className);
}

std::string objcGetIvarListSymbol(const char *className) {
  return objcGetSymbolName("_OBJC_$_INSTANCE_VARIABLES_", className);
}

std::string objcGetProtoSymbol(const char *name) {
  return objcGetSymbolName("_OBJC_PROTOCOL_$_", name);
}

std::string objcGetProtoListSymbol(const char *name, bool isProtocol) {
  return objcGetSymbolName(isProtocol ? "_OBJC_$_PROTOCOL_REFS_" : "_OBJC_CLASS_PROTOCOLS_$_", name);
}

std::string objcGetProtoLabelSymbol(const char *name) {
  return objcGetSymbolName("_OBJC_LABEL_PROTOCOL_$_", name);
}

std::string objcGetIvarSymbol(const char *className, const char *varName) {
  return ("OBJC_IVAR_$_" + std::string(className) + "." + std::string(varName));
}

std::string objcGetSymbolName(const char *dsymPrefix, const char *dsymName) {
  return (std::string(dsymPrefix) + std::string(dsymName));
}

const char *objcResolveName(Dsymbol *decl) {

  // Function names are based on selector.
  if (auto funcdecl = decl->isFuncDeclaration()) {
    return funcdecl->objc.selector->stringvalue;
  }

  // Class and interface names are determined by objc identifier.
  if (auto classdecl = decl->isClassDeclaration()) {
    return classdecl->objc.identifier->toChars();
  }

  return decl->ident->toChars();
}

//
//      TYPE HELPERS
//

LLStructType *objcGetStubClassType(const llvm::Module& module) {
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

LLStructType *objcGetClassRoType(const llvm::Module& module) {
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

LLStructType *objcGetClassType(const llvm::Module& module) {
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

LLStructType *objcGetMethodType(const llvm::Module& module) {
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

LLStructType *objcGetIvarType(const llvm::Module& module) {
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

LLStructType *objcGetProtocolType(const llvm::Module& module) {
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

//
//    *_list_t helpers.
//

LLConstant *objcEmitList(llvm::Module &module, LLConstantList objects, bool alignSizeT, bool countOnly) {
  LLConstantList members;
  
  // Emit nullptr for empty lists.
  if (objects.empty())
    return nullptr;

  if (!countOnly) {

    // Size of stored struct.
    size_t allocSize = getTypeAllocSize(objects.front()->getType());
    members.push_back(
      alignSizeT ?
      DtoConstSize_t(allocSize) :
      DtoConstUint(allocSize)
    );
  }

  // Object count
  size_t objCount = objects.size();
  members.push_back(
    alignSizeT ?
    DtoConstSize_t(objCount) :
    DtoConstUint(objCount)
  );

  // Insert all the objects in to a constant array.
  // This matches the codegen by the objective-c compiler.
  auto arrayType = LLArrayType::get(
    objects.front()->getType(),
    objects.size()
  );
  members.push_back(LLConstantArray::get(
    arrayType,
    objects
  ));

  return LLConstantStruct::getAnon(
    members,
    true
  );
}

//
//    Other helpers
//

LLConstant *objcOffsetIvar(size_t ivaroffset) {
  return DtoConstUint(getPointerSize()+ivaroffset);
}

size_t objcGetClassFlags(ClassDeclaration *decl) {
  size_t flags = 0;
  if (!decl->baseClass)
    flags |= RO_ROOT;
  
  if (decl->objc.isMeta)
    flags |= RO_META;

  return flags;
}

ClassDeclaration *objcGetMetaClass(ClassDeclaration *decl) {
  if (decl->objc.isMeta) {
    auto curr = decl;
    while (curr->baseClass)
      curr = curr->baseClass;

    return curr;
  }

  // Meta class for normal class.
  return decl->objc.metaclass;
}

ClassDeclaration *objcGetSuper(ClassDeclaration *decl) {
  return (decl->objc.isRootClass() || !decl->baseClass) ?
    decl : 
    decl->baseClass;
}

//
//    CLASSES
//

ptrdiff_t objcGetInstanceStart(llvm::Module &module, ClassDeclaration *decl, bool meta) {
  ptrdiff_t start = meta ? 
    getTypeAllocSize(objcGetClassType(module)) :
    getPointerSize();

  // Meta-classes have no body.
  if (meta || !decl->members || decl->members->length == 0)
    return start;

  for(d_size_t idx = 0; idx < decl->members->length; idx++)
  {
    auto var = (*decl->members)[idx]->isVarDeclaration();

    if (var && var->isField())
      return start+var->offset;
  }
  return start;
}

size_t objcGetInstanceSize(llvm::Module &module, ClassDeclaration *decl, bool meta) {
  size_t start = meta ? 
    getTypeAllocSize(objcGetClassType(module)) :
    getPointerSize();

  if (meta)
    return start;
  
  return start+decl->size(decl->loc);
}

// Gets the empty cache variable, and creates a reference to it
// if needed.
LLGlobalVariable *getEmptyCache() {
  static LLGlobalVariable *objcCache;
  if(!objcCache)
    objcCache = makeGlobal("_objc_empty_cache", nullptr, "", true, false);
  return objcCache;
}

LLConstant *ObjCState::getClassRoTable(ClassDeclaration *decl) {
  if (auto it = classRoTables.find(decl); it != classRoTables.end()) {
    return it->second;
  }
  
  // No need to generate RO tables for externs.
  // nor for null declarations.
  if (!decl || decl->objc.isExtern)
    return getNullPtr();
  
  // Base Methods
  auto meta = decl->objc.isMeta;
  auto name = objcResolveName(decl);
  auto sym = objcGetClassRoSymbol(name, meta);

  LLConstantList members;
  LLGlobalVariable *ivarList = nullptr;
  LLGlobalVariable *protocolList = nullptr;
  LLGlobalVariable *methodList = nullptr;

  if (auto baseMethods = createMethodList(decl)) {
    methodList = getOrCreate(objcGetClassMethodListSymbol(name, meta), baseMethods->getType(), OBJC_SECNAME_CONST);
    methodList->setInitializer(baseMethods);
  }

  // Base Protocols
  if (auto baseProtocols = createProtocolList(decl)) {
    protocolList = getOrCreate(objcGetProtoListSymbol(name, false), baseProtocols->getType(), OBJC_SECNAME_CONST);
    protocolList->setInitializer(baseProtocols);
  }

  if (!meta) {
    
    // Instance variables
    if (auto baseIvars = createIvarList(decl)) {
      ivarList = getOrCreate(objcGetIvarListSymbol(name), baseIvars->getType(), OBJC_SECNAME_CONST);
      ivarList->setInitializer(baseIvars);
    }
  }

  // Build struct.
  members.push_back(DtoConstUint(objcGetClassFlags(decl)));
  members.push_back(DtoConstUint(objcGetInstanceStart(module, decl, meta)));
  members.push_back(DtoConstUint(objcGetInstanceSize(module, decl, meta)));
  members.push_back(getNullPtr());
  members.push_back(getClassName(decl));
  members.push_back(wrapNull(methodList));
  members.push_back(wrapNull(protocolList));
  members.push_back(wrapNull(ivarList));
  members.push_back(getNullPtr());
  members.push_back(getNullPtr());

  auto table = makeGlobalWithBytes(sym, members, objcGetClassRoType(module));
  table->setSection(OBJC_SECNAME_DATA);

  classRoTables[decl] = table;
  this->retain(table);
  return table;
}

LLConstant *ObjCState::getClassTable(ClassDeclaration *decl) {
  if (auto it = classTables.find(decl); it != classTables.end()) {
    return it->second;
  }
  
  // If decl is null, just return a null pointer.
  if (!decl)
    return getNullPtr();

  auto name = objcResolveName(decl);
  auto sym = objcGetClassSymbol(name, decl->objc.isMeta);

  auto table = getOrCreate(sym, objcGetClassType(module), OBJC_SECNAME_DATA, decl->objc.isExtern);
  classTables[decl] = table;
  this->retain(table);
  
  // Extern tables don't need a body.
  if (decl->objc.isExtern)
    return table;

  LLConstantList members;
  members.push_back(getClassTable(objcGetMetaClass(decl)));   // isa
  members.push_back(getClassTable(objcGetSuper(decl)));       // super
  members.push_back(getEmptyCache());                         // cache
  members.push_back(getNullPtr());                            // vtbl
  members.push_back(getClassRoTable(decl));                   // ro
  table->setInitializer(LLConstantStruct::get(
    objcGetClassType(module),
    members
  ));
  return table;
}

ObjcClassInfo *ObjCState::getClass(ClassDeclaration *decl) {
  assert(!decl->isInterfaceDeclaration() && "Attempted to pass protocol into getClass!");
  if (auto it = classes.find(decl); it != classes.end()) {
    return &classes[decl];
  }

  // Since we may end up referring to this very quickly
  // the name should be assigned ASAP.
  classes[decl] = { /*.decl =*/ decl };
  auto classInfo = &classes[decl];

  classInfo->table = (LLGlobalVariable *)getClassTable(decl);
  classInfo->name = (LLGlobalVariable *)getClassName(decl);
  classInfo->ref = (LLGlobalVariable *)getClassRef(decl);

  if (!decl->objc.isMeta)
    classInfo->ref->setInitializer(classInfo->table);
  
  return classInfo;
}

LLConstant *ObjCState::getClassName(ClassDeclaration *decl) {
  LLStringRef className(objcResolveName(decl));
  if (auto it = classNames.find(className); it != classNames.end()) {
    return it->second;
  }

  auto retval = makeGlobalStr(className, "OBJC_CLASS_NAME_", OBJC_SECNAME_CLASSNAME);
  classNames[className] = retval;
  this->retain(retval);
  return retval;
}

LLConstant *ObjCState::getClassRef(ClassDeclaration *decl) {
  LLStringRef className(objcResolveName(decl));
  if (auto it = classRefs.find(className); it != classRefs.end()) {
    return it->second;
  }

  auto retval = makeGlobal("OBJC_CLASSLIST_REFERENCES_$_", getOpaquePtrType(), OBJC_SECNAME_CLASSREFS);
  classRefs[className] = retval;
  this->retain(retval);
  return retval;
}


//
//    PROTOCOLS
//

LLConstant *ObjCState::createProtocolTable(InterfaceDeclaration *decl) {
  LLConstantList members;
  LLGlobalVariable *protocolList = nullptr;
  LLGlobalVariable *classMethodList = nullptr;
  LLGlobalVariable *instanceMethodList = nullptr;
  LLGlobalVariable *optClassMethodList = nullptr;
  LLGlobalVariable *optInstanceMethodList = nullptr;

  auto protoInfo = &protocols[decl];
  auto name = objcResolveName(decl);

  // Base Protocols
  if (auto baseProtocols = createProtocolList(decl)) {
    auto sym = objcGetProtoListSymbol(name, true);

    protocolList = getOrCreateWeak(sym, baseProtocols->getType(), OBJC_SECNAME_CONST);
    protocolList->setInitializer(baseProtocols);
  }

  // Instance methods
  if (auto instanceMethodConsts = createMethodList(decl, false)) {
    auto sym = objcGetProtoMethodListSymbol(name, false, false);
    instanceMethodList = makeGlobal(sym, instanceMethodConsts->getType(), OBJC_SECNAME_CONST);
    instanceMethodList->setInitializer(instanceMethodConsts);

    instanceMethodList->setLinkage(llvm::GlobalValue::LinkageTypes::WeakAnyLinkage);
    instanceMethodList->setVisibility(llvm::GlobalValue::VisibilityTypes::HiddenVisibility);
  }

  // Optional instance methods
  if (auto optInstanceMethodConsts = createMethodList(decl, true)) {
    auto sym = objcGetProtoMethodListSymbol(name, false, true);
    optInstanceMethodList = makeGlobal(sym, optInstanceMethodConsts->getType(), OBJC_SECNAME_CONST);
    optInstanceMethodList->setInitializer(optInstanceMethodConsts);

    optInstanceMethodList->setLinkage(llvm::GlobalValue::LinkageTypes::WeakAnyLinkage);
    optInstanceMethodList->setVisibility(llvm::GlobalValue::VisibilityTypes::HiddenVisibility);
  }

  // Class methods
  if (auto classMethodConsts = createMethodList(decl->objc.metaclass, false)) {
    auto sym = objcGetProtoMethodListSymbol(name, true, false);
    classMethodList = makeGlobal(sym, classMethodConsts->getType(), OBJC_SECNAME_CONST);
    classMethodList->setInitializer(classMethodConsts);

    classMethodList->setLinkage(llvm::GlobalValue::LinkageTypes::WeakAnyLinkage);
    classMethodList->setVisibility(llvm::GlobalValue::VisibilityTypes::HiddenVisibility);
  }

  // Optional class methods
  if (auto optClassMethodConsts = createMethodList(decl->objc.metaclass, true)) {
    auto sym = objcGetProtoMethodListSymbol(name, true, true);
    optClassMethodList = makeGlobal(sym, optClassMethodConsts->getType(), OBJC_SECNAME_CONST);
    optClassMethodList->setInitializer(optClassMethodConsts);

    optClassMethodList->setLinkage(llvm::GlobalValue::LinkageTypes::WeakAnyLinkage);
    optClassMethodList->setVisibility(llvm::GlobalValue::VisibilityTypes::HiddenVisibility);
  }
  
  auto protoType = objcGetProtocolType(module);
  auto allocSize = getTypeAllocSize(protoType);

  members.push_back(getNullPtr());                    // isa
  members.push_back(protoInfo->name);                 // mangledName
  members.push_back(wrapNull(protocolList));          // protocols
  members.push_back(wrapNull(instanceMethodList));    // instanceMethods
  members.push_back(wrapNull(classMethodList));       // classMethods
  members.push_back(wrapNull(optInstanceMethodList)); // optionalInstanceMethods
  members.push_back(wrapNull(optClassMethodList));    // optionalClassMethods
  members.push_back(getNullPtr());                    // instanceProperties (TODO)
  members.push_back(DtoConstUint(allocSize));         // size
  members.push_back(DtoConstUint(0));                 // flags
  
  return LLConstantStruct::getAnon(
    members,
    true
  );
}

ObjcProtocolInfo *ObjCState::getProtocol(InterfaceDeclaration *decl) {
  assert(decl->isInterfaceDeclaration() && "Attempted to pass class into getProtocol!");
  if (auto it = protocols.find(decl); it != protocols.end()) {
    return &protocols[decl];
  }

  protocols[decl] = { /*.decl =*/ decl };
  auto protoInfo = &protocols[decl];

  auto name = objcResolveName(decl);
  auto protoName = objcGetProtoSymbol(name);
  auto protoLabel = objcGetProtoLabelSymbol(name);
  protoInfo->name = makeGlobalStr(name, "OBJC_CLASS_NAME_", OBJC_SECNAME_CLASSNAME);

  // We want it to be locally hidden and weak since the protocols
  // may be declared in multiple object files.
  auto protoTableConst = createProtocolTable(decl);
  protoInfo->table = getOrCreateWeak(protoName, protoTableConst->getType(), OBJC_SECNAME_DATA);
  protoInfo->table->setInitializer(protoTableConst);

  protoInfo->ref = getOrCreateWeak(protoLabel, getOpaquePtrType(), OBJC_SECNAME_PROTOLIST);
  protoInfo->ref->setInitializer(protoInfo->table);

  this->retain(protoInfo->table);
  this->retain(protoInfo->ref);
  return protoInfo;
}

LLConstant *ObjCState::createProtocolList(ClassDeclaration *decl) {
  LLConstantList protoList;
  auto ifaces = decl->interfaces;

  // Protocols
  for(size_t i = 0; i < ifaces.length; i++) {
    if (auto iface = ifaces.ptr[i]) {
      if (auto ifacesym = (InterfaceDeclaration *)iface->sym) {

        // Only add interfaces which have objective-c linkage
        // TODO: throw an error if you try to include a non-objective-c interface?
        if (ifacesym->classKind == ClassKind::objc) {
          if (auto proto = getProtocol(ifacesym)) {
            protoList.push_back(proto->table);
          }
        }
      }
    }
  }

  return objcEmitList(module, protoList, true, true);
}

//
//    METHODS
//

ObjcMethodInfo *ObjCState::getMethod(FuncDeclaration *decl) {
  if (auto it = methods.find(decl); it != methods.end()) {
    return &methods[decl];
  }

  // Skip functions not marked as extern(Objective-C).
  if (decl->_linkage != LINK::objc)
    return nullptr;

  methods[decl] = { /*.decl =*/ decl };
  auto methodInfo = &methods[decl];

  auto name = objcResolveName(decl);
  auto type = objcGetTypeEncoding(decl->type);
  methodInfo->name = makeGlobalStr(name, "OBJC_METH_VAR_NAME_", OBJC_SECNAME_METHNAME);
  methodInfo->type = makeGlobalStr(type, "OBJC_METH_VAR_TYPE_", OBJC_SECNAME_METHTYPE);
  methodInfo->selector = makeGlobalRef(methodInfo->name, "OBJC_SELECTOR_REFERENCES_", OBJC_SECNAME_SELREFS, false, true);
  methodInfo->llfunction = decl->fbody ?
    DtoBitCast(DtoCallee(decl), getOpaquePtrType()) :
    getNullPtr();
  
  this->retain(methodInfo->name);
  this->retain(methodInfo->type);
  this->retain(methodInfo->selector);

  return &methods[decl];
}

LLConstant *ObjCState::createMethodInfo(FuncDeclaration *decl) {
  auto method = getMethod(decl);
  return LLConstantStruct::get(
    objcGetMethodType(module),
    { method->name, method->type, method->llfunction }
  );
}

LLConstant *ObjCState::createMethodList(ClassDeclaration *decl, bool optional) {
  LLConstantList methodList;

  if (decl) {
    
    auto methodDeclList = getMethodsForType(decl, optional);
    for(auto func : methodDeclList) {
      methodList.push_back(createMethodInfo(func));
    }
  }
  return objcEmitList(module, methodList, false);
}


//
//    INSTANCE VARIABLES
//

ObjcIvarInfo* ObjCState::getIvar(VarDeclaration *decl) {
  if (auto it = ivars.find(decl); it != ivars.end()) {
    return &ivars[decl];
  }

  if (auto klass = decl->parent->isClassDeclaration()) {
    auto ivarsym = objcGetIvarSymbol(objcResolveName(decl->parent), objcResolveName(decl));
    ivars[decl] = { /*.decl =*/ decl };
    auto ivarInfo = &ivars[decl];

    // Extern classes should generate globals
    // which can be filled out by the Objective-C runtime.
    if (klass->objc.isExtern) {
      ivarInfo->name = makeGlobal("OBJC_METH_VAR_NAME_", nullptr, OBJC_SECNAME_METHNAME, true, true);
      ivarInfo->type = makeGlobal("OBJC_METH_VAR_TYPE_", nullptr, OBJC_SECNAME_METHTYPE, true, true);

      // It will be filled out by the runtime, but make sure it's there nontheless.
      ivarInfo->offset = getOrCreate(ivarsym, getI32Type(), OBJC_SECNAME_IVAR);
      ivarInfo->offset->setInitializer(objcOffsetIvar(0));

      this->retain(ivarInfo->name);
      this->retain(ivarInfo->type);
      this->retain(ivarInfo->offset);
      return &ivars[decl];
    }

    // Non-extern ivars should emit all the data so that the
    // objective-c runtime has a starting point.
    // the offset *WILL* change during runtime!
    ivarInfo->name = makeGlobalStr(decl->ident->toChars(), "OBJC_METH_VAR_NAME_", OBJC_SECNAME_METHNAME);
    ivarInfo->type = makeGlobalStr(objcGetTypeEncoding(decl->type), "OBJC_METH_VAR_TYPE_", OBJC_SECNAME_METHTYPE);
    ivarInfo->offset = getOrCreate(ivarsym, getI32Type(), OBJC_SECNAME_IVAR);
    ivarInfo->offset->setInitializer(objcOffsetIvar(decl->offset));

    this->retain(ivarInfo->name);
    this->retain(ivarInfo->type);
    this->retain(ivarInfo->offset);
    return &ivars[decl];
  }

  return nullptr;
}

LLConstant *ObjCState::createIvarInfo(VarDeclaration *decl) {
  auto ivar = getIvar(decl);
  LLConstantList members;

  members.push_back(ivar->offset);
  members.push_back(ivar->name);
  members.push_back(ivar->type);
  members.push_back(DtoConstUint(decl->alignment.isDefault() ? -1 : decl->alignment.get()));
  members.push_back(DtoConstUint(decl->size(decl->loc)));

  return LLConstantStruct::get(
    objcGetIvarType(module),
    members
  );
}

LLConstant *ObjCState::createIvarList(ClassDeclaration *decl) {
  LLConstantList ivarList;
  
  for(auto field : decl->fields) {
    ivarList.push_back(createIvarInfo(field));
  }
  return objcEmitList(module, ivarList, false);
}

//
//    HELPERS
//
LLValue *ObjCState::deref(ClassDeclaration *decl, LLType *as) {

  // Protocols can also have static functions
  // as such we need to also be able to dereference them.
  if (auto proto = decl->isInterfaceDeclaration()) {
    return DtoLoad(as, getProtocol(proto)->ref);
  }
  
  // Classes may be class stubs.
  // in that case, we need to call objc_loadClassRef instead of just
  // loading from the classref.
  auto classref = getClass(decl)->ref;
  if (decl->objc.isExtern && decl->objc.isSwiftStub) {
    auto loadClassFunc = getRuntimeFunction(decl->loc, module, "objc_loadClassRef");
    return DtoBitCast(
      gIR->CreateCallOrInvoke(loadClassFunc, classref, ""),
      as
    );
  }

  return DtoLoad(as, classref);
}

ObjcList<FuncDeclaration *> ObjCState::getMethodsForType(ClassDeclaration *decl, bool optional) {
  ObjcList<FuncDeclaration *> funcs;
  bool isProtocol = decl->isInterfaceDeclaration();

  if (decl) {
    for(size_t i = 0; i < decl->objc.methodList.length; i++) {
      auto method = decl->objc.methodList.ptr[i];
      
      if (isProtocol) {
        if (method->objc.isOptional == optional)
          funcs.push_back(method);
        continue;
      }

      if (method->fbody)
        funcs.push_back(method);
    }
  }
  return funcs;
}

//
//    FINALIZATION
//

void ObjCState::finalize() {
  if (retainedSymbols.size() > 0) {
    retainSymbols();
    genImageInfo();
  }
}

void ObjCState::retain(LLConstant *symbol) {
  retainedSymbols.push_back(symbol);
}

void ObjCState::genImageInfo() {
  module.addModuleFlag(llvm::Module::Error, "Objective-C Version", 2u); // Only support ABI 2. (Non-fragile)
  module.addModuleFlag(llvm::Module::Error, "Objective-C Image Info Version", 0u); // version
  module.addModuleFlag(llvm::Module::Error, "Objective-C Image Info Section", llvm::MDString::get(module.getContext(), OBJC_SECNAME_IMAGEINFO));
  module.addModuleFlag(llvm::Module::Override, "Objective-C Garbage Collection", 0u); // flags
}

void ObjCState::retainSymbols() {
  if (!retainedSymbols.empty()) {
    auto arrayType = LLArrayType::get(retainedSymbols.front()->getType(),
                                      retainedSymbols.size());
    auto usedArray = LLConstantArray::get(arrayType, retainedSymbols);
    auto var = new LLGlobalVariable(module, arrayType, false,
                                    LLGlobalValue::AppendingLinkage, usedArray,
                                    "llvm.compiler.used");
    var->setSection("llvm.metadata");
  }
}