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
    case llvm::Triple::arm:     // armv6 iOS
    case llvm::Triple::thumb:   // thumbv7 iOS, watchOS
    case llvm::Triple::x86_64:  // OSX, iOS, tvOS sim
      return true;
    case llvm::Triple::x86: // OSX, iOS, watchOS sim
      return false;
    default:
      break;
    }
  }
  return false;
}

// TYPE ENCODINGS
std::string getTypeEncoding(Type *t) {
  std::string tmp;
  switch (t->ty) {
    case TY::Tclass: {
      if (auto klass = t->isTypeClass()) {
        return klass->sym->classKind == ClassKind::objc ? "@" : "?";
      }
      return "?";
    }
    case TY::Tfunction: {
      tmp = getTypeEncoding(t->nextOf());
      tmp.append("@:");

      if (auto func = t->isTypeFunction()) {
        for (size_t i = 0; i < func->parameterList.length(); i++)
          tmp.append(getTypeEncoding(func->parameterList[i]->type));
      }
      return tmp;
    }
    case TY::Tpointer: {

      // C string (char*)
      if (t->nextOf()->ty == TY::Tchar)
        return "*";

      tmp.append("^");
      tmp.append(getTypeEncoding(t->nextOf()));
      return tmp;
    }
    case TY::Tsarray: {

      // Static arrays are encoded in the form of:
      // [<element count><element type>]
      auto typ = t->isTypeSArray();
      uinteger_t count = typ->dim->toUInteger();
      tmp.append("[");
      tmp.append(std::to_string(count));
      tmp.append(getTypeEncoding(typ->next));
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
        tmp.append(getTypeEncoding(sym->argType(i)));
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

std::string getObjcClassRoSymbol(const char *name, bool meta) {
  return getObjcSymbolName(meta ? "_OBJC_METACLASS_RO_$_" : "_OBJC_CLASS_RO_$_", name);
}

std::string getObjcClassSymbol(const char *name, bool meta) {
  return getObjcSymbolName(meta ? "OBJC_METACLASS_$_" : "OBJC_CLASS_$_", name);
}

std::string getObjcClassMethodListSymbol(const char *className, bool meta) {
  return getObjcSymbolName(meta ? "_OBJC_$_CLASS_METHODS_" : "_OBJC_$_INSTANCE_METHODS_", className);
}

std::string getObjcProtoMethodListSymbol(const char *className, bool meta) {
  return getObjcSymbolName(meta ? "_OBJC_$_PROTOCOL_CLASS_METHODS_" : "_OBJC_$_PROTOCOL_INSTANCE_METHODS_", className);
}

std::string getObjcIvarListSymbol(const char *className) {
  return getObjcSymbolName("_OBJC_$_INSTANCE_VARIABLES_", className);
}

std::string getObjcProtoSymbol(const char *name) {
  return getObjcSymbolName("_OBJC_PROTOCOL_$_", name);
}

std::string getObjcProtoListSymbol(const char *name, bool isProtocol) {
  return getObjcSymbolName(isProtocol ? "_OBJC_PROTOCOL_PROTOCOLS_$_" : "_OBJC_CLASS_PROTOCOLS_$_", name);
}

std::string getObjcIvarSymbol(const char *className, const char *varName) {
  return ("OBJC_IVAR_$_" + std::string(className) + "." + std::string(varName));
}

std::string getObjcSymbolName(const char *dsymPrefix, const char *dsymName) {
  return (std::string(dsymPrefix) + std::string(dsymName));
}

//
//      METHODS
//

LLConstant *ObjcMethod::emit() {

  // Extern declarations don't need to define
  // a var type.
  if (!decl->fbody) {
    name = makeGlobalStr(getSelector(), "OBJC_METH_VAR_NAME_", OBJC_SECNAME_METHNAME);
    selref = makeGlobalRef(name, "OBJC_SELECTOR_REFERENCES_", OBJC_SECNAME_SELREFS, true, true);
    return selref;
  }

  name = makeGlobalStr(getSelector(), "OBJC_METH_VAR_NAME_", OBJC_SECNAME_METHNAME);
  type = makeGlobalStr(getTypeEncoding(decl->type), "OBJC_METH_VAR_TYPE_", OBJC_SECNAME_METHTYPE);
  selref = makeGlobalRef(name, "OBJC_SELECTOR_REFERENCES_", OBJC_SECNAME_SELREFS, true, true);
  return selref;
}

// Implements the objc_method structure
LLConstant *ObjcMethod::info() {
  if (!name)
    emit();

  if (!decl->fbody)
    return nullptr;

  return LLConstantStruct::get(
    getObjcMethodType(module),
    { name, type, DtoCallee(decl) }
  );
}

LLConstant *ObjcMethod::get() {
  isUsed = true;
  if (!name)
    emit();
  
  return selref;
}

LLConstant *ObjcObject::emitList(llvm::Module &module, LLType *elemType, LLConstantList objects, bool isCountPtrSized) {
  LLConstantList members;
  
  // Size of stored struct.
  size_t allocSize = getTypeAllocSize(elemType);
  members.push_back(
    isCountPtrSized ?
    DtoConstSize_t(allocSize) :
    DtoConstUint(allocSize)
  );

  // Method count
  members.push_back(DtoConstUint(
    objects.size()
  ));

  // Insert all the objects in the constant list.
  members.insert(
    members.end(), 
    objects.begin(), 
    objects.end()
  );

  return LLConstantStruct::getAnon(
    members,
    true
  );
}

//
//      INSTANCE VARIABLES
//

LLConstant *ObjcIvar::emit() {
  name = makeGlobalStr(decl->ident->toChars(), OBJC_SECNAME_METHNAME, "OBJC_METH_VAR_NAME_");
  type = makeGlobalStr(getTypeEncoding(decl->type), OBJC_SECNAME_METHTYPE, "OBJC_METH_VAR_TYPE_");
  offset = makeGlobal(getObjcIvarSymbol(decl->parent->ident->toChars(), decl->ident->toChars()), nullptr, OBJC_SECNAME_IVAR);
  offset->setInitializer(DtoConstUint(decl->offset));
  return nullptr;
}

// Implements the objc_method structure
LLConstant *ObjcIvar::info() {
  LLConstantList members;

  members.push_back(offset);
  members.push_back(DtoConstCString(decl->ident->toChars()));
  members.push_back(DtoConstCString(getTypeEncoding(decl->type).c_str()));
  members.push_back(DtoConstUint(decl->alignment.value));
  members.push_back(DtoConstUint(decl->size(decl->loc)));

  return LLConstantStruct::get(
    ObjcIvar::getObjcIvarType(module),
    members
  );
}



//
//      CLASS-LIKES
//

void ObjcClasslike::onScan(bool meta) {
  auto methods =
    (meta && decl->objc.metaclass ) ? 
    decl->objc.metaclass->objc.methodList : 
    decl->objc.methodList;
  
  for(size_t i = 0; i < methods.length; i++) {
    auto method = methods.ptr[i];

    // Static functions are class methods.
    if (method->isStatic()) {
      auto mth = new ObjcMethod(module, method);
      mth->get();
      classMethods.push_back(mth);
      continue;
    }

    auto mth = new ObjcMethod(module, method);
    mth->get();
    instanceMethods.push_back(mth);
  }
}

bool ifaceListHas(ObjcList<InterfaceDeclaration *> &list, InterfaceDeclaration *curr) {
  for(auto it = list.begin(); it != list.end(); ++it) {
    if (*it == curr)
      return true;
  }
  return false;
}

LLConstant *ObjcClasslike::emitProtocolList() {
  LLConstantList list;

  auto ifaces = decl->interfaces;
  for(size_t i = 0; i < ifaces.length; i++) {
    if (auto iface = ifaces.ptr[i]) {

      // Only add interfaces which have objective-c linkage
      // TODO: throw an error if you try to include a non-objective-c interface?
      if (auto ifacesym = (InterfaceDeclaration *)iface->sym) {
        if (ifacesym->classKind == ClassKind::objc) {
          auto proto = getOrCreate(
            getObjcProtoSymbol(ifacesym->ident->toChars()), 
            ObjcProtocol::getObjcProtocolType(module), ""
          );

          list.push_back(proto);
        }
      }
    }
  }

  return ObjcObject::emitList(
    module, 
    getOpaquePtrType(), 
    list
  );
}

LLConstant *ObjcClasslike::emitMethodList(ObjcList<ObjcMethod *> &methods) {
  LLConstantList toAdd;
  
  // Find out how many functions have actual bodies.
  size_t count = 0;
  for(size_t i = 0; i < methods.size(); i++) {
    if (methods[i]->decl->fbody)
      toAdd.push_back(methods[i]->info());
  }

  return ObjcObject::emitList(
    module, 
    ObjcMethod::getObjcMethodType(module), 
    toAdd
  );
}

LLGlobalVariable *ObjcClasslike::emitName() {
  if (className)
    return className;
  
  className = makeGlobalStr(decl->objc.identifier->toChars(), "OBJC_CLASS_NAME_", OBJC_SECNAME_CLASSNAME);
  return className;
}

const char *ObjcClasslike::getName() {
  return decl->objc.identifier->toChars();
}

//
//      CLASSES
//

size_t ObjcClass::getClassFlags(const ClassDeclaration& decl) {
  size_t flags = 0;
  if (!decl.baseClass)
    flags |= RO_ROOT;
  
  if (decl.objc.isMeta)
    flags |= RO_META;

  return flags;
}

void ObjcClass::onScan(bool meta) {
  ObjcClasslike::onScan(meta);

  if (!meta) {
    
    // Push on all the ivars.
    for(size_t i = 0; i < decl->fields.size(); i++) {
      ivars.push_back(new ObjcIvar(module, decl->fields[i]));
    }
  }
}

LLConstant *ObjcClass::emitIvarList() {
  LLConstantList members;
  
  // Size of ivar_t
  members.push_back(DtoConstUint(
    getTypeAllocSize(ObjcIvar::getObjcIvarType(module))
  ));

  // Ivar count
  members.push_back(DtoConstUint(
    ivars.size()
  ));

  // Push on all the ivars.
  for(size_t i = 0; i < ivars.size(); i++) {
    members.push_back(ivars[i]->info());
  }

  return LLConstantStruct::getAnon(
    members,
    true
  );
}

ptrdiff_t ObjcClass::getInstanceStart(bool meta) {
  ptrdiff_t start = meta ? 
    getTypeAllocSize(ObjcClass::getObjcClassType(module)) :
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

size_t ObjcClass::getInstanceSize(bool meta) {
  size_t start = meta ? 
    getTypeAllocSize(ObjcClass::getObjcClassType(module)) :
    getPointerSize();

  if (meta)
    return start;
  
  return start+decl->size(decl->loc);
}

LLConstant *ObjcClass::getRootMetaClass() {
  auto curr = decl;
  while (curr->baseClass)
    curr = curr->baseClass;

  auto name = getObjcClassSymbol(curr->objc.identifier->toChars(), true);
  return getOrCreate(
    name, 
    ObjcClass::getObjcClassType(module), 
    curr->objc.isExtern ? "" : OBJC_SECNAME_DATA
  );
}

LLConstant *ObjcClass::getSuper(bool meta) {
  if (decl->objc.isRootClass() || !decl->baseClass) {
    return meta ? classTable : metaClassTable;
  }

  auto super = decl->baseClass;
  auto superName = getObjcClassSymbol(super->objc.identifier->toChars(), meta);
  return getOrCreate(
    superName, 
    ObjcClass::getObjcClassType(module), 
    super->objc.isExtern ? "" : OBJC_SECNAME_DATA
  );
}

void ObjcClass::emitTable(LLGlobalVariable *table, LLConstant *meta, LLConstant *super, LLConstant *roTable) {
  LLConstantList members;
  members.push_back(meta);
  members.push_back(super);
  members.push_back(getEmptyCache());
  members.push_back(getNullPtr());
  members.push_back(roTable);

  table->setInitializer(LLConstantStruct::get(
    getObjcClassType(module),
    members
  ));
}

void ObjcClass::emitRoTable(LLGlobalVariable *table, bool meta) {
  LLConstantList members;
  LLGlobalVariable *ivarList = nullptr;
  LLGlobalVariable *protocolList = nullptr;
  LLGlobalVariable *methodList = nullptr;

  // Base Methods
  auto baseMethods = meta ?
    this->emitMethodList(classMethods) : 
    this->emitMethodList(instanceMethods);

  methodList = getOrCreate(getObjcClassMethodListSymbol(getName(), meta), baseMethods->getType(), OBJC_SECNAME_CONST);
  methodList->setInitializer(baseMethods);

  if (!meta) {
    // Base Protocols
    auto baseProtocols = emitProtocolList();
    protocolList = getOrCreate(getObjcProtoListSymbol(getName(), true), baseProtocols->getType(), OBJC_SECNAME_CONST);
    protocolList->setInitializer(baseProtocols);

    // Instance variables
    auto baseIvars = emitIvarList();
    ivarList = getOrCreate(getObjcIvarListSymbol(getName()), baseIvars->getType(), OBJC_SECNAME_CONST);
    ivarList->setInitializer(baseIvars);
  }


  // Build struct.
  members.push_back(DtoConstUint(getClassFlags(meta ? *decl->objc.metaclass : *decl)));
  members.push_back(DtoConstUint(getInstanceStart(meta)));
  members.push_back(DtoConstUint(getInstanceSize(meta)));
  members.push_back(getNullPtr());
  members.push_back(emitName());
  members.push_back(wrapNull(methodList));
  members.push_back(wrapNull(protocolList));
  members.push_back(wrapNull(ivarList));
  members.push_back(getNullPtr());
  members.push_back(getNullPtr()); //TODO: Add properties?

  table->setInitializer(LLConstantStruct::get(
    getObjcClassRoType(module),
    members
  ));
}

LLConstant *ObjcClass::emit() {
  if (decl->objc.isSwiftStub && !decl->objc.isExtern) {
    error(decl->loc, "Cannot define non-extern swift stub classes!");
    fatal();
  }

  // Class already exists, just return that.
  if (classTable)
    return classTable;

  auto name = decl->objc.identifier->toChars();
  auto className = getObjcClassSymbol(name, false);
  auto metaName = getObjcClassSymbol(name, true);

  // Extern classes only need non-ro refs.
  if (decl->objc.isExtern) {
    this->scan();

    classTable = makeGlobal(className, ObjcClass::getObjcClassType(module), "", true, false);
    metaClassTable = makeGlobal(metaName, ObjcClass::getObjcClassType(module), "", true, false);
    return classTable;
  }

  auto classNameRo = getObjcClassRoSymbol(name, false);
  auto metaNameRo = getObjcClassRoSymbol(name, true);

  // If we were weakly declared before, go grab our declarations.
  // Otherwise, create all the base tables for the type.
  classTable = getOrCreate(className, ObjcClass::getObjcClassType(module), OBJC_SECNAME_DATA);
  metaClassTable = getOrCreate(metaName, ObjcClass::getObjcClassType(module), OBJC_SECNAME_DATA);
  classRoTable = getOrCreate(classNameRo, ObjcClass::getObjcClassRoType(module), OBJC_SECNAME_CONST);
  metaClassRoTable = getOrCreate(metaNameRo, ObjcClass::getObjcClassRoType(module), OBJC_SECNAME_CONST);

  this->scan();

  // Emit their structure.
  this->emitName();
  this->emitTable(classTable, metaClassTable, getSuper(classTable), classRoTable);
  this->emitTable(metaClassTable, getRootMetaClass(), getSuper(true), metaClassRoTable);
  this->emitRoTable(classRoTable, false);
  this->emitRoTable(metaClassRoTable, true);
  return classTable;
}

LLValue *ObjcClass::deref(LLValue *classptr) {
  if (decl->objc.isExtern && decl->objc.isSwiftStub) {
    auto loadClassFunc = getRuntimeFunction(decl->loc, module, "objc_loadClassRef");
    return gIR->CreateCallOrInvoke(loadClassFunc, classptr, "");
  }

  return classptr;
}

LLValue *ObjcClass::ref() {
  return deref(classTable);
}

LLValue *ObjcClass::getRefFor(LLValue *id) {
  if (decl->objc.isExtern) {

    // We can't be sure that the isa "pointer" is actually a pointer to a class
    // In extern scenarios, therefore we call object_getClass.
    auto getClassFunc = getRuntimeFunction(decl->loc, module, "object_getClass");
    auto classref = gIR->CreateCallOrInvoke(getClassFunc, id, "");
    return deref(classref);
  }

  // If we defined the type we can be 100% sure of the layout.
  // so this is a fast path.
  return deref(classTable);
}

LLConstant *ObjcClass::get() {
  isUsed = true;

  if (!classTable)
    return emit();

  return classTable;
}

//
//    PROTOCOLS
//

void ObjcProtocol::emitTable(LLGlobalVariable *table) {
  size_t allocSize = getTypeAllocSize(getObjcProtocolType(module));
  LLConstantList members;

  // Base Protocols
  auto baseProtocols = this->emitProtocolList();
  auto protocolList = getOrCreate(getObjcProtoListSymbol(getName(), false), baseProtocols->getType(), OBJC_SECNAME_CONST);
  protocolList->setInitializer(baseProtocols);

  // Class methods
  auto classMethodConsts = this->emitMethodList(classMethods);
  auto classMethodList = getOrCreate(getObjcProtoMethodListSymbol(getName(), true), classMethodConsts->getType(), OBJC_SECNAME_CONST);
  classMethodList->setInitializer(classMethodConsts);

  // Instance methods
  auto instanceMethodConsts = this->emitMethodList(instanceMethods);
  auto instanceMethodList = getOrCreate(getObjcProtoMethodListSymbol(getName(), false), instanceMethodConsts->getType(), OBJC_SECNAME_CONST);
  instanceMethodList->setInitializer(instanceMethodConsts);

  members.push_back(getNullPtr());              // isa
  members.push_back(emitName());                // mangledName
  members.push_back(protocolList);              // protocol_list
  members.push_back(instanceMethodList);        // instanceMethods
  members.push_back(classMethodList);           // classMethods
  members.push_back(getNullPtr());              // optionalInstanceMethods (TODO)
  members.push_back(getNullPtr());              // optionalClassMethods (TODO)
  members.push_back(getNullPtr());              // instanceProperties (TODO)
  members.push_back(DtoConstUint(allocSize));   // size
  members.push_back(DtoConstUint(0));           // flags

  table->setInitializer(LLConstantStruct::get(
    getObjcProtocolType(module),
    members
  ));
}

LLConstant *ObjcProtocol::emit() {

  // Protocol already exists, just return that.
  if (protoref)
    return protoref;

  auto name = decl->objc.identifier->toChars();
  auto protoName = getObjcProtoSymbol(name);

  // Extern classes only need non-ro refs.
  if (decl->objc.isExtern) {
    this->scan();
    protocolTable = makeGlobal(protoName, nullptr, "", true, false);
    protoref = makeGlobalRef(protocolTable, "objc_protoref", OBJC_SECNAME_PROTOREFS, true, false);
    return protoref;
  }


  // If we were weakly declared before, go grab our declarations.
  // Otherwise, create all the base tables for the type.
  protocolTable = getOrCreate(protoName, ObjcProtocol::getObjcProtocolType(module), "");
  protoref = makeGlobalRef(protocolTable, "objc_protoref", OBJC_SECNAME_PROTOREFS);

  // Emit their structure.
  this->scan();
  this->emitName();
  this->emitTable(protocolTable);
  return protoref;
}


//
//    STATE
//


ObjcClass *ObjCState::getClassRef(ClassDeclaration *cd) {
  for(auto it = classes.begin(); it != classes.end(); ++it) {
    if (auto klass = *it) {
      if (klass->decl == cd) {
        return klass;
      }
    }
  }

  auto klass = new ObjcClass(module, cd);
  klass->get();
  classes.push_back(klass);
  return klass;
}

ObjcProtocol *ObjCState::getProtocolRef(InterfaceDeclaration *id) {
  for(auto it = protocols.begin(); it != protocols.end(); ++it) {  
    if (auto proto = *it) {
      if (proto->decl == id) {
        return proto;
      }
    }
  }

  auto proto = new ObjcProtocol(module, id);
  proto->get();
  protocols.push_back(proto);
  return proto;
}

ObjcMethod *ObjCState::getMethodRef(ClassDeclaration *cd, FuncDeclaration *fd) {
  if (auto klass = getClassRef(cd)) {
    klass->scan();
    return klass->getMethod(fd);
  }

  return nullptr;
}

ObjcMethod *ObjCState::getMethodRef(InterfaceDeclaration *id, FuncDeclaration *fd) {
  if (auto proto = getProtocolRef(id)) {
    proto->scan();
    return proto->getMethod(fd);
  }

  return nullptr;
}

ObjcMethod *ObjCState::getMethodRef(FuncDeclaration *fd) {
  if (auto cd = fd->parent->isClassDeclaration()) {
    if (auto func = getMethodRef(cd, fd)) {
      return func;
    }
  }

  if (auto id = fd->parent->isInterfaceDeclaration()) {
    if (auto func = getMethodRef(id, fd)) {
      return func;
    }
  }

  return nullptr;
}

ObjcIvar *ObjCState::getIVarRef(ClassDeclaration *cd, VarDeclaration *vd) {
  if (auto klass = getClassRef(cd)) {
    return klass->get(vd);
  }

  return nullptr;
}

//
//    FINALIZATION
//

void ObjCState::finalize() {
  size_t totalObjects = classes.size()+protocols.size();
  if (totalObjects > 0) {
    genImageInfo();
  }
}

void ObjCState::genImageInfo() {
  module.addModuleFlag(llvm::Module::Error, "Objective-C Version", 2u); // Only support ABI 2. (Non-fragile)
  module.addModuleFlag(llvm::Module::Error, "Objective-C Image Info Version", 0u); // version
  module.addModuleFlag(llvm::Module::Error, "Objective-C Image Info Section", llvm::MDString::get(module.getContext(), OBJC_SECNAME_IMAGEINFO));
  module.addModuleFlag(llvm::Module::Override, "Objective-C Garbage Collection", 0u); // flags
}