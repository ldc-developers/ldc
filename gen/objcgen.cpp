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

std::string getObjcClassLabelSymbol(const char *name) {
  return getObjcSymbolName("OBJC_LABEL_CLASS_$_", name);
}

std::string getObjcClassMethodListSymbol(const char *className, bool meta) {
  return getObjcSymbolName(meta ? "_OBJC_$_CLASS_METHODS_" : "_OBJC_$_INSTANCE_METHODS_", className);
}

std::string getObjcProtoMethodListSymbol(const char *className, bool meta, bool optional) {  
  return optional ?
    getObjcSymbolName(meta ? "_OBJC_$_PROTOCOL_CLASS_METHODS_OPT_" : "_OBJC_$_PROTOCOL_INSTANCE_METHODS_OPT_", className) :
    getObjcSymbolName(meta ? "_OBJC_$_PROTOCOL_CLASS_METHODS_" : "_OBJC_$_PROTOCOL_INSTANCE_METHODS_", className);
}

std::string getObjcIvarListSymbol(const char *className) {
  return getObjcSymbolName("_OBJC_$_INSTANCE_VARIABLES_", className);
}

std::string getObjcProtoSymbol(const char *name) {
  return getObjcSymbolName("_OBJC_PROTOCOL_$_", name);
}

std::string getObjcProtoListSymbol(const char *name, bool isProtocol) {
  return getObjcSymbolName(isProtocol ? "_OBJC_$_PROTOCOL_REFS_" : "_OBJC_CLASS_PROTOCOLS_$_", name);
}

std::string getObjcProtoLabelSymbol(const char *name) {
  return getObjcSymbolName("_OBJC_LABEL_PROTOCOL_$_", name);
}

std::string getObjcIvarSymbol(const char *className, const char *varName) {
  return ("OBJC_IVAR_$_" + std::string(className) + "." + std::string(varName));
}

std::string getObjcSymbolName(const char *dsymPrefix, const char *dsymName) {
  return (std::string(dsymPrefix) + std::string(dsymName));
}

const char *getResolvedName(ClassDeclaration *decl) {
  if (auto mo = decl->pMangleOverride) {
    return mo->id->toChars();
  }

  return decl->objc.identifier->toChars();
}

//
//      ObjcObject
//
void ObjcObject::retain(LLGlobalVariable *toRetain) {
  for(auto elem : objc.retained) {
    if (elem == toRetain)
      return;
  }

  objc.retained.push_back(toRetain);
}

LLConstant *offsetIvar(size_t ivaroffset) {
  return DtoConstUint(getPointerSize()+ivaroffset);
}

LLConstant *ObjcObject::emitList(llvm::Module &module, LLConstantList objects, bool alignSizeT) {
  LLConstantList members;
  
  // Emit nullptr for empty lists.
  if (objects.empty())
    return nullptr;

  // Size of stored struct.
  size_t allocSize = getTypeAllocSize(objects.front()->getType());
  members.push_back(
    alignSizeT ?
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

LLConstant *ObjcObject::emitCountList(llvm::Module &module, LLConstantList objects, bool alignSizeT) {
  LLConstantList members;

  // Method count
  members.push_back(
    alignSizeT ?
    DtoConstSize_t(objects.size()) :
    DtoConstUint(objects.size())
  );

  // Insert all the objects in the constant list.
  if (!objects.empty()) {
    members.insert(
      members.end(), 
      objects.begin(), 
      objects.end()
    );
  }

  // These lists need to be null terminated.
  members.push_back(getNullPtr());
  return LLConstantStruct::getAnon(
    members,
    true
  );
}


//
//      METHODS
//

LLConstant *ObjcMethod::emit() {
  name = makeGlobalStr(getSelector(), "OBJC_METH_VAR_NAME_", OBJC_SECNAME_METHNAME);
  type = makeGlobalStr(getTypeEncoding(decl->type), "OBJC_METH_VAR_TYPE_", OBJC_SECNAME_METHTYPE);
  selref = makeGlobalRef(name, "OBJC_SELECTOR_REFERENCES_", OBJC_SECNAME_SELREFS, false, true);
  llfunc = decl->fbody ? 
    DtoBitCast(DtoCallee(decl), getOpaquePtrType()) :
    getNullPtr();

  this->retain(name);
  this->retain(type);
  this->retain(selref);
  return selref;
}

// Implements the objc_method structure
LLConstant *ObjcMethod::info(bool emitExtern) {
  if (!emitExtern && !decl->fbody)
    return nullptr;

  return LLConstantStruct::getAnon(
    { name, type, llfunc },
    true
  );
}

LLConstant *ObjcMethod::get() {
  return selref;
}

//
//      INSTANCE VARIABLES
//

LLConstant *ObjcIvar::emit() {
  auto ivarsym = getObjcIvarSymbol(decl->parent->ident->toChars(), decl->ident->toChars());

  // Extern, emit data.
  if (auto klass = decl->parent->isClassDeclaration()) {
    if (klass->objc.isExtern) {
      name = makeGlobal("OBJC_METH_VAR_NAME_", nullptr, OBJC_SECNAME_METHNAME, true, true);
      type = makeGlobal("OBJC_METH_VAR_TYPE_", nullptr, OBJC_SECNAME_METHTYPE, true, true);

      // It will be filled out by the runtime, but make sure it's there nontheless.
      offset = getOrCreate(ivarsym, getI32Type(), OBJC_SECNAME_IVAR);
      offset->setInitializer(offsetIvar(0));
      return nullptr;
    }

    // Non-extern, emit data.
    name = makeGlobalStr(decl->ident->toChars(), "OBJC_METH_VAR_NAME_", OBJC_SECNAME_METHNAME);
    type = makeGlobalStr(getTypeEncoding(decl->type), "OBJC_METH_VAR_TYPE_", OBJC_SECNAME_METHTYPE);

    offset = getOrCreate(ivarsym, getI32Type(), OBJC_SECNAME_IVAR);
    offset->setInitializer(offsetIvar(decl->offset));
  }
  return nullptr;
}

// Implements the objc_method structure
LLConstant *ObjcIvar::info() {
  LLConstantList members;

  members.push_back(offset);
  members.push_back(name);
  members.push_back(type);
  members.push_back(DtoConstUint(decl->alignment.isDefault() ? -1 : decl->alignment.get()));
  members.push_back(DtoConstUint(decl->size(decl->loc)));

  return LLConstantStruct::get(
    ObjcIvar::getObjcIvarType(module),
    members
  );
}

//
//      CLASS-LIKES
//

void ObjcClasslike::onScan() {

  // Class funcs
  if (auto metaclass = decl->objc.metaclass) {
    auto metamethods = metaclass->objc.methodList;

    for(size_t i = 0; i < metamethods.length; i++) {
      auto method = metamethods.ptr[i];

      // Static functions are class methods.
      if (method->isStatic()) {
        classMethods.push_back(
          new ObjcMethod(module, objc, method)
        );
      } else {
        instanceMethods.push_back(
          new ObjcMethod(module, objc, method)
        );
      }
    }
  }

  auto methods = decl->objc.methodList;
  for(size_t i = 0; i < methods.length; i++) {
    auto method = methods.ptr[i];

    // Static functions are class methods.
    if (method->isStatic()) {
      classMethods.push_back(
        new ObjcMethod(module, objc, method)
      );
    } else {
      instanceMethods.push_back(
        new ObjcMethod(module, objc, method)
      );
    }
  }
}

LLConstant *ObjcClasslike::emitProtocolList() {
  LLConstantList list;
  auto ifaces = decl->interfaces;

  // Protocols
  for(size_t i = 0; i < ifaces.length; i++) {
    if (auto iface = ifaces.ptr[i]) {

      // Only add interfaces which have objective-c linkage
      // TODO: throw an error if you try to include a non-objective-c interface?
      if (auto ifacesym = (InterfaceDeclaration *)iface->sym) {
        if (ifacesym->classKind == ClassKind::objc) {
          if (auto proto = this->objc.getProtocolRef(ifacesym)) {
            list.push_back(proto->get());
          }
        }
      }
    }
  }

  return ObjcObject::emitCountList(
    module,
    list,
    true
  );
}

LLConstant *ObjcClasslike::emitMethodList(ObjcList<ObjcMethod *> &methods, bool optionalMethods) {
  LLConstantList toAdd;

  // Emit nullptr for empty lists.
  if (methods.empty())
    return nullptr;

  bool isProtocol = decl->isInterfaceDeclaration();

  // Find out how many functions have actual bodies.
  for(size_t i = 0; i < methods.size(); i++) {
    auto method = methods[i];

    if (method->isOptional() == optionalMethods) {
      auto methodInfo = method->info(isProtocol);

      if (methodInfo)
        toAdd.push_back(methodInfo);
    }
  }

  // List is empty too, but due to optionals not matching.
  if (toAdd.empty())
    return nullptr;

  return ObjcObject::emitList(
    module, 
    toAdd
  );
}

LLGlobalVariable *ObjcClasslike::emitName() {
  if (className)
    return className;
  
  className = makeGlobalStr(getName(), "OBJC_CLASS_NAME_", OBJC_SECNAME_CLASSNAME);
  return className;
}

const char *ObjcClasslike::getName() {
  return getResolvedName(decl);
}

//
//      CLASSES
//

LLGlobalVariable *ObjcClass::getIVarOffset(VarDeclaration *vd) {
  auto ivarsym = getObjcIvarSymbol(decl->ident->toChars(), vd->ident->toChars());
  auto ivoffset = getOrCreate(ivarsym, getI32Type(), OBJC_SECNAME_IVAR);
  ivoffset->setInitializer(offsetIvar(vd->offset));
  this->retain(ivoffset);
  
  return ivoffset;
}

size_t ObjcClass::getClassFlags(const ClassDeclaration& decl) {
  size_t flags = 0;
  if (!decl.baseClass)
    flags |= RO_ROOT;
  
  if (decl.objc.isMeta)
    flags |= RO_META;

  return flags;
}

void ObjcClass::onScan() {
  ObjcClasslike::onScan();

  // Push on all the ivars.
  for(size_t i = 0; i < decl->fields.size(); i++) {
    auto ivar = new ObjcIvar(module, objc, decl->fields[i]);
    ivars.push_back(ivar);
    ivar->get();
  }
}

LLConstant *ObjcClass::emitIvarList() {
  LLConstantList ivarList;

  // Push on all the ivars.
  for(size_t i = 0; i < ivars.size(); i++) {
    auto ivarInfo = ivars[i]->info();

    if (ivarInfo)
      ivarList.push_back(ivarInfo);
  }

  return this->emitList(
    module,
    ivarList
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

  if (baseMethods) {
    methodList = getOrCreate(getObjcClassMethodListSymbol(getName(), meta), baseMethods->getType(), OBJC_SECNAME_CONST);
    methodList->setInitializer(baseMethods);
  }

  // Base Protocols
  if (auto baseProtocols = emitProtocolList()) {
    protocolList = getOrCreate(getObjcProtoListSymbol(getName(), false), baseProtocols->getType(), OBJC_SECNAME_CONST);
    protocolList->setInitializer(baseProtocols);
  }

  if (!meta) {

    // Instance variables
    if (auto baseIvars = emitIvarList()) {
      ivarList = getOrCreate(getObjcIvarListSymbol(getName()), baseIvars->getType(), OBJC_SECNAME_CONST);
      ivarList->setInitializer(baseIvars);
    }
  }


  // Build struct.
  members.push_back(DtoConstUint(getClassFlags(meta ? *decl->objc.metaclass : *decl)));
  members.push_back(DtoConstUint(getInstanceStart(meta)));
  members.push_back(DtoConstUint(getInstanceSize(meta)));
  members.push_back(getNullPtr());
  members.push_back(this->emitName());
  members.push_back(wrapNull(methodList));
  members.push_back(wrapNull(protocolList));
  members.push_back(wrapNull(ivarList));
  members.push_back(getNullPtr());
  members.push_back(getNullPtr());

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

  assert(decl && !decl->isInterfaceDeclaration());

  // Class already exists, just return that.
  if (classTable)
    return classTable;

  auto name = getName();
  auto className = getObjcClassSymbol(name, false);
  auto metaName = getObjcClassSymbol(name, true);

  // Extern classes only need non-ro refs.
  if (decl->objc.isExtern) {

    classTable = makeGlobal(className, ObjcClass::getObjcClassType(module), "", true, false);
    metaClassTable = makeGlobal(metaName, ObjcClass::getObjcClassType(module), "", true, false);
    
    // Still emit ivars.
    if (auto baseIvars = emitIvarList()) {
      auto ivarList = getOrCreate(getObjcIvarListSymbol(getName()), baseIvars->getType(), OBJC_SECNAME_CONST);
      ivarList->setInitializer(baseIvars);
      this->retain(ivarList);
    }

    // Still emit classref
    classref = makeGlobalRef(classTable, "OBJC_CLASSLIST_REFERENCES_$_", OBJC_SECNAME_CLASSREFS, false, true);
    this->retain(classref);
    this->retain(metaClassTable);
    this->retain(classTable);
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

  classref = makeGlobalRef(classTable, "OBJC_CLASSLIST_REFERENCES_$_", OBJC_SECNAME_CLASSREFS, false, true);
  this->scan();

  // Emit their structure.
  this->emitName();
  this->emitTable(classTable, metaClassTable, getSuper(false), classRoTable);
  this->emitTable(metaClassTable, getRootMetaClass(), getSuper(true), metaClassRoTable);
  this->emitRoTable(classRoTable, false);
  this->emitRoTable(metaClassRoTable, true);

  this->retain(classTable);
  this->retain(metaClassTable);
  this->retain(classRoTable);
  this->retain(metaClassRoTable);
  this->retain(classref);
  return classTable;
}

LLValue *ObjcClass::deref(LLType *as) {
  if (decl->objc.isExtern && decl->objc.isSwiftStub) {
    auto loadClassFunc = getRuntimeFunction(decl->loc, module, "objc_loadClassRef");
    return DtoBitCast(
      gIR->CreateCallOrInvoke(loadClassFunc, classref, ""),
      as
    );
  }

  return DtoLoad(as, classref);
}

LLConstant *ObjcClass::ref() { 
  return classref;
}

LLConstant *ObjcClass::get() {
  return classTable;
}

//
//    PROTOCOLS
//

LLConstant *ObjcProtocol::emitTable() {
  LLConstantList members;
  LLGlobalVariable *protocolList = nullptr;
  LLGlobalVariable *classMethodList = nullptr;
  LLGlobalVariable *instanceMethodList = nullptr;
  LLGlobalVariable *optClassMethodList = nullptr;
  LLGlobalVariable *optInstanceMethodList = nullptr;

  this->scan();

  // Base Protocols
  if (auto baseProtocols = this->emitProtocolList()) {
    auto sym = getObjcProtoListSymbol(getName(), true);

    protocolList = getOrCreateWeak(sym, baseProtocols->getType(), OBJC_SECNAME_CONST);
    protocolList->setInitializer(baseProtocols);
  }

  // Class methods
  if (auto classMethodConsts = this->emitMethodList(classMethods)) {
    auto sym = getObjcProtoMethodListSymbol(getName(), true, false);
    classMethodList = makeGlobal(sym, classMethodConsts->getType(), OBJC_SECNAME_CONST);
    classMethodList->setInitializer(classMethodConsts);

    classMethodList->setLinkage(llvm::GlobalValue::LinkageTypes::WeakAnyLinkage);
    classMethodList->setVisibility(llvm::GlobalValue::VisibilityTypes::HiddenVisibility);
  }

  // Instance methods
  if (auto instanceMethodConsts = this->emitMethodList(instanceMethods)) {
    auto sym = getObjcProtoMethodListSymbol(getName(), false, false);
    instanceMethodList = makeGlobal(sym, instanceMethodConsts->getType(), OBJC_SECNAME_CONST);
    instanceMethodList->setInitializer(instanceMethodConsts);

    instanceMethodList->setLinkage(llvm::GlobalValue::LinkageTypes::WeakAnyLinkage);
    instanceMethodList->setVisibility(llvm::GlobalValue::VisibilityTypes::HiddenVisibility);
  }

  // Optional class methods
  if (auto optClassMethodConsts = this->emitMethodList(classMethods, true)) {
    auto sym = getObjcProtoMethodListSymbol(getName(), true, true);
    optClassMethodList = makeGlobal(sym, optClassMethodConsts->getType(), OBJC_SECNAME_CONST);
    optClassMethodList->setInitializer(optClassMethodConsts);

    optClassMethodList->setLinkage(llvm::GlobalValue::LinkageTypes::WeakAnyLinkage);
    optClassMethodList->setVisibility(llvm::GlobalValue::VisibilityTypes::HiddenVisibility);
  }

  // Optional instance methods
  if (auto optInstanceMethodConsts = this->emitMethodList(instanceMethods, true)) {
    auto sym = getObjcProtoMethodListSymbol(getName(), false, true);
    optInstanceMethodList = makeGlobal(sym, optInstanceMethodConsts->getType(), OBJC_SECNAME_CONST);
    optInstanceMethodList->setInitializer(optInstanceMethodConsts);

    optInstanceMethodList->setLinkage(llvm::GlobalValue::LinkageTypes::WeakAnyLinkage);
    optInstanceMethodList->setVisibility(llvm::GlobalValue::VisibilityTypes::HiddenVisibility);
  }
  
  auto protoType = getObjcProtocolType(module);
  auto allocSize = getTypeAllocSize(protoType);

  members.push_back(getNullPtr());                    // isa
  members.push_back(this->emitName());                // mangledName
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

LLConstant *ObjcProtocol::emit() {
  if (protocolTable)
    return protocolTable;

  auto name = getName();
  auto protoName = getObjcProtoSymbol(name);
  auto protoLabel = getObjcProtoLabelSymbol(name);

  // Emit their structure.
  auto protoTableConst = this->emitTable();

  // We want it to be locally hidden and weak since the protocols
  // may be declared in multiple object files.
  protocolTable = getOrCreateWeak(protoName, protoTableConst->getType(), OBJC_SECNAME_DATA);
  protocolTable->setInitializer(protoTableConst);

  protoref = getOrCreateWeak(protoLabel, getOpaquePtrType(), OBJC_SECNAME_PROTOLIST);
  protoref->setInitializer(protocolTable);

  this->retain(protocolTable);
  this->retain(protoref);
  return protocolTable;
}

LLValue *ObjcProtocol::deref(LLType *as) { 
  return DtoLoad(as, protoref);
}


LLConstant *ObjcProtocol::ref() { 
  return protoref;
}

//
//    STATE
//
ObjcClass *ObjCState::getClassRef(ClassDeclaration *cd) {
  if (auto id = cd->isInterfaceDeclaration())
    return nullptr;

  auto classList = this->classes;
  if (!classList.empty()) {
    for(auto it : classList) {
      if (it->decl == cd) {
        return it;
      }
    }
  }

  auto klass = new ObjcClass(module, *this, cd);
  classes.push_back(klass);
  klass->get();
  return klass;
}

ObjcProtocol *ObjCState::getProtocolRef(InterfaceDeclaration *id) {
  
  auto protoList = this->protocols;
  if (!protoList.empty()) {
    for(auto it : protoList) {
      if (it->decl == id) {
        return it;
      }
    }
  }

  auto proto = new ObjcProtocol(module, *this, id);
  protocols.push_back(proto);
  proto->get();
  return proto;
}

ObjcMethod *ObjCState::getMethodRef(ClassDeclaration *cd, FuncDeclaration *fd) {
  if (auto id = cd->isInterfaceDeclaration()) {
    if (auto proto = getProtocolRef(id)) {

      // Attempt to get the method, if not found
      // try the parent.
      auto method = proto->getMethod(fd);
      if (!method) {
        for (auto baseclass : *id->baseclasses) {
          method = getMethodRef(baseclass->sym, fd);

          if (method)
            break;
        }
      }
      return method;
    }
  } else if (auto klass = getClassRef(cd)) {

    // Attempt to get the method, if not found
    // try the parent.
    auto method = klass->getMethod(fd);
    if (!method) {
      for (auto baseclass : *id->baseclasses) {
        method = getMethodRef(baseclass->sym, fd);

        if (method)
          break;
      }
    }

    return method;
  }

  return nullptr;
}

ObjcMethod *ObjCState::getMethodRef(FuncDeclaration *fd) {
  if (auto cd = fd->parent->isClassDeclaration()) {
    if (auto retval = getMethodRef(cd, fd))
      return retval;
  }

  return nullptr;
}

ObjcIvar *ObjCState::getIVarRef(ClassDeclaration *cd, VarDeclaration *vd) {
  if (auto klass = getClassRef(cd)) {
    return klass->get(vd);
  }

  return nullptr;
}

LLGlobalVariable *ObjCState::getIVarOffset(ClassDeclaration *cd, VarDeclaration *vd) {
  if (auto klass = getClassRef(cd)) {
    return klass->getIVarOffset(vd);
  }

  return nullptr;
}

void ObjCState::emit(ClassDeclaration *cd) {

  // Protocols should more or less always be emitted.
  if (auto id = cd->isInterfaceDeclaration()) {
    getProtocolRef(id);
    return;
  }
  
  // Meta-classes are emitted automatically with the class,
  // as such we only need to emit a classref for the base class.
  if (!cd || cd->objc.isMeta)
    return;

  getClassRef(cd);
}

//
//    FINALIZATION
//

void ObjCState::finalize() {
  size_t totalObjects = retained.size();
  if (totalObjects > 0) {

    genImageInfo();
    retainSymbols();
  }
}

void ObjCState::genImageInfo() {
  module.addModuleFlag(llvm::Module::Error, "Objective-C Version", 2u); // Only support ABI 2. (Non-fragile)
  module.addModuleFlag(llvm::Module::Error, "Objective-C Image Info Version", 0u); // version
  module.addModuleFlag(llvm::Module::Error, "Objective-C Image Info Section", llvm::MDString::get(module.getContext(), OBJC_SECNAME_IMAGEINFO));
  module.addModuleFlag(llvm::Module::Override, "Objective-C Garbage Collection", 0u); // flags
}

void ObjCState::retainSymbols() {
  auto retainedSymbols = this->retained;

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