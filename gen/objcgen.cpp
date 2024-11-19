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

#include "dmd/mtype.h"
#include "dmd/objc.h"
#include "dmd/expression.h"
#include "dmd/declaration.h"
#include "dmd/identifier.h"
#include "gen/irstate.h"
#include "gen/tollvm.h"
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

llvm::GlobalVariable *ObjCState::getGlobal(llvm::Module& module, llvm::StringRef name, llvm::Type* type = nullptr) {
  if (type == nullptr)
    type = getOpaquePtrType();

  auto var = new LLGlobalVariable(
    module,
    type,
    false,
    LLGlobalValue::ExternalLinkage,
    nullptr,
    name,
    nullptr,
    LLGlobalVariable::NotThreadLocal,
    0,
    true
  );
  return var;
}

llvm::GlobalVariable *ObjCState::getGlobalWithBytes(llvm::Module& module, llvm::StringRef name, ConstantList packedContents) {
  auto init = llvm::ConstantStruct::getAnon(
    packedContents,
    true
  );

  auto var = new LLGlobalVariable(
    module,
    init->getType(),
    false,
    LLGlobalValue::ExternalLinkage,
    init,
    name,
    nullptr,
    LLGlobalVariable::NotThreadLocal,
    0,
    false
  );
  
  var->setSection(dataSection);
  return var;
}

LLGlobalVariable *ObjCState::getCStringVar(const char *symbol,
                                           const llvm::StringRef &str,
                                           const char *section) {
  auto init = llvm::ConstantDataArray::getString(module.getContext(), str);
  auto var = new LLGlobalVariable(module, init->getType(), false,
                                  LLGlobalValue::PrivateLinkage, init, symbol);
  var->setSection(section);
  return var;
}

std::string ObjCState::getObjcTypeEncoding(Type *t) {
  std::string tmp;

  switch (t->ty) {
    case TY::Tpointer:

      // C string (char*)
      if (t->nextOf()->ty == TY::Tchar)
        return "*";

      tmp.append("^");
      tmp.append(getObjcTypeEncoding(t->nextOf()));
      return tmp;
    case TY::Tsarray:

      // Static arrays are encoded in the form of:
      // [<element count><element type>]
      auto typ = t->isTypeSArray();
      uinteger_t count = typ->dim->toUInteger();
      tmp.append("[");
      tmp.append(std::to_string(count));
      tmp.append(getObjcTypeEncoding(typ->next));
      tmp.append("]");
      return tmp;
    case TY::Tstruct:

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
        tmp.append(getObjcTypeEncoding(sym->argType(i)));
      }

      tmp.append(isUnion ? ")" : "}");
      return tmp;
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
    case TY::Tclass:
      if (auto klass = t->isTypeClass()) {
        return klass->sym->classKind == ClassKind::objc ? "@" : "?";
      }
      return "?";
    default: return "?"; // unknown
  }
}

// Helper functions to generate name symbols

//
//      STRING HELPERS
//

std::string ObjCState::getObjcClassRoSymbol(const ClassDeclaration& cd, bool meta) {
  return getObjcSymbolName(meta ? "OBJC_METACLASS_RO_$_" : "OBJC_CLASS_RO_$_", cd.toChars());
}

std::string ObjCState::getObjcClassSymbol(const ClassDeclaration& cd, bool meta) {
  return getObjcSymbolName(meta ? "OBJC_METACLASS_$_" : "OBJC_CLASS_$_", cd.toChars());
}

std::string ObjCState::getObjcMethodListSymbol(const ClassDeclaration& cd, bool meta) {
  return getObjcSymbolName(meta ? "OBJC_$_CLASS_METHODS_" : "OBJC_$_INSTANCE_METHODS_", cd.objc.identifier->toChars());
}

std::string ObjCState::getObjcProtoSymbol(const InterfaceDeclaration& iface) {
  return getObjcSymbolName("OBJC_PROTOCOL_$_", iface.toChars());
}

std::string ObjCState::getObjcIvarSymbol(const ClassDeclaration& cd, const VarDeclaration& var) {
  return std::string("OBJC_IVAR_$_", cd.objc.identifier->toChars(), ".", var.iden->toChars());
}

std::string ObjCState::getObjcSymbolName(const char *dsymPrefix, const char *dsymName) {
  return std::string(dsymPrefix, dsymName);
}

//
//      UTILITIES
//

LLGlobalVariable *ObjCState::getTypeEncoding(const Declaration& decl) {
  std::string out_ = getObjcTypeEncoding(decl.type->nextOf());

  if (decl.parameters) {
    for (size_t i = 0; i < decl.parameters->length; i++)
      out_.append(getObjcTypeEncoding(decl.parameters[i]->type));
  }

  return getMethodVarType(out_);
}

llvm::GlobalVariable* ObjCState::getEmptyCache() {
	static llvm::GlobalVariable* g;
	if(g == nullptr)
		g = getGlobal(module, "_objc_empty_cache");
	return g;
}

llvm::Value *ObjCState::unmaskPointer(llvm::Value *value) {
  return gIR->ir->CreateAnd(value, OBJC_PTRMASK);
}

//
//      CLASSES
//

unsigned int getClassFlags(const ClassDeclaration& cd) {
  unsigned int flags = 0;
  if (cd.objc.isRootClass())
    flags |= RO_ROOT;
  
  if (cd.objc.isMeta)
    flags |= RO_META;

  return flags;
}

LLGlobalVariable *ObjCState::getClassSymbol(const ClassDeclaration& cd, bool meta) {
  llvm::StringRef name(cd.toChars());
  auto it = classSymbolTable.find(name);
  if (it != classSymbolTable.end()) {
    return it->second;
  }

  // Extern objects 
  if (cd.objc.isExtern) {
    auto var = getGlobal(module, name);

    classSymbolTable[name] = var;
    retain(var);
    return var;
  }

  // Classes and Metaclasses are the same thing in Objective-C
  // 
  // Class symbol layout is as follows:
  // struct objc_class {
  //   Class isa;         // inherited from objc_object
  //   Class superclass;
  //   cache_t cache;     // Formerly vtable and cache pointer
  //   class_data_bits_t bits;
  // }
  // 
  // the isa pointer points to the metaclass of the class.
  // If the class is a metaclass, it points to the *root* metaclass.
  //
  // The superclass pointer will always will always point to the superclass
  // of either the class or meta class.
  ConstantList members;
  if (meta) {

    // Find root meta-class.
    const ClassDeclaration *metaDecl = &cd;
    while(metaDecl->baseClass)
      metaDecl = metaDecl->baseClass;
    
    members.push_back(getClassSymbol(*metaDecl, true));
  } else {

    // Both register class and push metaclass on as the isa pointer.
    classes.push_back(const_cast<ClassDeclaration *>(&cd));
    members.push_back(getClassSymbol(cd, true));
  }

  // Set the superclass field.
  members.push_back(
    wrapNull(cd.baseClass ? getClassSymbol(*cd.baseClass, meta) : nullptr)
  );

  // cache_t and class_data_bits_t
  members.push_back(getEmptyCache());
  members.push_back(getNullPtr());

  // Attach Class read-only struct ref.
  members.push_back(
    wrapNull(getClassRoSymbol(cd, meta))
  );

  // Cache it.
  auto var = getGlobalWithBytes(module, name, members);
  classSymbolTable[name] = var;
  retain(var);
  return var;
}

LLGlobalVariable *ObjCState::getClassRoName(const ClassDeclaration& cd) {
  llvm::StringRef name(cd.toChars());
  auto it = classRoNameTable.find(name);
  if (it != classRoNameTable.end()) {
    return it->second;
  }

  auto var = getCStringVar("OBJC_CLASS_NAME", name, classNameSection);
  classRoNameTable[name] = var;
  retain(var);
  return var;
}

LLGlobalVariable *ObjCState::getClassRoSymbol(const ClassDeclaration& cd, bool meta) {
  auto name = getObjcClassRoSymbol(cd, meta);
  auto it = classRoSymbolTable.find(name);
  if (it != classRoSymbolTable.end()) {
    return it->second;
  }

  // Class read-only table layout is as follows:
  // struct class_ro_t {
  //     uint32_t flags;
  //     uint32_t instanceStart;
  //     uint32_t instanceSize;
  //     uint32_t reserved; // Only on 64 bit platforms!
  //     const uint8_t * ivarLayout;
  //     const char * name;
  //     method_list_t * baseMethodList;
  //     protocol_list_t * baseProtocols;
  //     const ivar_list_t * ivars;
  //     const uint8_t * weakIvarLayout;
  //     property_list_t *baseProperties;
  // };
  std::vector<llvm::Constant*> members;
  members.push_back(DtoConstUint(getClassFlags(cd))); // flags
  members.push_back(DtoConstUint(0)); // instanceStart
  members.push_back(DtoConstUint(0)); // instanceSize
  if (getPointerSizeInBits() == 64)
    members.push_back(DtoConstUint(0)); // reserved
  members.push_back(getNullPtr()); // ivarLayout
  members.push_back(getClassRoName(cd)); // name
  members.push_back(wrapNull(getMethodListFor(cd, meta, false))); // baseMethodList
  members.push_back(wrapNull(getProtocolListFor(cd))); // baseProtocols

  if (meta) {
    members.push_back(DtoConstUint(0)); // ivars
    members.push_back(DtoConstUint(0)); // weakIvarLayout
    members.push_back(DtoConstUint(0)); // baseProperties
  } else {
    auto ivarList = getIVarListFor(cd);

    members.push_back(ivarList);        // ivars
    members.push_back(DtoConstUint(0)); // weakIvarLayout

    // TODO: Implement Objective-C properties.
    members.push_back(DtoConstUint(0)); // baseProperties
  }

  
  auto var = getGlobalWithBytes(module, name, members);
  var->setSection(constSection);
  classRoSymbolTable[name] = var;
  retain(var);
  return var;
}

LLGlobalVariable *ObjCState::getClassReference(const ClassDeclaration& cd) {
  llvm::StringRef name(cd.objc.identifier->toChars());
  auto it = classReferenceTable.find(name);
  if (it != classReferenceTable.end()) {
    return it->second;
  }

  auto gvar = getClassSymbol(cd);
  auto classref = new LLGlobalVariable(
      module, gvar->getType(),
      false,
      LLGlobalValue::PrivateLinkage, gvar, "OBJC_CLASSLIST_REFERENCES_$_", nullptr,
      LLGlobalVariable::NotThreadLocal, 0,
      true
  ); // externally initialized
  
  classref->setSection(classRefsSection);

  // Save for later lookup and prevent optimizer elimination
  classReferenceTable[name] = classref;
  retain(classref);
  return classref;
}

//
//      CATEGORIES
//



//
//      PROTOCOLS
//

LLGlobalVariable *ObjCState::getMethodName(const llvm::StringRef &name) {
  auto it = methodNameTable.find(name);
  if (it != methodNameTable.end()) {
    return it->second;
  }

  auto var = getCStringVar("OBJC_METH_VAR_NAME_", name, methodNameSection);
  methodNameTable[name] = var;
  retain(var);
  return var;
}

LLGlobalVariable *ObjCState::getProtocolListFor(const ClassDeclaration& cd) {
  llvm::StringRef name(cd.objc.identifier->toChars());
  auto it = protocolListTable.find(name);
  if (it != protocolListTable.end()) {
    return it->second;
  }

  // Protocol list layout is as follows:
  // struct protocol_list_t {
  //   uintptr_t count; // count is 64-bit by accident. 
  //   protocol_ref_t list[0]; // variable-size
  // }
  std::vector<llvm::Constant*> members;
  members.push_back(constU64(cd.interfaces.length));
  for(size_t i; i < cd.interfaces.length; i++) {
    auto iface = cd.interfaces[i]->sym->isInterfaceDeclaration();
    members.push_back(getProtocolReference(*iface));
  }

  // Cache and return.
  auto var = getGlobalWithBytes(module, name, members);
  protocolListTable[name] = var;
  retain(var);
  return var;
}

LLGlobalVariable *ObjCState::getProtocolSymbol(const InterfaceDeclaration& iface) {
  llvm::StringRef name(iface.objc.identifier->toChars());
  auto it = protocolTable.find(name);
  if (it != protocolTable.end()) {
    return it->second;
  }

  // See: https://github.com/opensource-apple/objc4/blob/master/runtime/objc-runtime-new.h#L277
  // 
  // Protocol symbol layout is as follows:
  // struct protocol_t {
  //   Class isa; // inherited from objc_object
  //   const char *mangledName;
  //   protocol_list_t *protocols; // This list is seperate from the module-level protocol list!
  //   method_list_t *instanceMethods;
  //   method_list_t *classMethods;
  //   method_list_t *optionalInstanceMethods;
  //   method_list_t *optionalClassMethods;
  //   property_list_t *instanceProperties;
  //   uint32_t size;   // sizeof(protocol_t)
  //   uint32_t flags;
  // }
  std::vector<llvm::Constant*> members;

  size_t protocolTSize = (getPointerSize()*8)+8;

  auto nameConst = llvm::ConstantDataArray::getString(module.getContext(), name);
  members.push_back(getNullPtr());              // unused?
  members.push_back(nameConst);                 // mangledName
  members.push_back(getProtocolListFor(iface)); // protocols
  members.push_back(getMethodListFor(iface, false)); // instanceMethods
  members.push_back(getMethodListFor(iface, true));  // classMethods
  members.push_back(getNullPtr()); // TODO: optionalInstanceMethods
  members.push_back(getNullPtr()); // TODO: optionalClassMethods
  members.push_back(getNullPtr()); // TODO: instanceProperties
  members.push_back(DtoConstUint(protocolTSize));
  members.push_back(DtoConstUint(0)); // Should always be 0, other values are reserved for runtime.

  auto var = getGlobalWithBytes(module, name, members);
  protocolTable[name] = var;
  retain(var);
}

LLGlobalVariable *ObjCState::getProtocolReference(const InterfaceDeclaration& iface) {
  llvm::StringRef name(iface.objc.identifier->toChars());
  auto it = protocolReferenceTable.find(name);
  if (it != protocolReferenceTable.end()) {
    return it->second;
  }

  auto gvar = getProtocolSymbol(iface);
  auto protoref = new LLGlobalVariable(
      module, gvar->getType(),
      false, // prevent const elimination optimization
      LLGlobalValue::PrivateLinkage, gvar, "OBJC_PROTOLIST_REFERENCES_$_", nullptr,
      LLGlobalVariable::NotThreadLocal, 0,
      true
  ); // externally initialized
  
  protoref->setSection(protoRefsSection);

  // Save for later lookup and prevent optimizer elimination
  protocolReferenceTable[name] = protoref;
  retain(protoref);
  return protoref;
}

//
//      INSTANCE VARIABLES
//
LLGlobalVariable *ObjCState::getIVarListFor(const ClassDeclaration& cd) {
  auto name = getObjcSymbolName("OBJC_$_INSTANCE_VARIABLES_", cd.objc.identifier->toChars());
  auto it = ivarListTable.find(name);
  if (it != ivarListTable.end()) {
    return it->second;
  }

  // If there's no fields, just return a null constant.
  if (cd.fields.length > 0) {

    auto var = getNullPtr();
    ivarListTable[name] = var;
    return var;
  }

  ConstantList members;
  members.push_back(DtoConstUint(OBJC_IVAR_ENTSIZE));
  members.push_back(DtoConstUint(cd.fields.length));

  for(size_t i; i < cd.fields.length; i++) {
    if (auto vd = cd.fields[i]->isVarDeclaration()) (
      members.push_back(getIVarSymbol(cd, *vd));
    )
  }

  auto var = getGlobalWithBytes(module, globalName, members);
  var->setSection(constSection);
  ivarListTable[name] = var;
  retain(var);
  return var;
}

LLGlobalVariable *ObjCState::getIVarSymbol(const ClassDeclaration& cd, const VarDeclaration& var) {
  // struct ivar_t {
  //   int32_t *offset;
  //   const char *name;
  //   const char *type;
  //   // alignment is sometimes -1; use alignment() instead
  //   uint32_t alignment_raw;
  //   uint32_t size;
  // }
  auto name = getObjcIvarSymbol(cd, var);
  auto it = ivarTable.find(name);
  if (it != ivarTable.end()) {
    return it->second;
  }

  ConstantList members;
  members.push_back(getIVarOffset(cd, var, false));
  members.push_back(getMethodVarName(var.ident->toChars()));
  members.push_back(getTypeEncoding(var.type));
  members.push_back(DtoConstUint(var.alignment.isDefault() ? -1 : var.alignment.get()));
  members.push_back(DtoConstUint(var.size(var.loc)));
  
  auto var = getGlobalWithBytes(module, name, members);
  ivarTable[name] = var;
  retain(var);
  return var;
}

llvm::GlobalVariable *ObjCState::getIVarOffset(const ClassDeclaration& cd, const VarDeclaration& var, bool outputSymbol) {
  auto nameBuf = std::string("OBJC_IVAR_$_", cd.objc.identifier->toChars(), ".", var.iden->toChars());
  auto name = llvm::StringRef(nameBuf);
  auto it = ivarOffsetTable.find(name);
  if (it != ivarOffsetTable.end()) {
    return it->second;
  }
  
  LLGlobalVariable *var;
  if (cd.objc.isMeta) {

    var = getGlobal(module, name);
    ivarOffsetTable[name] = var;
    retain(var);
    return var;
  }

  ConstantList members;
  members.push_back(DtoConstUlong(var.offset));

  var = getGlobalWithBytes(module, name, members);
  ivarOffsetTable[name] = var;
  retain(var);
  return var;
}

//
//      METHODS
//

LLGlobalVariable *ObjCState::getMethodVarName(const llvm::StringRef &name) {
  auto it = methodNameTable.find(name);
  if (it != methodNameTable.end()) {
    return it->second;
  }

  auto var = getCStringVar("OBJC_METH_VAR_NAME_", name, methodNameSection);
  methodNameTable[name] = var;
  retain(var);
  return var;
}

LLGlobalVariable *ObjCState::getMethodListFor(const ClassDeclaration& cd, bool meta, bool optional=false) {
  llvm::StringRef name(cd.objc.identifier->toChars());
  auto it = methodListTable.find(name);
  if (it != methodListTable.end()) {
    return it->second;
  }

  auto methods = meta ? cd.objc.metaclass->objc.methodList : cd.objc.methodList;

  // Count the amount of methods with a body.
  size_t methodCount;
  for(size_t i; i < methods.length; i++) {
    if (methods[i]->fbody) methodCount++;
  }

  // Empty classes don't need a method list generated.
  if (!methodCount)
    return nullptr;
  
  ConstantList members;

  // See: https://github.com/opensource-apple/objc4/blob/master/runtime/objc-runtime-new.h#L93
  members.push_back(DtoConstUint(OBJC_METHOD_SIZEOF));
  members.push_back(DtoConstUint(methodCount));
  for(size_t i; i < methods.length; i++) {
    if (methods[i]->fbody) {
      auto selector = methods[i]->objc.selector;

      // See: https://github.com/opensource-apple/objc4/blob/master/runtime/objc-runtime-new.h#L207
      llvm::StringRef name(selector->stringvalue, selector->stringlen);
      members.push_back(getMethodSymbol(name));
      members.push_back(getMethodType(name));
      members.push_back(DtoCallee(method));
    }
  }

  auto var = getGlobalWithBytes(module, getObjcMethodListSymbol(cd, meta), members);
  methodListTable[name] = var;
  retain(var);
  return var;
}

LLGlobalVariable *ObjCState::getMethodVarType(const llvm::StringRef& name) {
  auto it = methodTypeTable.find(name);
  if (it != methodTypeTable.end()) {
    return it->second;
  }

  auto var = getCStringVar("OBJC_METH_VAR_TYPE_", encoding, methodTypeSection);
  methodTypeTable[name] = var;
  retain(var);
  return var;
}

LLGlobalVariable *ObjCState::getSelector(const ObjcSelector &sel) {
  llvm::StringRef name(sel.stringvalue, sel.stringlen);
  auto it = selectorTable.find(name);
  if (it != selectorTable.end()) {
      return it->second;
  }

  auto gvar = getMethodName(name);
  auto selref = new LLGlobalVariable(
      module, gvar->getType(),
      false, // prevent const elimination optimization
      LLGlobalValue::PrivateLinkage, gvar, "OBJC_SELECTOR_REFERENCES_", nullptr,
      LLGlobalVariable::NotThreadLocal, 0,
      true
  ); // externally initialized
  
  selref->setSection(selectorRefsSection);

  // Save for later lookup and prevent optimizer elimination
  selectorTable[name] = selref;
  retain(selref);
  return selref;
}

//
//    FINALIZATION
//

void ObjCState::retain(LLConstant *sym) {
  retainedSymbols.push_back(sym);
}

llvm::Constant *ObjCState::finalizeClasses() {

  // Objective-C needs to know which classes are in the output
  // As such a protocol list needs to be generated.
  std::vector<llvm::Constant *> members;
  for(auto classRef = classes.begin(); classRef != classes.end(); ++classRef) {
    auto klass = *classRef;
    if (!klass->objc.isExtern && !klass->objc.isMeta) {
      members.push_back(getClassSymbol(*klass));
    }
  }

	auto var = getGlobalWithBytes(module, "OBJC_CLASS_$_", members);
	var->setSection(classListSection);
  return var;
}

llvm::Constant *ObjCState::finalizeProtocols() {

  // Objective-C needs to know which protocols are in the output
  // As such a protocol list needs to be generated.
  std::vector<llvm::Constant *> members;
  for(auto protoRef = protocols.begin(); protoRef != protocols.end(); ++protoRef) {
    auto proto = *protoRef;
    if (!proto->objc.isExtern) {
      members.push_back(getProtocolSymbol(*proto));
    }
  }

	auto var = getGlobalWithBytes(module, "OBJC_PROTOCOL_$_", members);
	var->setSection(protoListSection);
  return var;
}

void ObjCState::finalize() {
  if (!retainedSymbols.empty()) {

    retainedSymbols.push_back(finalizeProtocols());
    retainedSymbols.push_back(finalizeClasses());

    genImageInfo();

    // add in references so optimizer won't remove symbols.
    retainSymbols();
  }
}

void ObjCState::genImageInfo() {
  // Use LLVM to generate image info
  module.addModuleFlag(llvm::Module::Error, "Objective-C Version", 2); // Only support ABI 2. (Non-fragile)
  module.addModuleFlag(llvm::Module::Error, "Objective-C Image Info Version",
                      0u); // version
  module.addModuleFlag(llvm::Module::Error, "Objective-C Image Info Section",
                      llvm::MDString::get(module.getContext(), imageInfoSectionName));
  module.addModuleFlag(llvm::Module::Override, "Objective-C Garbage Collection",
                      0u); // flags
}

void ObjCState::retainSymbols() {

  // put all objc symbols in the llvm.compiler.used array so optimizer won't
  // remove.
  auto arrayType = LLArrayType::get(retainedSymbols.front()->getType(),
                                  retainedSymbols.size());
  auto usedArray = LLConstantArray::get(arrayType, retainedSymbols);
  auto var = new LLGlobalVariable(module, arrayType, false,
                                  LLGlobalValue::AppendingLinkage, usedArray,
                                  "llvm.compiler.used");
  var->setSection("llvm.metadata");
}
