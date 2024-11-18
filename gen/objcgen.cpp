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
#include "gen/irstate.h"

#define OBJC_CLASS_LIST = "__DATA,__objc_classlist";
#define OBJC_PROTOCOL_LIST = "__DATA,__objc_protolist";

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

const char *ObjCState::getObjcType(Type *t) {
  switch (t->ty) {
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
    case TY::Tclass: return "@";
    default: return "?"; // unknown
  }
}

llvm::GlobalVariable *getGlobal(
    llvm::Module& module, 
    llvm::StringRef& name, 
    llvm::Type* type = nullptr
) {
    if (type == nullptr)
        type = llvm::PointerType::get(llvm::Type::getVoidTy(module.getContext()), 0);
    
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

llvm::GlobalVariable *getGlobalWithBytes(
    llvm::Module& module,
    llvm::StringRef name,
    std::vector<llvm::Constant*> packedContents
) {
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
    
    var->setSection("__DATA,objc_data,regular");
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

llvm::Constant *ObjCState::constU32(uint32_t value) {
  return llvm::ConstantInt::get(
    llvm::Type::getInt32Ty(module.getContext()),
    value
  );
}

llvm::Constant *ObjCState::constU64(uint64_t value) {
  return llvm::ConstantInt::get(
    llvm::Type::getInt64Ty(module.getContext()),
    value
  );
}

llvm::Constant *ObjCState::constSizeT(size_t value) {
  return llvm::ConstantInt::get(
    llvm::Type::getIntNTy(module.getContext(), module.getDataLayout().getPointerSizeInBits()),
    value
  );
}

//
//      CLASSES
//
LLGlobalVariable *ObjCState::getMethodListFor(const ClassDeclaration& cd, bool meta) {

}

LLGlobalVariable *ObjCState::getClassSymbol(const ClassDeclaration& cd, bool meta) {

}

LLGlobalVariable *ObjCState::getClassRoSymbol(const ClassDeclaration& cd, bool meta) {

}

LLGlobalVariable *ObjCState::getClassReference(const ClassDeclaration& cd) {

}



//
//      PROTOCOLS
//

LLGlobalVariable *ObjCState::getProtocoList(const InterfaceDeclaration& iface) {

}

LLGlobalVariable *ObjCState::getProtocolSymbol(const InterfaceDeclaration& iface) {
  
  llvm::StringRef name(iface.ident->toChars());
  auto it = protocolTable.find(name);
  if (it != protocolTable.end()) {
    return it->second;
  }

  std::vector<llvm::Constant*> members;
}

LLGlobalVariable *ObjCState::getProtocolReference(const InterfaceDeclaration& iface) {

}



//
//      METHODS
//

LLGlobalVariable *ObjCState::getMethodVarType(const llvm::StringRef& ty) {
  auto it = methodVarTypeTable.find(ty);
  if (it != methodVarTypeTable.end()) {
    return it->second;
  }

  auto var = getCStringVar("OBJC_METH_VAR_TYPE_", ty, 
                          "__TEXT,__objc_methtype,cstring_literals");
  methodVarTypeTable[ty] = var;
  retain(var);
  return var;
}

LLGlobalVariable *ObjCState::getMethodVarName(const llvm::StringRef &name) {
  auto it = methodVarNameTable.find(name);
  if (it != methodVarNameTable.end()) {
    return it->second;
  }

  auto var = getCStringVar("OBJC_METH_VAR_NAME_", name, 
                          "__TEXT,__objc_methname,cstring_literals");
  methodVarNameTable[name] = var;
  retain(var);
  return var;
}

LLGlobalVariable *ObjCState::getMethodVarRef(const ObjcSelector &sel) {
  llvm::StringRef s(sel.stringvalue, sel.stringlen);
  auto it = methodVarRefTable.find(s);
  if (it != methodVarRefTable.end()) {
      return it->second;
  }

  auto gvar = getMethodVarName(s);
  auto selref = new LLGlobalVariable(
      module, gvar->getType(),
      false, // prevent const elimination optimization
      LLGlobalValue::PrivateLinkage, gvar, "OBJC_SELECTOR_REFERENCES_", nullptr,
      LLGlobalVariable::NotThreadLocal, 0,
      true
  ); // externally initialized
  
  selref->setSection("__DATA,__objc_selrefs,literal_pointers,no_dead_strip");

  // Save for later lookup and prevent optimizer elimination
  methodVarRefTable[s] = selref;
  retain(selref);

  return selref;
}

//
//    FINALIZATION
//

void ObjCState::retain(LLConstant *sym) {
  retainedSymbols.push_back(sym);
}

void ObjCState::finalize() {
  if (!retainedSymbols.empty()) {
    genImageInfo();

    // add in references so optimizer won't remove symbols.
    retainSymbols();
  }
}

void ObjCState::genImageInfo() {
  // Use LLVM to generate image info
  const char *section = "__DATA,__objc_imageinfo,regular,no_dead_strip";
  module.addModuleFlag(llvm::Module::Error, "Objective-C Version", 2); // Only support ABI 2. (Non-fragile)
  module.addModuleFlag(llvm::Module::Error, "Objective-C Image Info Version",
                      0u); // version
  module.addModuleFlag(llvm::Module::Error, "Objective-C Image Info Section",
                      llvm::MDString::get(module.getContext(), section));
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
