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
    case llvm::Triple::aarch64: // arm64 iOS, tvOS
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

LLGlobalVariable *ObjCState::getMethodVarName(const llvm::StringRef &name) {
    auto it = methVarNameMap.find(name);
    if (it != methVarNameMap.end()) {
    return it->second;
    }

    auto var = getCStringVar("OBJC_METH_VAR_NAME_", name, 
                            "__TEXT,__objc_methname,cstring_literals");
    methVarNameMap[name] = var;
    retain(var);
    return var;
}

LLGlobalVariable *ObjCState::getMethodVarRef(const ObjcSelector &sel) {
    llvm::StringRef s(sel.stringvalue, sel.stringlen);
    auto it = methodVarRefTable.find(s);
    if (it != methodVarRefTable.end()) {
        return it->second;
    }

    auto gvar = getMethVarName(s);
    auto selref = new LLGlobalVariable(
        module, gvar->getType(),
        false, // prevent const elimination optimization
        LLGlobalValue::PrivateLinkage, gvar, "OBJC_SELECTOR_REFERENCES_", nullptr,
        LLGlobalVariable::NotThreadLocal, 0,
        true
    ); // externally initialized
    
    selref->setSection("__DATA,__objc_selrefs,literal_pointers,no_dead_strip");

    // Save for later lookup and prevent optimizer elimination
    methVarRefMap[s] = selref;
    retain(selref);

    return selref;
}

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
