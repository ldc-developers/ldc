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

namespace {
enum ABI { none = 0, fragile = 1, nonFragile = 2 };
ABI abi = nonFragile;
}

bool objc_isSupported(const llvm::Triple &triple) {
  if (triple.isOSDarwin()) {
    // Objective-C only supported on Darwin at this time
    switch (triple.getArch()) {
    case llvm::Triple::aarch64: // arm64 iOS, tvOS
    case llvm::Triple::arm:     // armv6 iOS
    case llvm::Triple::thumb:   // thumbv7 iOS, watchOS
    case llvm::Triple::x86_64:  // OSX, iOS, tvOS sim
      abi = nonFragile;
      return true;
    case llvm::Triple::x86: // OSX, iOS, watchOS sim
      abi = fragile;
      return true;
    default:
      break;
    }
  }
  return false;
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

LLGlobalVariable *ObjCState::getMethVarName(const llvm::StringRef &name) {
  auto it = methVarNameMap.find(name);
  if (it != methVarNameMap.end()) {
    return it->second;
  }

  auto var = getCStringVar("OBJC_METH_VAR_NAME_", name,
                           abi == nonFragile
                               ? "__TEXT,__objc_methname,cstring_literals"
                               : "__TEXT,__cstring,cstring_literals");
  methVarNameMap[name] = var;
  retain(var);
  return var;
}

LLGlobalVariable *ObjCState::getMethVarRef(const ObjcSelector &sel) {
  llvm::StringRef s(sel.stringvalue, sel.stringlen);
  auto it = methVarRefMap.find(s);
  if (it != methVarRefMap.end()) {
    return it->second;
  }

  auto gvar = getMethVarName(s);
  auto selref = new LLGlobalVariable(
      module, gvar->getType(),
      false, // prevent const elimination optimization
      LLGlobalValue::PrivateLinkage, gvar, "OBJC_SELECTOR_REFERENCES_", nullptr,
      LLGlobalVariable::NotThreadLocal, 0,
      true); // externally initialized
  selref->setSection(
      abi == nonFragile
          ? "__DATA,__objc_selrefs,literal_pointers,no_dead_strip"
          : "__OBJC,__message_refs,literal_pointers,no_dead_strip");

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
  const char *section =
      (abi == nonFragile ? "__DATA,__objc_imageinfo,regular,no_dead_strip"
                         : "__OBJC,__image_info");
  module.addModuleFlag(llvm::Module::Error, "Objective-C Version",
                       abi); //  unused?
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
