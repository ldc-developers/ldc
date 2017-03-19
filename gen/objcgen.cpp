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

#include "mtype.h"
#include "objc.h"
#include "gen/irstate.h"
#include "gen/objcgen.h"

namespace {
// Were any Objective-C symbols generated?
bool hasSymbols;

enum ABI {
  none = 0,
  fragile = 1,
  nonFragile = 2
};

ABI abi = nonFragile;

// symbols that shouldn't be optimized away
std::vector<LLConstant *> retainedSymbols;

llvm::StringMap<LLGlobalVariable *> methVarNameMap;
llvm::StringMap<LLGlobalVariable *> methVarRefMap;

void retain(LLConstant *sym) {
    retainedSymbols.push_back(DtoBitCast(sym, getVoidPtrType()));
}

void retainSymbols() {
  // put all objc symbols in the llvm.compiler.used array so optimizer won't
  // remove.  Should do just once per module.
  auto arrayType = LLArrayType::get(getVoidPtrType(), retainedSymbols.size());
  auto usedArray = LLConstantArray::get(arrayType, retainedSymbols);
  auto var = new LLGlobalVariable
    (gIR->module, usedArray->getType(), false,
     LLGlobalValue::AppendingLinkage,
     usedArray,
     "llvm.compiler.used");
  var->setSection("llvm.metadata");
}

void genImageInfo() {
  // Use LLVM to generate image info
  const char *section = (abi == nonFragile ?
                         "__DATA,__objc_imageinfo,regular,no_dead_strip" :
                         "__OBJC,__image_info");
  gIR->module.addModuleFlag(llvm::Module::Error,
                            "Objective-C Version", abi); //  unused?
  gIR->module.addModuleFlag(llvm::Module::Error,
                            "Objective-C Image Info Version", 0u); // version
  gIR->module.addModuleFlag(llvm::Module::Error,
                            "Objective-C Image Info Section",
                            llvm::MDString::get(gIR->context(), section));
  gIR->module.addModuleFlag(llvm::Module::Override,
                            "Objective-C Garbage Collection", 0u); // flags
}

LLGlobalVariable *getCStringVar(const char *symbol,
                                const llvm::StringRef &str,
                                const char *section) {
  auto init = llvm::ConstantDataArray::getString(gIR->context(), str);
  auto var = new LLGlobalVariable
    (gIR->module, init->getType(), false,
     LLGlobalValue::PrivateLinkage, init, symbol);
    var->setSection(section);
    return var;
}

LLGlobalVariable *getMethVarName(const llvm::StringRef &name) {
  auto it = methVarNameMap.find(name);
  if (it != methVarNameMap.end()) {
    return it->second;
  }

  auto var = getCStringVar("OBJC_METH_VAR_NAME_", name,
                           abi == nonFragile ?
                           "__TEXT,__objc_methname,cstring_literals" :
                           "__TEXT,__cstring,cstring_literals");
  methVarNameMap[name] = var;
  retain(var);
  return var;
}
} // end local stuff

bool objc_isSupported(const llvm::Triple &triple) {
  if (triple.isOSDarwin()) {
    // Objective-C only supported on Darwin at this time
    switch (triple.getArch()) {
#if LDC_LLVM_VER == 305
    case llvm::Triple::arm64:
#endif
    case llvm::Triple::aarch64:              // arm64 iOS, tvOS
    case llvm::Triple::arm:                  // armv6 iOS
    case llvm::Triple::thumb:                // thumbv7 iOS, watchOS
    case llvm::Triple::x86_64:               // OSX, iOS, tvOS sim
      abi = nonFragile;
      return true;
    case llvm::Triple::x86:                  // OSX, iOS, watchOS sim
      abi = fragile;
      return true;
    default:
      break;
    }
  }
  return false;
}

// called by ddmd.objc.objc_tryMain_init()
void objc_initSymbols() {
  hasSymbols = false;
  retainedSymbols.clear();
  methVarNameMap.clear();
  methVarRefMap.clear();
}

LLGlobalVariable *objc_getMethVarRef(const ObjcSelector &sel) {
  llvm::StringRef s(sel.stringvalue, sel.stringlen);
  auto it = methVarRefMap.find(s);
  if (it != methVarRefMap.end()) {
    return it->second;
  }

  auto gvar = getMethVarName(s);
  auto selref = new LLGlobalVariable
    (gIR->module, gvar->getType(),
     false,                            // prevent const elimination optimization
     LLGlobalValue::PrivateLinkage,
     gvar,
     "OBJC_SELECTOR_REFERENCES_",
     nullptr, LLGlobalVariable::NotThreadLocal, 0,
     true);                                  // externally initialized
  selref->setSection(abi == nonFragile ?
                     "__DATA,__objc_selrefs,literal_pointers,no_dead_strip" :
                     "__OBJC,__message_refs,literal_pointers,no_dead_strip");

  // Save for later lookup and prevent optimizer elimination
  methVarRefMap[s] = selref;
  retain(selref);
  hasSymbols = true;

  return selref;
}

void objc_Module_genmoduleinfo_classes() {
  if (hasSymbols) {
    genImageInfo();
    // add in references so optimizer won't remove symbols.
    retainSymbols();
  }
}
