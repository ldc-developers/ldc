//===-- objcgen.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "mtype.h"
#include "objc.h"
#include "gen/irstate.h"
#include "gen/objcgen.h"

// Currently generates symbols for the non-fragile ABI

namespace {
bool hasSymbols;

std::vector<LLConstant *> usedSymbols;

llvm::StringMap<LLGlobalVariable *> methVarNameMap;
llvm::StringMap<LLGlobalVariable *> methVarRefMap;

void initSymbols() {
  hasSymbols = false;
  usedSymbols.clear();
  methVarNameMap.clear();
  methVarRefMap.clear();
}

void use(LLConstant *sym)
{
    usedSymbols.push_back(DtoBitCast(sym, getVoidPtrType()));
}

void retainSymbols() {
  // put all objc symbols in the llvm.compiler.used array so optimizer won't
  // remove.  Should do just once per module.
  auto arrayType = LLArrayType::get(getVoidPtrType(), usedSymbols.size());
  auto usedArray = LLConstantArray::get(arrayType, usedSymbols);
  auto var = new LLGlobalVariable
    (gIR->module, usedArray->getType(), false,
     LLGlobalValue::AppendingLinkage,
     usedArray,
     "llvm.compiler.used");
  var->setSection("llvm.metadata");
}

void genImageInfo() {
  // First option uses LLMV to produce image info ala module flags and
  // TargetLoweringObjectFileMachO::emitModuleFlags()
#if 1
  // Use LLVM to generate image info
  const char *section = "__DATA,__objc_imageinfo,regular,no_dead_strip";

  gIR->module.addModuleFlag(llvm::Module::Error,
                            "Objective-C Version", 2); //  unused?
  gIR->module.addModuleFlag(llvm::Module::Error,
                            "Objective-C Image Info Version", 0u); // version
  gIR->module.addModuleFlag(llvm::Module::Error,
                            "Objective-C Image Info Section",
                            llvm::MDString::get(gIR->context(), section));
  gIR->module.addModuleFlag(llvm::Module::Override,
                            "Objective-C Garbage Collection", 0u); // flags
#else
  // Second option just does it directly
  LLType* intType = LLType::getInt32Ty(gIR->context());
  LLConstant* values[2] = {
    LLConstantInt::get(intType, 0),  // version
    LLConstantInt::get(intType, 0)   // flag
  };

  auto data = LLConstantStruct::getAnon(values);
  auto var = new LLGlobalVariable
    (gIR->module, data->getType(), false,
     LLGlobalValue::PrivateLinkage,
     data,
     "OBJC_IMAGE_INFO");
  var->setSection("__DATA,__objc_imageinfo,regular,no_dead_strip");
  use(var);
#endif
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
                            "__TEXT,__objc_methname,cstring_literals");
  methVarNameMap[name] = var;
  use(var);
  return var;
}

} // end local stuff

void objc_init() {
  // TODO: fix so resolved with ddmd/objc.d version
  initSymbols();
  ObjcSelector::_init();
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
  selref->setSection("__DATA,__objc_selrefs,literal_pointers,no_dead_strip");

  // Save for later lookup and prevent optimizer elimination
  methVarRefMap[s] = selref;
  use(selref);
  hasSymbols = true;

  return selref;
}

const char* objc_getMsgSend(Type *ret, bool hasHiddenArg) {
  // TODO: Need the abi to vote on if objc_msgSend_fpret or fp2ret are used
  if (hasHiddenArg)
    return "objc_msgSend_stret";

  if (ret) {
    // not sure if DMD can handle this
    if (ret->ty == Tcomplex80)          // x86_64 only
      return "objc_msgSend_fp2ret";

    if (ret->ty == Tfloat80)            // x86 and x86_64
      return "objc_msgSend_fpret";
  }

  return "objc_msgSend";
}

void objc_Module_genmoduleinfo_classes() {
  if (hasSymbols) {
    genImageInfo();
    // add in references so optimizer won't remove symbols.
    retainSymbols();
  }
}
