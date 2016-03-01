#include "mtype.h"
#include "objc.h"
#include "gen/irstate.h"
#include "gen/objcgen.h"

namespace {
bool hasSymbols;

void initSymbols() {
  hasSymbols = false;
}

void genImageInfo() {
  // Note: this would normally be produced by LLMV
  // TargetLoweringObjectFileMachO::emitModuleFlags()
  // if module flags are added ala CGObjCMac.cpp

  // Hard coded for ObjCABI version = 0
  LLType* intType = LLType::getInt32Ty(gIR->context());
  LLConstant* values[2] = {
    LLConstantInt::get(intType, 0),  // version
    LLConstantInt::get(intType, 0)   // flag
  };
  LLConstant* data = LLConstantStruct::getAnon(values);

  LLGlobalVariable* s = new LLGlobalVariable
    (gIR->module,
     data->getType(),
     false,
     llvm::GlobalValue::PrivateLinkage,
     data,
     "OBJC_IMAGE_INFO");
  s->setSection("__DATA,__objc_imageinfo,regular,no_dead_strip");
}
} // end local stuff

void objc_init() {
  // TODO: fix so resolved with ddmd/objc.d version
  initSymbols();
  ObjcSelector::_init();
}

llvm::GlobalVariable *objc_getMethVarRef(const char *name, size_t len) {
  // TODO: have better implementation in work from last summer.  Hack it in
  // as proof of concept
  hasSymbols = true;
    
  llvm::StringRef s(name, len);
  LLConstant* init = llvm::ConstantDataArray::getString(gIR->context(), s, true);

  //.section	__TEXT,__objc_methname,cstring_literals
  llvm::GlobalVariable* gvar = new llvm::GlobalVariable
    (gIR->module, init->getType(), true,
     llvm::GlobalValue::PrivateLinkage, init, "OBJC_METH_VAR_NAME_");
  gvar->setSection("__TEXT,__objc_methname,cstring_literals");
  // do we need this?
  gvar->setUnnamedAddr(true);

  //.section	__DATA,__objc_selrefs,literal_pointers,no_dead_strip
  llvm::GlobalVariable* selptr = new llvm::GlobalVariable
    (gIR->module, gvar->getType(), true,
     llvm::GlobalValue::PrivateLinkage,
     gvar, "OBJC_SELECTOR_REFERENCES_");
  selptr->setSection("__DATA,__objc_selrefs,literal_pointers,no_dead_strip");

  return selptr;
}

const char* objc_getMsgSend(Type *ret, bool hasHiddenArg) {
  // Need the abi to vote on if objc_msgSend_fpret or fp2ret are used
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
    //TODO: add in references so optimizer won't remove selector symbols.
    // this is implmented in on my work last summer
    // useSymbols();
  }
}
