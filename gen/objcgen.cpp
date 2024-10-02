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
#include "dmd/expression.h"
#include "dmd/identifier.h"
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

ObjCState::ObjCState(llvm::Module &module) : module(module) {
  llvm::LLVMContext &c = module.getContext();
  _class_t = llvm::StructType::create(c, "_class_t");
  _objc_cache = llvm::StructType::create(c, "_objc_cache");
  _class_ro_t = llvm::StructType::create(c, "_class_ro_t");
  __method_list_t = llvm::StructType::create(c, "__method_list_t");
  _objc_method = llvm::StructType::create(c, "_objc_method");
  _objc_protocol_list = llvm::StructType::create(c, "_objc_protocol_list");
  _protocol_t = llvm::StructType::create(c, "_protocol_t");
  _ivar_t = llvm::StructType::create(c, "_ivar_t");
  _ivar_list_t = llvm::StructType::create(c, "_ivar_list_t");
  _prop_t = llvm::StructType::create(c, "_prop_t");

  _class_t->setBody(
      {_class_t->getPointerTo(), _class_t->getPointerTo(),
       _objc_cache->getPointerTo(), _class_t->getPointerTo(),
       /// i8* (i8*, i8*)**
       llvm::FunctionType::get(
           llvm::Type::getInt8PtrTy(c),
           {llvm::Type::getInt8PtrTy(c), llvm::Type::getInt8PtrTy(c)}, false)
           ->getPointerTo()
           ->getPointerTo(),
       _class_ro_t->getPointerTo()});

  _class_ro_t->setBody(
      {llvm::Type::getInt32Ty(c), llvm::Type::getInt32Ty(c),
       llvm::Type::getInt32Ty(c), llvm::Type::getInt8PtrTy(c),
       llvm::Type::getInt8PtrTy(c), __method_list_t->getPointerTo(),
       _objc_protocol_list->getPointerTo(), _ivar_list_t->getPointerTo(),
       llvm::Type::getInt8PtrTy(c), _prop_list_t->getPointerTo()});

  __method_list_t->setBody({llvm::Type::getInt32Ty(c),
                            llvm::Type::getInt32Ty(c),
                            llvm::ArrayType::get(_objc_method, 0)});

  _objc_method->setBody({llvm::Type::getInt8PtrTy(c),
                         llvm::Type::getInt8PtrTy(c),
                         llvm::Type::getInt8PtrTy(c)});

  _objc_protocol_list->setBody(
      {llvm::Type::getInt64Ty(c),
       llvm::ArrayType::get(_protocol_t->getPointerTo(), 0)});

  _protocol_t->setBody({
      llvm::Type::getInt8PtrTy(c),
      llvm::Type::getInt8PtrTy(c),
      _objc_protocol_list->getPointerTo(),
      __method_list_t->getPointerTo(),
      __method_list_t->getPointerTo(),
      __method_list_t->getPointerTo(),
      __method_list_t->getPointerTo(),
      _prop_list_t->getPointerTo(),
      llvm::Type::getInt32Ty(c),
      llvm::Type::getInt32Ty(c),
      llvm::Type::getInt8PtrTy(c)->getPointerTo(),
      llvm::Type::getInt8PtrTy(c),
      _prop_list_t->getPointerTo(),
  });

  _ivar_list_t->setBody({llvm::Type::getInt32Ty(c), llvm::Type::getInt32Ty(c),
                         llvm::ArrayType::get(_ivar_t, 0)});

  _ivar_t->setBody({llvm::Type::getInt64PtrTy(c), llvm::Type::getInt8PtrTy(c),
                    llvm::Type::getInt8PtrTy(c), llvm::Type::getInt32Ty(c),
                    llvm::Type::getInt32Ty(c)});

  _prop_list_t->setBody({llvm::Type::getInt32Ty(c), llvm::Type::getInt32Ty(c),
                         llvm::ArrayType::get(_prop_t, 0)});

  _prop_t->setBody({llvm::Type::getInt8PtrTy(c), llvm::Type::getInt8PtrTy(c)});
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

LLGlobalVariable *ObjCState::classVarRef(const ObjcClassReferenceExp &cre) {
  llvm::StringRef s(std::string("OBJC_CLASS_$_")
                    + cre.classDeclaration->objc.identifier->toChars());

  auto it = methVarRefMap.find(s);
  if (it != methVarRefMap.end()) {
    return it->second;
  }
  auto selref = new LLGlobalVariable(module, _class_t, false, LLGlobalValue::ExternalLinkage, llvm::ConstantPointerNull::get(_class_t->getPointerTo()), s);
  selref->setSection("__DATA,__objc_classrefs");
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
