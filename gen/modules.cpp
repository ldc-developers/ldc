//===-- modules.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "module.h"
#include "aggregate.h"
#include "attrib.h"
#include "declaration.h"
#include "enum.h"
#include "id.h"
#include "import.h"
#include "init.h"
#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "scope.h"
#include "statement.h"
#include "target.h"
#include "template.h"
#include "driver/cl_options_instrumentation.h"
#include "gen/abi.h"
#include "gen/arrays.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/mangling.h"
#include "gen/moduleinfo.h"
#include "gen/optimizer.h"
#include "gen/runtime.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "ir/irdsymbol.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"
#include "ir/irvar.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#if _AIX || __sun
#include <alloca.h>
#endif

static llvm::cl::opt<bool, true>
    preservePaths("op", llvm::cl::ZeroOrMore,
                  llvm::cl::desc("Preserve source path for output files"),
                  llvm::cl::location(global.params.preservePaths));

static llvm::cl::opt<bool, true>
    fqnNames("oq", llvm::cl::ZeroOrMore,
             llvm::cl::desc("Write object files with fully qualified names"),
             llvm::cl::location(global.params.fullyQualifiedObjectFiles));

void Module::checkAndAddOutputFile(File *file) {
  static std::map<std::string, Module *> files;

  std::string key(file->name->str);
  auto i = files.find(key);
  if (i != files.end()) {
    Module *previousMod = i->second;
    ::error(Loc(),
            "Output file '%s' for module `%s` collides with previous "
            "module `%s`. See the -oq option",
            key.c_str(), toPrettyChars(), previousMod->toPrettyChars());
    fatal();
  }

  files.emplace(std::move(key), this);
}

void Module::makeObjectFilenameUnique() {
  assert(objfile);

  const char *ext = FileName::ext(objfile->name->str);
  const char *stem = FileName::removeExt(objfile->name->str);

  llvm::SmallString<128> unique;
  auto EC = llvm::sys::fs::createUniqueFile(
      llvm::Twine(stem) + "-%%%%%%%." + ext, unique);
  if (!EC) // success
    objfile->name->str = mem.xstrdup(unique.c_str());
}

namespace {
/// Ways the druntime module registry system can be implemented.
enum class RegistryStyle {
  /// Modules are inserted into a linked list starting at the _Dmodule_ref
  /// global.
  legacyLinkedList,

  /// Module references are emitted into the .minfo section.
  sectionMSVC,

  /// Module references are emitted into the .minfo section. Global
  /// constructors/destructors make sure _d_dso_registry is invoked once per ELF
  /// object.
  sectionELF,

  /// Module references are emitted into the .minfo section. Global
  /// constructors/destructors make sure _d_dso_registry is invoked once per
  /// shared object. A "TLS anchor" function to identify the TLS range
  /// corresponding to this image is also passed to druntime.
  sectionDarwin
};

/// Returns the module registry style to use for the current target triple.
RegistryStyle getModuleRegistryStyle() {
  const auto t = global.params.targetTriple;

  if (t->isWindowsMSVCEnvironment()) {
    return RegistryStyle::sectionMSVC;
  }

  if (t->isMacOSX()) {
    return RegistryStyle::sectionDarwin;
  }

  if (t->isOSLinux() || t->isOSFreeBSD() ||
      t->isOSNetBSD() || t->isOSOpenBSD() || t->isOSDragonFly()) {
    return RegistryStyle::sectionELF;
  }

  return RegistryStyle::legacyLinkedList;
}

/// Build ModuleReference and register function, to register the module info in
/// the global linked list.
///
/// Implements getModuleRegistryStyle() == RegistryStyle::legacyLinkedList.
LLFunction *build_module_reference_and_ctor(const char *moduleMangle,
                                            LLConstant *moduleinfo) {
  // build ctor type
  LLFunctionType *fty = LLFunctionType::get(LLType::getVoidTy(gIR->context()),
                                            std::vector<LLType *>(), false);

  // build ctor name
  std::string fname = "_D";
  fname += moduleMangle;
  fname += "16__moduleinfoCtorZ";

  // build a function that registers the moduleinfo in the global moduleinfo
  // linked list
  LLFunction *ctor =
      LLFunction::Create(fty, LLGlobalValue::InternalLinkage,
                         getIRMangledFuncName(fname, LINKd), &gIR->module);

  // provide the default initializer
  LLStructType *modulerefTy = DtoModuleReferenceType();
  LLConstant *mrefvalues[] = {
      LLConstant::getNullValue(modulerefTy->getContainedType(0)),
      llvm::ConstantExpr::getBitCast(moduleinfo,
                                     modulerefTy->getContainedType(1))};
  LLConstant *thismrefinit = LLConstantStruct::get(
      modulerefTy, llvm::ArrayRef<LLConstant *>(mrefvalues));

  // create the ModuleReference node for this module
  const auto thismrefIRMangle = getIRMangledModuleRefSymbolName(moduleMangle);
  LLGlobalVariable *thismref =
      defineGlobal(Loc(), gIR->module, thismrefIRMangle, thismrefinit,
                   LLGlobalValue::InternalLinkage, false);
  // make sure _Dmodule_ref is declared
  const auto mrefIRMangle = getIRMangledVarName("_Dmodule_ref", LINKc);
  LLConstant *mref = gIR->module.getNamedGlobal(mrefIRMangle);
  LLType *modulerefPtrTy = getPtrToType(modulerefTy);
  if (!mref) {
    mref = declareGlobal(Loc(), gIR->module, modulerefPtrTy, mrefIRMangle, false);
  }
  mref = DtoBitCast(mref, getPtrToType(modulerefPtrTy));

  // make the function insert this moduleinfo as the beginning of the
  // _Dmodule_ref linked list
  llvm::BasicBlock *bb =
      llvm::BasicBlock::Create(gIR->context(), "moduleinfoCtorEntry", ctor);
  IRBuilder<> builder(bb);

  // debug info
  gIR->DBuilder.EmitModuleCTor(ctor, fname.c_str());

  // get current beginning
  LLValue *curbeg = builder.CreateLoad(mref, "current");

  // put current beginning as the next of this one
  LLValue *gep = builder.CreateStructGEP(
      modulerefTy, thismref, 0, "next");
  builder.CreateStore(curbeg, gep);

  // replace beginning
  builder.CreateStore(thismref, mref);

  // return
  builder.CreateRetVoid();

  return ctor;
}

/// Builds a void*() function with hidden visibility that returns the address of
/// a dummy TLS global (also with hidden visibility).
///
/// The global is non-zero-initialised and aligned to 16 bytes.
llvm::Function *buildGetTLSAnchor() {
  // Create a dummmy TLS global private to this module.
  const auto one =
      llvm::ConstantInt::get(llvm::Type::getInt8Ty(gIR->context()), 1);
  const auto anchor = defineGlobal(Loc(), gIR->module, "ldc.tls_anchor", one,
                                   llvm::GlobalValue::LinkOnceODRLinkage, false,
                                   /*isThreadLocal=*/true);
  anchor->setVisibility(llvm::GlobalValue::HiddenVisibility);
  anchor->setAlignment(16);

  const auto getAnchor =
      llvm::Function::Create(llvm::FunctionType::get(getVoidPtrType(), false),
                             llvm::GlobalValue::LinkOnceODRLinkage,
                             "ldc.get_tls_anchor", &gIR->module);
  getAnchor->setVisibility(llvm::GlobalValue::HiddenVisibility);

  IRBuilder<> builder(llvm::BasicBlock::Create(gIR->context(), "", getAnchor));
  builder.CreateRet(anchor);

  return getAnchor;
}

/// Builds the ldc.register_dso function, which is called by the global
/// {c, d}tors to invoke _d_dso_registry.
///
/// Pseudocode:
/// void ldc.register_dso(bool isShutdown, void* minfoUsedPointer) {
///   if (dsoInitialized == isShutdown) {
///     dsoInitialized = !isShutdown;
///     auto record = {1, dsoSlot, minfoBeg, minfoEnd[, getTlsAnchor],
///       minfoUsedPointer};
///     _d_dso_registry(cast(CompilerDSOData*)&record);
///   }
/// }
///
/// On Darwin platforms, the record contains an extra pointer to a function
/// which returns the address of a TLS global.
llvm::Function *buildRegisterDSO(RegistryStyle style,
                                 llvm::Value *dsoInitialized,
                                 llvm::Value *dsoSlot, llvm::Value *minfoBeg,
                                 llvm::Value *minfoEnd) {
  llvm::Type *argTypes[] = {llvm::Type::getInt1Ty(gIR->context()),
                            llvm::Type::getInt8PtrTy(gIR->context())};
  const auto fnType = llvm::FunctionType::get(
      llvm::Type::getVoidTy(gIR->context()), argTypes, false);
  const auto fn =
      llvm::Function::Create(fnType, llvm::GlobalValue::LinkOnceODRLinkage,
                             "ldc.register_dso", &gIR->module);
  fn->setVisibility(llvm::GlobalValue::HiddenVisibility);
  auto argIt = fn->arg_begin();
  const auto isShutdown = &*argIt;
  isShutdown->setName("isShutdown");
  ++argIt;
  const auto minfoUsedPointer = &*argIt;
  minfoUsedPointer->setName("minfoUsedPointer");

  // Never inline – the functions is only called on startup/shutdown, hence
  // it isn't worth the increase in code size.
  fn->addFnAttr(llvm::Attribute::NoInline);

  const auto dsoRegistry =
      getRuntimeFunction(Loc(), gIR->module, "_d_dso_registry");
  const auto recordPtrTy = dsoRegistry->getFunctionType()->getContainedType(1);

  llvm::Function *getTlsAnchorPtr = nullptr;
  if (style == RegistryStyle::sectionDarwin) {
    getTlsAnchorPtr = buildGetTLSAnchor();
  }

  const auto entryBB = llvm::BasicBlock::Create(gIR->context(), "", fn);
  const auto initBB = llvm::BasicBlock::Create(gIR->context(), "init", fn);
  const auto endBB = llvm::BasicBlock::Create(gIR->context(), "end", fn);

  {
    IRBuilder<> b(entryBB);
    const auto loadedFlag =
        b.CreateTrunc(b.CreateLoad(dsoInitialized), b.getInt1Ty());
    const auto condEval =
        b.CreateICmp(llvm::ICmpInst::ICMP_EQ, loadedFlag, isShutdown);
    b.CreateCondBr(condEval, initBB, endBB);
  }
  {
    IRBuilder<> b(initBB);
    const auto newFlag = b.CreateXor(isShutdown, b.getTrue());
    b.CreateStore(b.CreateZExt(newFlag, b.getInt8Ty()), dsoInitialized);

    llvm::Constant *version = DtoConstSize_t(1);
    llvm::SmallVector<llvm::Type *, 6> memberTypes;
    memberTypes.push_back(version->getType());
    memberTypes.push_back(dsoSlot->getType());
    memberTypes.push_back(minfoBeg->getType());
    memberTypes.push_back(minfoEnd->getType());
    if (style == RegistryStyle::sectionDarwin) {
      memberTypes.push_back(getTlsAnchorPtr->getType());
    }
    memberTypes.push_back(minfoUsedPointer->getType());
    llvm::StructType *stype =
        llvm::StructType::get(gIR->context(), memberTypes, false);
    llvm::Value *record = b.CreateAlloca(stype);

    unsigned i = 0;
    b.CreateStore(version, b.CreateStructGEP(stype, record, i++));
    b.CreateStore(dsoSlot, b.CreateStructGEP(stype, record, i++));
    b.CreateStore(minfoBeg, b.CreateStructGEP(stype, record, i++));
    b.CreateStore(minfoEnd, b.CreateStructGEP(stype, record, i++));
    if (style == RegistryStyle::sectionDarwin) {
      b.CreateStore(getTlsAnchorPtr, b.CreateStructGEP(stype, record, i++));
    }
    b.CreateStore(minfoUsedPointer, b.CreateStructGEP(stype, record, i++));

    b.CreateCall(dsoRegistry, b.CreateBitCast(record, recordPtrTy));
    b.CreateBr(endBB);
  }
  {
    IRBuilder<> b(endBB);
    b.CreateRetVoid();
  }

  return fn;
}

void emitModuleRefToSection(RegistryStyle style, std::string moduleMangle,
                            llvm::Constant *thisModuleInfo) {
  assert(style == RegistryStyle::sectionMSVC ||
         style == RegistryStyle::sectionELF ||
         style == RegistryStyle::sectionDarwin);
  // Only for the first D module to be emitted into this llvm::Module we need to
  // create the global ctors/dtors. The magic linker symbols used to get the
  // start and end of the .minfo section also only need to be emitted for the
  // first D module.
  // For all subsequent ones, we just need to emit an additional reference into
  // the .minfo section (even with --gc-sections, the section is already kept
  // alive by the first module's reference being used in the ctor/dtor
  // functions).
  const bool isFirst = !gIR->module.getGlobalVariable("ldc.dso_slot");

  llvm::Type *const moduleInfoPtrTy = DtoPtrToType(getModuleInfoType());
  const auto sectionName =
      style == RegistryStyle::sectionMSVC
          ? ".minfo"
          : style == RegistryStyle::sectionDarwin ? "__DATA,.minfo" : "__minfo";

  const auto thismrefIRMangle =
      getIRMangledModuleRefSymbolName(moduleMangle.c_str());
  auto thismref = defineGlobal(Loc(), gIR->module, thismrefIRMangle,
                               DtoBitCast(thisModuleInfo, moduleInfoPtrTy),
                               llvm::GlobalValue::LinkOnceODRLinkage,
                               false // FIXME: mRelocModel != llvm::Reloc::PIC_
  );
  thismref->setSection(sectionName);
  gIR->usedArray.push_back(thismref);

  // Android doesn't need register_dso and friends- see rt.sections_android-
  // so bail out here.
  if (!isFirst || style == RegistryStyle::sectionMSVC ||
      global.params.targetTriple->getEnvironment() == llvm::Triple::Android) {
    // Nothing left to do.
    return;
  }

  // Use magic linker symbol names to obtain the begin and end of the .minfo
  // section.
  const auto magicBeginSymbolName = (style == RegistryStyle::sectionDarwin)
                                        ? "\1section$start$__DATA$.minfo"
                                        : "__start___minfo";
  const auto magicEndSymbolName = (style == RegistryStyle::sectionDarwin)
                                      ? "\1section$end$__DATA$.minfo"
                                      : "__stop___minfo";
  auto minfoBeg = declareGlobal(Loc(), gIR->module, moduleInfoPtrTy,
                                magicBeginSymbolName, false);
  auto minfoEnd = declareGlobal(Loc(), gIR->module, moduleInfoPtrTy,
                                magicEndSymbolName, false);
  minfoBeg->setVisibility(llvm::GlobalValue::HiddenVisibility);
  minfoEnd->setVisibility(llvm::GlobalValue::HiddenVisibility);

  // Build the ctor to invoke _d_dso_registry.

  // This is the DSO slot for use by the druntime implementation.
  auto dsoSlot = defineGlobal(Loc(), gIR->module, "ldc.dso_slot",
                              getNullPtr(getVoidPtrType()),
                              llvm::GlobalValue::LinkOnceODRLinkage, false);
  dsoSlot->setVisibility(llvm::GlobalValue::HiddenVisibility);

  // Okay, so the theory is easy: We want to have one global constructor and
  // destructor per object (i.e. executable/shared library) that calls
  // _d_dso_registry with the respective DSO record. However, there are a
  // couple of issues that make this harder than necessary:
  //
  //  1) The natural way to implement the "one-per-image" part would be to
  //     emit a weak reference to a weak function into a .ctors.<somename>
  //     section (llvm.global_ctors doesn't support the necessary
  //     functionality, so we'd use our knowledge of the linker script to work
  //     around that). But as of LLVM 3.4, emitting a symbol both as weak and
  //     into a custom section is not supported by the MC layer. Thus, we have
  //     to use a normal ctor/dtor and manually ensure that we only perform
  //     the call once. This is done by introducing ldc.dso_initialized.
  //
  //  2) To make sure the .minfo section isn't removed by the linker when
  //     using --gc-sections, we need to keep a reference to it around in
  //     _every_ object file (as --gc-sections works per object file). The
  //     natural place for this is the ctor, where we just load a reference
  //     on the stack after the DSO record (to ensure LLVM doesn't optimize
  //     it out). However, this way, we need to have at least one ctor
  //     instance per object file be pulled into the final executable. We
  //     do this here by making the module mangle string part of its name,
  //     even thoguht this is slightly wasteful on -singleobj builds.
  //
  // It might be a better idea to simply use a custom linker script (using
  // INSERT AFTER… so as to still keep the default one) to avoid all these
  // problems. This would mean that it is no longer safe to link D objects
  // directly using e.g. "g++ dcode.o cppcode.o", though.

  auto dsoInitialized = defineGlobal(
      Loc(), gIR->module, "ldc.dso_initialized",
      llvm::ConstantInt::get(llvm::Type::getInt8Ty(gIR->context()), 0),
      llvm::GlobalValue::LinkOnceODRLinkage, false);
  dsoInitialized->setVisibility(llvm::GlobalValue::HiddenVisibility);

  // There is no reason for this cast to void*, other than that removing it
  // seems to trigger a bug in the llvm::Linker (at least on LLVM 3.4)
  // causing it to not merge the %object.ModuleInfo types properly. This
  // manifests itself in a type mismatch assertion being triggered on the
  // minfoUsedPointer store in the ctor as soon as the optimizer runs.
  llvm::Value *minfoRefPtr = DtoBitCast(thismref, getVoidPtrType());

  const auto registerDSO =
      buildRegisterDSO(style, dsoInitialized, dsoSlot, minfoBeg, minfoEnd);

  std::string ctorName = "ldc.dso_ctor.";
  ctorName += moduleMangle;
  const auto dsoCtor = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(gIR->context()), false),
      llvm::GlobalValue::LinkOnceODRLinkage, ctorName, &gIR->module);
  dsoCtor->setVisibility(llvm::GlobalValue::HiddenVisibility);
  {
    const auto bb = llvm::BasicBlock::Create(gIR->context(), "", dsoCtor);
    IRBuilder<> b{bb};
    LLValue *params[] = {b.getFalse(), minfoRefPtr};
    b.CreateCall(registerDSO, params);
    b.CreateRetVoid();
  }
  llvm::appendToGlobalCtors(gIR->module, dsoCtor, 65535);

  std::string dtorName = "ldc.dso_dtor.";
  dtorName += moduleMangle;
  const auto dsoDtor = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(gIR->context()), false),
      llvm::GlobalValue::LinkOnceODRLinkage, dtorName, &gIR->module);
  dsoDtor->setVisibility(llvm::GlobalValue::HiddenVisibility);
  {
    const auto bb = llvm::BasicBlock::Create(gIR->context(), "", dsoDtor);
    IRBuilder<> b{bb};
    LLValue *params[] = {b.getTrue(), minfoRefPtr};
    b.CreateCall(registerDSO, params);
    b.CreateRetVoid();
  }
  llvm::appendToGlobalDtors(gIR->module, dsoDtor, 65535);
}

// Add module-private variables and functions for coverage analysis.
void addCoverageAnalysis(Module *m) {
  IF_LOG {
    Logger::println("Adding coverage analysis for module %s (%d lines)",
                    m->srcfile->toChars(), m->numlines);
    Logger::indent();
  }

  // size_t[# source lines / # bits in sizeTy] _d_cover_valid
  LLValue *d_cover_valid_slice = nullptr;
  {
    unsigned Dsizet_bits = gDataLayout->getTypeSizeInBits(DtoSize_t());
    size_t array_size = (m->numlines + (Dsizet_bits - 1)) / Dsizet_bits; // ceil

    // Work around a bug in the interface of druntime's _d_cover_register2
    // https://issues.dlang.org/show_bug.cgi?id=14417
    // For safety, make the array large enough such that the slice passed to
    // _d_cover_register2 is completely valid.
    array_size = m->numlines;

    IF_LOG Logger::println(
        "Build private variable: size_t[%llu] _d_cover_valid",
        static_cast<unsigned long long>(array_size));

    llvm::ArrayType *type = llvm::ArrayType::get(DtoSize_t(), array_size);
    llvm::ConstantAggregateZero *zeroinitializer =
        llvm::ConstantAggregateZero::get(type);
    m->d_cover_valid = new llvm::GlobalVariable(
        gIR->module, type, /*isConstant=*/true, LLGlobalValue::InternalLinkage,
        zeroinitializer, "_d_cover_valid");
    LLConstant *idxs[] = {DtoConstUint(0), DtoConstUint(0)};
    d_cover_valid_slice =
        DtoConstSlice(DtoConstSize_t(type->getArrayNumElements()),
                      llvm::ConstantExpr::getGetElementPtr(
                          type, m->d_cover_valid, idxs, true));

    // Assert that initializer array elements have enough bits
    assert(sizeof(m->d_cover_valid_init[0]) * 8 >=
           gDataLayout->getTypeSizeInBits(DtoSize_t()));
    m->d_cover_valid_init.setDim(array_size);
    m->d_cover_valid_init.zero();
  }

  // uint[# source lines] _d_cover_data
  LLValue *d_cover_data_slice = nullptr;
  {
    IF_LOG Logger::println("Build private variable: uint[%d] _d_cover_data",
                           m->numlines);

    LLArrayType *type =
        LLArrayType::get(LLType::getInt32Ty(gIR->context()), m->numlines);
    llvm::ConstantAggregateZero *zeroinitializer =
        llvm::ConstantAggregateZero::get(type);
    m->d_cover_data = new llvm::GlobalVariable(
        gIR->module, type, false, LLGlobalValue::InternalLinkage,
        zeroinitializer, "_d_cover_data");
    LLConstant *idxs[] = {DtoConstUint(0), DtoConstUint(0)};
    d_cover_data_slice =
        DtoConstSlice(DtoConstSize_t(type->getArrayNumElements()),
                      llvm::ConstantExpr::getGetElementPtr(
                          type, m->d_cover_data, idxs, true));
  }

  // Create "static constructor" that calls _d_cover_register2(string filename,
  // size_t[] valid, uint[] data, ubyte minPercent)
  // Build ctor name
  LLFunction *ctor = nullptr;

  OutBuffer mangleBuf;
  mangleBuf.writestring("_D");
  mangleToBuffer(m, &mangleBuf);
  mangleBuf.writestring("12_coverageanalysisCtor1FZv");
  const char *ctorname = mangleBuf.peekString();

  {
    IF_LOG Logger::println("Build Coverage Analysis constructor: %s", ctorname);

    LLFunctionType *ctorTy =
        LLFunctionType::get(LLType::getVoidTy(gIR->context()), {}, false);
    ctor =
        LLFunction::Create(ctorTy, LLGlobalValue::InternalLinkage,
                           getIRMangledFuncName(ctorname, LINKd), &gIR->module);
    ctor->setCallingConv(gABI->callingConv(LINKd));
    // Set function attributes. See functions.cpp:DtoDefineFunction()
    if (global.params.targetTriple->getArch() == llvm::Triple::x86_64) {
      ctor->addFnAttr(LLAttribute::UWTable);
    }

    llvm::BasicBlock *bb = llvm::BasicBlock::Create(gIR->context(), "", ctor);
    IRBuilder<> builder(bb);

    // Set up call to _d_cover_register2
    llvm::Function *fn =
        getRuntimeFunction(Loc(), gIR->module, "_d_cover_register2");
    LLValue *args[] = {DtoConstString(m->srcfile->name->toChars()),
                       d_cover_valid_slice, d_cover_data_slice,
                       DtoConstUbyte(global.params.covPercent)};
    // Check if argument types are correct
    for (unsigned i = 0; i < 4; ++i) {
      assert(args[i]->getType() == fn->getFunctionType()->getParamType(i));
    }

    builder.CreateCall(fn, args);

    builder.CreateRetVoid();
  }

  // Add the ctor to the module's order-independent ctors list.
  {
    IF_LOG Logger::println("Set %s as module's static constructor for coverage",
                           ctorname);
    getIrModule(m)->coverageCtor = ctor;
  }

  IF_LOG Logger::undent();
}

// Initialize _d_cover_valid for coverage analysis
void addCoverageAnalysisInitializer(Module *m) {
  IF_LOG Logger::println("Adding coverage analysis _d_cover_valid initializer");

  size_t array_size = m->d_cover_valid_init.size();

  llvm::ArrayType *type = llvm::ArrayType::get(DtoSize_t(), array_size);
  std::vector<LLConstant *> arrayInits(array_size);
  for (size_t i = 0; i < array_size; i++) {
    arrayInits[i] = DtoConstSize_t(m->d_cover_valid_init[i]);
  }
  m->d_cover_valid->setInitializer(llvm::ConstantArray::get(type, arrayInits));
}

// Load InstrProf data from file and store in it IrState
// TODO: This is probably not the right place, we should load it once for all
// modules?
void loadInstrProfileData(IRState *irs) {
  // Only load from datafileInstrProf if we are doing frontend-based PGO.
  if (opts::isUsingASTBasedPGOProfile() && global.params.datafileInstrProf) {
    IF_LOG Logger::println("Read profile data from %s",
                           global.params.datafileInstrProf);

    auto readerOrErr =
        llvm::IndexedInstrProfReader::create(global.params.datafileInstrProf);
#if LDC_LLVM_VER >= 309
    if (auto E = readerOrErr.takeError()) {
      handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EI) {
        irs->dmodule->error("Could not read profile file '%s': %s",
                            global.params.datafileInstrProf,
                            EI.message().c_str());
      });
      fatal();
    }
#else
    std::error_code EC = readerOrErr.getError();
    if (EC) {
      irs->dmodule->error("Could not read profile file '%s': %s",
                          global.params.datafileInstrProf,
                          EC.message().c_str());
      fatal();
    }
#endif
    irs->PGOReader = std::move(readerOrErr.get());

#if LDC_LLVM_VER >= 309
    if (!irs->module.getProfileSummary()) {
      // Don't reset the summary. There is only one profile data file per LDC
      // invocation so the summary must be the same as the one that is already
      // set.
      irs->module.setProfileSummary(
          irs->PGOReader->getSummary().getMD(irs->context()));
    }
#elif LDC_LLVM_VER == 308
    auto maxCount = irs->PGOReader->getMaximumFunctionCount();
    if (!irs->module.getMaximumFunctionCount()) {
      // Don't reset the max function count. There is only one profile data file
      // per LDC invocation so the information must be the same as the one that
      // is already set.
      irs->module.setMaximumFunctionCount(maxCount);
    }
#endif
  }
}

void registerModuleInfo(Module *m) {
  const auto moduleInfoSym = genModuleInfo(m);
  const auto style = getModuleRegistryStyle();

  OutBuffer mangleBuf;
  mangleToBuffer(m, &mangleBuf);
  const char *mangle = mangleBuf.peekString();

  if (style == RegistryStyle::legacyLinkedList) {
    const auto miCtor = build_module_reference_and_ctor(mangle, moduleInfoSym);
    AppendFunctionToLLVMGlobalCtorsDtors(miCtor, 65535, true);
  } else {
    emitModuleRefToSection(style, mangle, moduleInfoSym);
  }
}
}

void codegenModule(IRState *irs, Module *m) {
  assert(!irs->dmodule &&
         "irs->module not null, codegen already in progress?!");
  irs->dmodule = m;
  assert(!gIR && "gIR not null, codegen already in progress?!");
  gIR = irs;

  initRuntime();

  // Skip pseudo-modules for coverage analysis
  std::string name = m->toChars();
  const bool isPseudoModule = (name == "__entrypoint") || (name == "__main");
  if (global.params.cov && !isPseudoModule) {
    addCoverageAnalysis(m);
  }

  if (!isPseudoModule) {
    loadInstrProfileData(gIR);
  }

  // process module members
  // NOTE: m->members may grow during codegen
  for (unsigned k = 0; k < m->members->dim; k++) {
    Dsymbol *dsym = (*m->members)[k];
    assert(dsym);
    Declaration_codegen(dsym);
  }

  if (global.errors) {
    fatal();
  }

  // Skip emission of all the additional module metadata if:
  // a) the -betterC switch is on,
  // b) requested explicitly by the user via pragma(LDC_no_moduleinfo), or if
  // c) there's no ModuleInfo declaration.
  if (global.params.useModuleInfo && !m->noModuleInfo && Module::moduleinfo) {
    // generate ModuleInfo
    registerModuleInfo(m);
  }

  if (m->d_cover_valid) {
    addCoverageAnalysisInitializer(m);
  }

  gIR = nullptr;
  irs->dmodule = nullptr;
}
