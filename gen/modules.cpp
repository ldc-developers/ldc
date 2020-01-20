//===-- modules.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/aggregate.h"
#include "dmd/attrib.h"
#include "dmd/declaration.h"
#include "dmd/enum.h"
#include "dmd/errors.h"
#include "dmd/id.h"
#include "dmd/import.h"
#include "dmd/init.h"
#include "dmd/mangle.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "dmd/scope.h"
#include "dmd/statement.h"
#include "dmd/target.h"
#include "dmd/template.h"
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
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
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

void Module::checkAndAddOutputFile(const FileName &file) {
  static std::map<std::string, Module *> files;

  std::string key(file.toChars());
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
  assert(objfile.toChars());

  const char *ext = FileName::ext(objfile.toChars());
  const char *stem = FileName::removeExt(objfile.toChars());

  llvm::SmallString<128> unique;
  auto EC = llvm::sys::fs::createUniqueFile(
      llvm::Twine(stem) + "-%%%%%%%." + ext, unique);
  if (!EC) // success
    objfile.reset(unique.c_str());
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

LLGlobalVariable *declareDSOGlobal(llvm::StringRef mangledName, LLType *type,
                                   bool isThreadLocal = false) {
  auto global = declareGlobal(Loc(), gIR->module, type, mangledName, false,
                              isThreadLocal);
  global->setVisibility(LLGlobalValue::HiddenVisibility);
  return global;
}

LLGlobalVariable *defineDSOGlobal(llvm::StringRef mangledName, LLConstant *init,
                                  bool isThreadLocal = false) {
  auto global =
      defineGlobal(Loc(), gIR->module, mangledName, init,
                   LLGlobalValue::LinkOnceODRLinkage, false, isThreadLocal);
  global->setVisibility(LLGlobalValue::HiddenVisibility);
  return global;
}

LLFunction *createDSOFunction(llvm::StringRef mangledName,
                              LLFunctionType *type) {
  auto fn = LLFunction::Create(type, LLGlobalValue::LinkOnceODRLinkage,
                               mangledName, &gIR->module);
  fn->setVisibility(LLGlobalValue::HiddenVisibility);
  return fn;
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
  const auto one = llvm::ConstantInt::get(LLType::getInt8Ty(gIR->context()), 1);
  const auto anchor =
      defineDSOGlobal("ldc.tls_anchor", one, /*isThreadLocal=*/true);
  anchor->setAlignment(LLMaybeAlign(16));

  const auto getAnchor = createDSOFunction(
      "ldc.get_tls_anchor", LLFunctionType::get(getVoidPtrType(), false));

  IRBuilder<> builder(llvm::BasicBlock::Create(gIR->context(), "", getAnchor));
  builder.CreateRet(anchor);

  return getAnchor;
}

/// Builds the ldc.register_dso function, which is called by the global
/// {c, d}tors to invoke _d_dso_registry.
///
/// Pseudocode:
/// void ldc.register_dso() {
///   auto record = {1, dsoSlot, minfoBeg, minfoEnd[, getTlsAnchor]};
///   _d_dso_registry(cast(CompilerDSOData*)&record);
/// }
///
/// On Darwin platforms, the record contains an extra pointer to a function
/// which returns the address of a TLS global.
llvm::Function *buildRegisterDSO(RegistryStyle style, llvm::Value *dsoSlot,
                                 llvm::Value *minfoBeg, llvm::Value *minfoEnd) {
  const auto fn = createDSOFunction(
      "ldc.register_dso",
      LLFunctionType::get(LLType::getVoidTy(gIR->context()), false));

  const auto dsoRegistry =
      getRuntimeFunction(Loc(), gIR->module, "_d_dso_registry");
  const auto recordPtrTy = dsoRegistry->getFunctionType()->getContainedType(1);

  llvm::Function *getTlsAnchorPtr = nullptr;
  if (style == RegistryStyle::sectionDarwin) {
    getTlsAnchorPtr = buildGetTLSAnchor();
  }

  {
    const auto bb = llvm::BasicBlock::Create(gIR->context(), "", fn);
    IRBuilder<> b(bb);

    llvm::Constant *version = DtoConstSize_t(1);
    llvm::SmallVector<llvm::Type *, 6> memberTypes;
    memberTypes.push_back(version->getType());
    memberTypes.push_back(dsoSlot->getType());
    memberTypes.push_back(minfoBeg->getType());
    memberTypes.push_back(minfoEnd->getType());
    if (style == RegistryStyle::sectionDarwin) {
      memberTypes.push_back(getTlsAnchorPtr->getType());
    }
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

    b.CreateCall(dsoRegistry, b.CreateBitCast(record, recordPtrTy));
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
  // the .minfo section.
  const bool isFirst = !gIR->module.getGlobalVariable("ldc.dso_slot");

  const auto moduleInfoPtrTy = DtoPtrToType(getModuleInfoType());
  const auto moduleInfoRefsSectionName =
      style == RegistryStyle::sectionMSVC
          ? ".minfo"
          : style == RegistryStyle::sectionDarwin ? "__DATA,.minfo" : "__minfo";

  const auto thismrefIRMangle =
      getIRMangledModuleRefSymbolName(moduleMangle.c_str());
  auto thismref = defineDSOGlobal(thismrefIRMangle,
                                  DtoBitCast(thisModuleInfo, moduleInfoPtrTy));
  thismref->setSection(moduleInfoRefsSectionName);
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
  auto minfoBeg = declareDSOGlobal(magicBeginSymbolName, moduleInfoPtrTy);
  auto minfoEnd = declareDSOGlobal(magicEndSymbolName, moduleInfoPtrTy);

  // We want to have one global constructor and destructor per object (i.e.
  // executable/shared library) that calls _d_dso_registry with the respective
  // DSO record.
  // To enable safe direct linking of D objects (e.g., "g++ dcode.o cppcode.o"),
  // we emit a pair of global {c,d}tors into each object file, both pointing to
  // a common ldc.register_dso() function.
  // These per-object-file pairs will be folded to a single one when linking the
  // DSO, together with the ldc.dso_slot globals and associated
  // ldc.register_dso() functions.

  // This is the DSO slot for use by the druntime implementation.
  const auto dsoSlot =
      defineDSOGlobal("ldc.dso_slot", getNullPtr(getVoidPtrType()));

  const auto registerDSO = buildRegisterDSO(style, dsoSlot, minfoBeg, minfoEnd);

  // We need to discard the {c,d}tor refs if this IR module's ldc.register_dso()
  // function is discarded to prevent duplicate refs.
  // Unfortunately, this doesn't work for macOS (v10.12, Xcode v9.2, LLVM
  // v7.0.0).
  if (style == RegistryStyle::sectionELF) {
    llvm::appendToGlobalCtors(gIR->module, registerDSO, 65535, registerDSO);
    llvm::appendToGlobalDtors(gIR->module, registerDSO, 65535, registerDSO);
    return;
  }

  // macOS: emit the {c,d}tor refs manually
  const auto dsoCtor = defineDSOGlobal("ldc.dso_ctor", registerDSO);
  const auto dsoDtor = defineDSOGlobal("ldc.dso_dtor", registerDSO);
  gIR->usedArray.push_back(dsoCtor);
  gIR->usedArray.push_back(dsoDtor);
  dsoCtor->setSection("__DATA,__mod_init_func,mod_init_funcs");
  dsoDtor->setSection("__DATA,__mod_term_func,mod_term_funcs");
}

// Add module-private variables and functions for coverage analysis.
void addCoverageAnalysis(Module *m) {
  IF_LOG {
    Logger::println("Adding coverage analysis for module %s (%d lines)",
                    m->srcfile.toChars(), m->numlines);
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
  const char *ctorname = mangleBuf.peekChars();

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
    LLValue *args[] = {DtoConstString(m->srcfile.toChars()),
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
    if (auto E = readerOrErr.takeError()) {
      handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EI) {
        irs->dmodule->error("Could not read profile file '%s': %s",
                            global.params.datafileInstrProf,
                            EI.message().c_str());
      });
      fatal();
    }
    irs->PGOReader = std::move(readerOrErr.get());

    if (!irs->module.getProfileSummary(

#if LDC_LLVM_VER >= 900
            /*is context sensitive profile=*/false
#endif
            )) {

      // Don't reset the summary. There is only one profile data file per LDC
      // invocation so the summary must be the same as the one that is already
      // set.
      irs->module.setProfileSummary(

#if LDC_LLVM_VER >= 900
          irs->PGOReader->getSummary(/*is context sensitive profile=*/false)
              .getMD(irs->context()),
          llvm::ProfileSummary::PSK_Instr
#else
          irs->PGOReader->getSummary().getMD(irs->context())
#endif
      );
    }
  }
}

void registerModuleInfo(Module *m) {
  const auto moduleInfoSym = genModuleInfo(m);
  const auto style = getModuleRegistryStyle();

  OutBuffer mangleBuf;
  mangleToBuffer(m, &mangleBuf);
  const char *mangle = mangleBuf.peekChars();

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

  irs->DBuilder.EmitModule(m);

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
  for (unsigned k = 0; k < m->members->length; k++) {
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
