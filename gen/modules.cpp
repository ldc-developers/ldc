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
#include "driver/timetrace.h"
#include "gen/abi/abi.h"
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
#if LDC_LLVM_VER >= 1700
#include "llvm/Support/VirtualFileSystem.h"
#endif

#if _AIX || __sun
#include <alloca.h>
#endif

using namespace dmd;

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

namespace {
/// Ways the druntime ModuleInfo registry system can be implemented.
enum class RegistryStyle {
  /// ModuleInfo refs are inserted into a linked list starting at the
  /// _Dmodule_ref global.
  legacyLinkedList,

  /// Pointers to defined ModuleInfos are emitted into the special .minfo /
  /// __minfo section. A linked binary then contains pointers to all ModuleInfos
  /// of linked object files in that section.
  section,
};

/// Returns the ModuleInfo registry style to use for the current target triple.
RegistryStyle getModuleRegistryStyle() {
  const auto &t = *global.params.targetTriple;
  if (t.isOSWindows() || t.getEnvironment() == llvm::Triple::Android ||
      t.isOSBinFormatWasm() || t.isOSDarwin() || t.isOSLinux() ||
      t.isOSFreeBSD() || t.isOSNetBSD() || t.isOSOpenBSD() ||
      t.isOSDragonFly()) {
    return RegistryStyle::section;
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
                         getIRMangledFuncName(fname, LINK::d), &gIR->module);

  // provide the default initializer
  LLStructType *modulerefTy = DtoModuleReferenceType();
  LLConstant *mrefvalues[] = {LLConstant::getNullValue(getVoidPtrType()),
                              moduleinfo};
  LLConstant *thismrefinit = LLConstantStruct::get(
      modulerefTy, llvm::ArrayRef<LLConstant *>(mrefvalues));

  // create the ModuleReference node for this module
  const auto thismrefIRMangle = getIRMangledModuleRefSymbolName(moduleMangle);
  LLGlobalVariable *thismref =
      defineGlobal(Loc(), gIR->module, thismrefIRMangle, thismrefinit,
                   LLGlobalValue::InternalLinkage, false);
  // make sure _Dmodule_ref is declared
  const auto mrefIRMangle = getIRMangledVarName("_Dmodule_ref", LINK::c);
  LLConstant *mref = gIR->module.getNamedGlobal(mrefIRMangle);
  LLType *modulerefPtrTy = getVoidPtrType();
  if (!mref) {
    mref =
        declareGlobal(Loc(), gIR->module, modulerefPtrTy, mrefIRMangle, false,
                      false, global.params.dllimport != DLLImport::none);
  }

  // make the function insert this moduleinfo as the beginning of the
  // _Dmodule_ref linked list
  llvm::BasicBlock *bb =
      llvm::BasicBlock::Create(gIR->context(), "moduleinfoCtorEntry", ctor);
  IRBuilder<> builder(bb);

  // debug info
  gIR->DBuilder.EmitModuleCTor(ctor, fname.c_str());

  // get current beginning
  LLValue *curbeg = builder.CreateLoad(modulerefPtrTy, mref, "current");

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

// Emits a pointer to the specified ModuleInfo into the special
// .minfo (COFF & MachO) / __minfo section.
void emitModuleRefToSection(std::string moduleMangle,
                            llvm::Constant *thisModuleInfo) {
  const auto &triple = *global.params.targetTriple;
  const auto sectionName =
      triple.isOSBinFormatCOFF()
          ? ".minfo"
          : triple.isOSBinFormatMachO() ? "__DATA,.minfo" : "__minfo";

  const auto thismrefIRMangle =
      getIRMangledModuleRefSymbolName(moduleMangle.c_str());
  auto thismref =
      defineGlobal(Loc(), gIR->module, thismrefIRMangle, thisModuleInfo,
                   LLGlobalValue::LinkOnceODRLinkage, false, false);
  thismref->setVisibility(LLGlobalValue::HiddenVisibility);
  thismref->setSection(sectionName);
  gIR->usedArray.push_back(thismref);
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
    d_cover_valid_slice = DtoConstSlice(
        DtoConstSize_t(type->getArrayNumElements()), m->d_cover_valid);

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

    llvm::Constant *init;
    if (!m->ctfe_cov) {
      init = llvm::ConstantAggregateZero::get(type);
    } else {
      std::vector<unsigned> initData(m->numlines);
      m->initCoverageDataWithCtfeCoverage(initData.data());
      init = llvm::ConstantDataArray::get(gIR->context(), initData);
    }

    m->d_cover_data = new llvm::GlobalVariable(gIR->module, type, false,
                                               LLGlobalValue::InternalLinkage,
                                               init, "_d_cover_data");

    d_cover_data_slice =
        DtoConstSlice(DtoConstSize_t(m->numlines), m->d_cover_data);
  }

  // Create "static constructor" that calls _d_cover_register2(string filename,
  // size_t[] valid, uint[] data, ubyte minPercent)
  // Build ctor name
  LLFunction *ctor = nullptr;

  OutBuffer mangleBuf;
  mangleBuf.writestring("_D");
  mangleToBuffer(m, mangleBuf);
  mangleBuf.writestring("12_coverageanalysisCtor1FZv");
  const char *ctorname = mangleBuf.peekChars();

  {
    IF_LOG Logger::println("Build Coverage Analysis constructor: %s", ctorname);

    LLFunctionType *ctorTy =
        LLFunctionType::get(LLType::getVoidTy(gIR->context()), {}, false);
    ctor =
        LLFunction::Create(ctorTy, LLGlobalValue::InternalLinkage,
                           getIRMangledFuncName(ctorname, LINK::d), &gIR->module);
    ctor->setCallingConv(gABI->callingConv(LINK::d));
    // Set function attributes. See functions.cpp:DtoDefineFunction()
    if (global.params.targetTriple->getArch() == llvm::Triple::x86_64) {
      ctor->setUWTableKind(llvm::UWTableKind::Default);
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
        llvm::IndexedInstrProfReader::create(global.params.datafileInstrProf
#if LDC_LLVM_VER >= 1700
                                             ,
                                             *llvm::vfs::getRealFileSystem()
#endif
        );
    if (auto E = readerOrErr.takeError()) {
      handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EI) {
        error(irs->dmodule->loc, "Could not read profile file '%s': %s",
              global.params.datafileInstrProf, EI.message().c_str());
      });
      fatal();
    }
    irs->PGOReader = std::move(readerOrErr.get());

    if (!irs->module.getProfileSummary(
            /*is context sensitive profile=*/false)) {
      // Don't reset the summary. There is only one profile data file per LDC
      // invocation so the summary must be the same as the one that is already
      // set.
      irs->module.setProfileSummary(
          irs->PGOReader->getSummary(/*is context sensitive profile=*/false)
              .getMD(irs->context()),
          llvm::ProfileSummary::PSK_Instr);
    }
  }
}

void registerModuleInfo(Module *m) {
  const auto moduleInfoSym = genModuleInfo(m);
  const auto style = getModuleRegistryStyle();

  OutBuffer mangleBuf;
  mangleToBuffer(m, mangleBuf);
  const char *mangle = mangleBuf.peekChars();

  if (style == RegistryStyle::legacyLinkedList) {
    const auto miCtor = build_module_reference_and_ctor(mangle, moduleInfoSym);
    AppendFunctionToLLVMGlobalCtorsDtors(miCtor, 65535, true);
  } else {
    emitModuleRefToSection(mangle, moduleInfoSym);
  }
}

void addModuleFlags(llvm::Module &m) {
  const auto ModuleMinFlag = llvm::Module::Min;

  if (opts::fCFProtection == opts::CFProtectionType::Return ||
      opts::fCFProtection == opts::CFProtectionType::Full) {
    m.addModuleFlag(ModuleMinFlag, "cf-protection-return", 1);
  }

  if (opts::fCFProtection == opts::CFProtectionType::Branch ||
      opts::fCFProtection == opts::CFProtectionType::Full) {
    m.addModuleFlag(ModuleMinFlag, "cf-protection-branch", 1);
  }
}

} // anonymous namespace

void codegenModule(IRState *irs, Module *m) {
  TimeTraceScope timeScope("Generate IR", m->toChars(), m->loc);

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
  for (d_size_t k = 0; k < m->members->length; k++) {
    Dsymbol *dsym = (*m->members)[k];
    assert(dsym);
    Declaration_codegen(dsym);
  }

  if (global.errors) {
    fatal();
  }

  // Skip emission of the ModuleInfo if:
  // a) the -betterC or -fno-moduleinfo switch is on,
  // b) requested explicitly by the user via pragma(LDC_no_moduleinfo),
  // c) there's no ModuleInfo declaration, or if
  // d) the module is a C file.
  if (global.params.useModuleInfo && !m->noModuleInfo && Module::moduleinfo &&
      m->filetype != FileType::c) {
    // generate ModuleInfo
    registerModuleInfo(m);
  }

  if (m->d_cover_valid) {
    addCoverageAnalysisInitializer(m);
  }

  addModuleFlags(irs->module);

  gIR = nullptr;
  irs->dmodule = nullptr;
}
