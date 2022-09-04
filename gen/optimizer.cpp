//===-- optimizer.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/optimizer.h"

#include "dmd/errors.h"
#include "gen/cl_helpers.h"
#include "gen/logger.h"
#include "gen/passes/Passes.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/cl_options_sanitizers.h"
#include "driver/targetmachine.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Instrumentation/MemorySanitizer.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
#if LDC_LLVM_VER >= 1000
#include "llvm/Transforms/Instrumentation/SanitizerCoverage.h"
#endif

extern llvm::TargetMachine *gTargetMachine;
using namespace llvm;

static cl::opt<signed char> optimizeLevel(
    cl::desc("Setting the optimization level:"), cl::ZeroOrMore,
    cl::values(
        clEnumValN(3, "O", "Equivalent to -O3"),
        clEnumValN(0, "O0", "No optimizations (default)"),
        clEnumValN(1, "O1", "Simple optimizations"),
        clEnumValN(2, "O2", "Good optimizations"),
        clEnumValN(3, "O3", "Aggressive optimizations"),
        clEnumValN(4, "O4", "Equivalent to -O3"), // Not implemented yet.
        clEnumValN(5, "O5", "Equivalent to -O3"), // Not implemented yet.
        clEnumValN(-1, "Os", "Like -O2 with extra optimizations for size"),
        clEnumValN(-2, "Oz", "Like -Os but reduces code size further")),
    cl::init(0));

static cl::opt<bool> noVerify("disable-verify", cl::ZeroOrMore, cl::Hidden,
                              cl::desc("Do not verify result module"));

static cl::opt<bool>
    verifyEach("verify-each", cl::ZeroOrMore, cl::Hidden,
               cl::desc("Run verifier after D-specific and explicitly "
                        "specified optimization passes"));

static cl::opt<bool>
    disableLangSpecificPasses("disable-d-passes", cl::ZeroOrMore,
                              cl::desc("Disable all D-specific passes"));

static cl::opt<bool> disableSimplifyDruntimeCalls(
    "disable-simplify-drtcalls", cl::ZeroOrMore,
    cl::desc("Disable simplification of druntime calls"));

static cl::opt<bool> disableSimplifyLibCalls(
    "disable-simplify-libcalls", cl::ZeroOrMore,
    cl::desc("Disable simplification of well-known C runtime calls"));

static cl::opt<bool> disableGCToStack(
    "disable-gc2stack", cl::ZeroOrMore,
    cl::desc("Disable promotion of GC allocations to stack memory"));

static cl::opt<cl::boolOrDefault, false, opts::FlagParser<cl::boolOrDefault>>
    enableInlining(
        "inlining", cl::ZeroOrMore,
        cl::desc("(*) Enable function inlining (default in -O2 and higher)"));

static cl::opt<cl::boolOrDefault, false, opts::FlagParser<cl::boolOrDefault>>
    enableCrossModuleInlining(
        "cross-module-inlining", cl::ZeroOrMore, cl::Hidden,
        cl::desc("(*) Enable cross-module function inlining (default disabled)"));

static cl::opt<bool> stripDebug(
    "strip-debug", cl::ZeroOrMore,
    cl::desc("Strip symbolic debug information before optimization"));

static cl::opt<bool> disableLoopUnrolling(
    "disable-loop-unrolling", cl::ZeroOrMore,
    cl::desc("Disable loop unrolling in all relevant passes"));
static cl::opt<bool>
    disableLoopVectorization("disable-loop-vectorization", cl::ZeroOrMore,
                             cl::desc("Disable the loop vectorization pass"));

static cl::opt<bool>
    disableSLPVectorization("disable-slp-vectorization", cl::ZeroOrMore,
                            cl::desc("Disable the slp vectorization pass"));

static cl::opt<int> fSanitizeMemoryTrackOrigins(
    "fsanitize-memory-track-origins", cl::ZeroOrMore, cl::init(0),
    cl::desc(
        "Enable origins tracking in MemorySanitizer (0=disabled, default)"));

static cl::opt<signed char> passmanager("passmanger",
    cl::desc("Setting the passmanger (new,legacy):"), cl::ZeroOrMore, cl::init(0),
    cl::values(
        clEnumValN(0, "legacy", "Use the legacy passmanager (available for LLVM14 and below) "),
        clEnumValN(1, "new", "Use the new passmanager (available for LLVM14 and above)")));

unsigned optLevel() {
  // Use -O2 as a base for the size-optimization levels.
  return optimizeLevel >= 0 ? optimizeLevel : 2;
}

static unsigned sizeLevel() { return optimizeLevel < 0 ? -optimizeLevel : 0; }

// Determines whether or not to run the normal, full inlining pass.
bool willInline() {
  return enableInlining == cl::BOU_TRUE ||
         (enableInlining == cl::BOU_UNSET && optLevel() > 1);
}

bool willCrossModuleInline() {
  return enableCrossModuleInlining == llvm::cl::BOU_TRUE && willInline();
}

#if LDC_LLVM_VER < 1000
llvm::FramePointer::FP whichFramePointersToEmit() {
  if (auto option = opts::framePointerUsage())
    return *option;
  return isOptimizationEnabled() ? llvm::FramePointer::None
                                 : llvm::FramePointer::All;
}
#endif

bool isOptimizationEnabled() { return optimizeLevel != 0; }

llvm::CodeGenOpt::Level codeGenOptLevel() {
  // Use same appoach as clang (see lib/CodeGen/BackendUtil.cpp)
  if (optLevel() == 0) {
    return llvm::CodeGenOpt::None;
  }
  if (optLevel() >= 3) {
    return llvm::CodeGenOpt::Aggressive;
  }
  return llvm::CodeGenOpt::Default;
}

#if LDC_LLVM_VER < 1500
static inline void legacyAddPass(PassManagerBase &pm, Pass *pass) {
  pm.add(pass);

  if (verifyEach) {
    pm.add(createVerifierPass());
  }
}

static void legacyAddStripExternalsPass(const PassManagerBuilder &builder,
                                  PassManagerBase &pm) {
  if (builder.OptLevel >= 1) {
    legacyAddPass(pm, createStripExternalsPass());
    legacyAddPass(pm, createGlobalDCEPass());
  }
}

static void legacyAddSimplifyDRuntimeCallsPass(const PassManagerBuilder &builder,
                                         PassManagerBase &pm) {
  if (builder.OptLevel >= 2 && builder.SizeLevel == 0) {
    legacyAddPass(pm, createSimplifyDRuntimeCalls());
  }
}

static void legacyAddGarbageCollect2StackPass(const PassManagerBuilder &builder,
                                        PassManagerBase &pm) {
  if (builder.OptLevel >= 2 && builder.SizeLevel == 0) {
    legacyAddPass(pm, createGarbageCollect2Stack());
  }
}

static void legacyAddAddressSanitizerPasses(const PassManagerBuilder &Builder,
                                      PassManagerBase &PM) {
  PM.add(createAddressSanitizerFunctionPass());
  PM.add(createModuleAddressSanitizerLegacyPassPass());
}

static void legacyAddMemorySanitizerPass(const PassManagerBuilder &Builder,
                                   PassManagerBase &PM) {
  int trackOrigins = fSanitizeMemoryTrackOrigins;
  bool recover = false;
  bool kernel = false;
  PM.add(createMemorySanitizerLegacyPassPass(
      MemorySanitizerOptions{trackOrigins, recover, kernel}));

  // MemorySanitizer inserts complex instrumentation that mostly follows
  // the logic of the original code, but operates on "shadow" values.
  // It can benefit from re-running some general purpose optimization passes.
  if (Builder.OptLevel > 0) {
    PM.add(createEarlyCSEPass());
    PM.add(createReassociatePass());
    PM.add(createLICMPass());
    PM.add(createGVNPass());
    PM.add(createInstructionCombiningPass());
    PM.add(createDeadStoreEliminationPass());
  }
}

static void legacyAddThreadSanitizerPass(const PassManagerBuilder &Builder,
                                   PassManagerBase &PM) {
  PM.add(createThreadSanitizerLegacyPassPass());
}

static void legacyAddSanitizerCoveragePass(const PassManagerBuilder &Builder,
                                     legacy::PassManagerBase &PM) {
#if LDC_LLVM_VER >= 1000
  PM.add(createModuleSanitizerCoverageLegacyPassPass(
      opts::getSanitizerCoverageOptions()));
#else
  PM.add(
      createSanitizerCoverageModulePass(opts::getSanitizerCoverageOptions()));
#endif
}

// Adds PGO instrumentation generation and use passes.
static void legacyAddPGOPasses(PassManagerBuilder &builder,
                         legacy::PassManagerBase &mpm, unsigned optLevel) {
  if (opts::isInstrumentingForASTBasedPGO()) {
    InstrProfOptions options;
    options.NoRedZone = global.params.disableRedZone;
    if (global.params.datafileInstrProf)
      options.InstrProfileOutput = global.params.datafileInstrProf;
    mpm.add(createInstrProfilingLegacyPass(options));
  } else if (opts::isUsingASTBasedPGOProfile()) {
    // We are generating code with PGO profile information available.
    // Do indirect call promotion from -O1
    if (optLevel > 0) {
      mpm.add(createPGOIndirectCallPromotionLegacyPass());
    }
  } else if (opts::isInstrumentingForIRBasedPGO()) {
    builder.EnablePGOInstrGen = true;
    builder.PGOInstrGen = global.params.datafileInstrProf;
  } else if (opts::isUsingIRBasedPGOProfile()) {
    builder.PGOInstrUse = global.params.datafileInstrProf;
  }
}

/**
 * Adds a set of optimization passes to the given module/function pass
 * managers based on the given optimization and size reduction levels.
 *
 * The selection mirrors Clang behavior and is based on LLVM's
 * PassManagerBuilder.
 */
static void legacyAddOptimizationPasses(legacy::PassManagerBase &mpm,
                                  legacy::FunctionPassManager &fpm,
                                  unsigned optLevel, unsigned sizeLevel) {
  if (!noVerify) {
    fpm.add(createVerifierPass());
  }
  PassManagerBuilder builder;
  builder.OptLevel = optLevel;
  builder.SizeLevel = sizeLevel;
  builder.PrepareForLTO = opts::isUsingLTO();
  builder.PrepareForThinLTO = opts::isUsingThinLTO();

  if (willInline()) {
    auto params = llvm::getInlineParams(optLevel, sizeLevel);
    builder.Inliner = createFunctionInliningPass(params);
  } else {
    builder.Inliner = createAlwaysInlinerLegacyPass();
  }
  builder.DisableUnrollLoops = optLevel == 0;

  builder.DisableUnrollLoops = (disableLoopUnrolling.getNumOccurrences() > 0)
                                   ? disableLoopUnrolling
                                   : optLevel == 0;

  // This is final, unless there is a #pragma vectorize enable
  if (disableLoopVectorization) {
    builder.LoopVectorize = false;
    // If option wasn't forced via cmd line (-vectorize-loops, -loop-vectorize)
  } else if (!builder.LoopVectorize) {
    builder.LoopVectorize = optLevel > 1 && sizeLevel < 2;
  }

  // When #pragma vectorize is on for SLP, do the same as above
  builder.SLPVectorize =
      disableSLPVectorization ? false : optLevel > 1 && sizeLevel < 2;

  if (opts::isSanitizerEnabled(opts::AddressSanitizer)) {
    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                         legacyAddAddressSanitizerPasses);
    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                         legacyAddAddressSanitizerPasses);
  }

  if (opts::isSanitizerEnabled(opts::MemorySanitizer)) {
    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                         legacyAddMemorySanitizerPass);
    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                         legacyAddMemorySanitizerPass);
  }

  if (opts::isSanitizerEnabled(opts::ThreadSanitizer)) {
    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                         legacyAddThreadSanitizerPass);
    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                         legacyAddThreadSanitizerPass);
  }

  if (opts::isSanitizerEnabled(opts::CoverageSanitizer)) {
    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                         legacyAddSanitizerCoveragePass);
    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                         legacyAddSanitizerCoveragePass);
  }

  if (!disableLangSpecificPasses) {
    if (!disableSimplifyDruntimeCalls) {
      builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd,
                           legacyAddSimplifyDRuntimeCallsPass);
    }

    if (!disableGCToStack) {
      builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd,
                           legacyAddGarbageCollect2StackPass);
    }
  }

  // EP_OptimizerLast does not exist in LLVM 3.0, add it manually below.
  builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                       legacyAddStripExternalsPass);

  legacyAddPGOPasses(builder, mpm, optLevel);

  builder.populateFunctionPassManager(fpm);
  builder.populateModulePassManager(mpm);
}

// This function runs optimization passes based on command line arguments.
// Returns true if any optimization passes were invoked.
bool legacy_ldc_optimize_module(llvm::Module *M) {
  // Create a PassManager to hold and optimize the collection of
  // per-module passes we are about to build.
  legacy::PassManager mpm;

  // Dont optimise spirv modules because turning GEPs into extracts triggers
  // asserts in the IR -> SPIR-V translation pass. SPIRV doesn't have a target
  // machine, so any optimisation passes that rely on it to provide analysis,
  // like DCE can't be run.
  // The optimisation is supposed to happen between the SPIRV -> native machine
  // code pass of the consumer of the binary.
  // TODO: run rudimentary optimisations to improve IR debuggability.
  if (getComputeTargetType(M) == ComputeBackend::SPIRV)
    return false;

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl *tlii =
      new TargetLibraryInfoImpl(Triple(M->getTargetTriple()));

  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (disableSimplifyLibCalls)
    tlii->disableAllFunctions();

  mpm.add(new TargetLibraryInfoWrapperPass(*tlii));

  // The DataLayout is already set at the module (in module.cpp,
  // method Module::genLLVMModule())
  // FIXME: Introduce new command line switch default-data-layout to
  // override the module data layout

  // Add internal analysis passes from the target machine.
  mpm.add(createTargetTransformInfoWrapperPass(
      gTargetMachine->getTargetIRAnalysis()));

  // Also set up a manager for the per-function passes.
  legacy::FunctionPassManager fpm(M);

  // Add internal analysis passes from the target machine.
  fpm.add(createTargetTransformInfoWrapperPass(
      gTargetMachine->getTargetIRAnalysis()));

  // If the -strip-debug command line option was specified, add it before
  // anything else.
  if (stripDebug) {
    mpm.add(createStripSymbolsPass(true));
  }

  legacyAddOptimizationPasses(mpm, fpm, optLevel(), sizeLevel());

  if (global.params.dllimport != DLLImport::none) {
    mpm.add(createDLLImportRelocationPass());
  }

  // Run per-function passes.
  fpm.doInitialization();
  for (auto &F : *M) {
    fpm.run(F);
  }
  fpm.doFinalization();

  // Run per-module passes.
  mpm.run(*M);

  // Verify the resulting module.
  if (!noVerify) {
    verifyModule(M);
  }

  // Report that we run some passes.
  return true;
}
#endif

#if LDC_LLVM_VER >= 1400
//FIXME: Something still wrong with this:
template <typename PassT>
static inline void addPass_(ModulePassManager &mpm, PassT pass) {
  mpm.addPass(pass);

  if (verifyEach) {
    mpm.addPass(VerifierPass());
  }
}

static void addAddressSanitizerPasses(ModulePassManager &mpm,
                                      OptimizationLevel level ) {
  AddressSanitizerOptions aso;
  aso.CompileKernel = false;
  aso.Recover = false;
  aso.UseAfterScope = true;
  aso.UseAfterReturn = AsanDetectStackUseAfterReturnMode::Always;
  //FIXME: Would like to call instead of three lines after
  addPass_(mpm,ModuleAddressSanitizerPass(aso));

  /*mpm.addPass(ModuleAddressSanitizerPass(aso));
  if (verifyEach) {
    mpm.addPass(VerifierPass());
  }*/
}
/**
 * Adds a set of optimization passes to the given module/function pass
 * managers based on the given optimization and size reduction levels.
 *
 * The selection mirrors Clang behavior and is based on LLVM's
 * PassManagerBuilder.
 */
//Run optimization passes using the new pass manager
void runOptimizationPasses(llvm::Module *M) {
  // Create a ModulePassManager to hold and optimize the collection of
  // per-module passes we are about to build.
  
  unsigned optLevelVal = optLevel();
  unsigned sizeLevelVal = sizeLevel();
  

  PipelineTuningOptions pto;
  
//  builder.OptLevel = optLevel;
//  builder.SizeLevel = sizeLevel;
//  builder.PrepareForLTO = opts::isUsingLTO();
//  builder.PrepareForThinLTO = opts::isUsingThinLTO();
//
//  if (willInline()) {
//    auto params = llvm::getInlineParams(optLevel, sizeLevel);
//    builder.Inliner = createFunctionInliningPass(params);
//  } else {
//    builder.Inliner = createAlwaysInlinerLegacyPass();
//  }
  pto.LoopUnrolling = optLevelVal > 0;

  pto.LoopUnrolling = !((disableLoopUnrolling.getNumOccurrences() > 0)
                                   ? disableLoopUnrolling
                                   : optLevelVal == 0);

  // This is final, unless there is a #pragma vectorize enable
  if (disableLoopVectorization) {
    pto.LoopVectorization = false;
    // If option wasn't forced via cmd line (-vectorize-loops, -loop-vectorize)
  } else if (!pto.LoopVectorization) {
    pto.LoopVectorization = optLevelVal > 1 && sizeLevelVal < 2;
  }

  // When #pragma vectorize is on for SLP, do the same as above
  pto.SLPVectorization =
      disableSLPVectorization ? false : optLevelVal > 1 && sizeLevelVal < 2;

  PassBuilder pb(gTargetMachine, pto);
 
  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  ModuleAnalysisManager mam;

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  ModulePassManager mpm;

  // Also set up a manager for the per-function passes.
  FunctionPassManager fpm;

  mpm.addPass(createModuleToFunctionPassAdaptor(std::move(fpm)));
  
  if (!noVerify) {
    fpm.addPass(VerifierPass());
  }

  if (opts::isSanitizerEnabled(opts::AddressSanitizer)) {
    pb.registerOptimizerLastEPCallback([&](ModulePassManager &mpm,
                                        OptimizationLevel level) {
        addAddressSanitizerPasses(mpm,level);
    });
//    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
//                         legacyAddAddressSanitizerPasses);
//    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
//                         legacyAddAddressSanitizerPasses);
  }
//
//  if (opts::isSanitizerEnabled(opts::MemorySanitizer)) {
//    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
//                         legacyAddMemorySanitizerPass);
//    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
//                         legacyAddMemorySanitizerPass);
//  }
//
//  if (opts::isSanitizerEnabled(opts::ThreadSanitizer)) {
//    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
//                         legacyAddThreadSanitizerPass);
//    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
//                         legacyAddThreadSanitizerPass);
//  }
//
//  if (opts::isSanitizerEnabled(opts::CoverageSanitizer)) {
//    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
//                         legacyAddSanitizerCoveragePass);
//    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
//                         legacyAddSanitizerCoveragePass);
//  }
//
//  if (!disableLangSpecificPasses) {
//    if (!disableSimplifyDruntimeCalls) {
//      builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd,
//                           legacyAddSimplifyDRuntimeCallsPass);
//    }
//
//    if (!disableGCToStack) {
//      builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd,
//                           legacyAddGarbageCollect2StackPass);
//    }
//  }
//
//  // EP_OptimizerLast does not exist in LLVM 3.0, add it manually below.
//  builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
//                       legacyAddStripExternalsPass);
//
//  legacyAddPGOPasses(builder, mpm, optLevel);
//
//  builder.populateFunctionPassManager(fpm);
//  builder.populateModulePassManager(mpm);
  mpm.run(*M,mam);
}
//Run codgen passes using the legacy pass manager
void runCodegenPasses(llvm::Module* M) {
  legacy::PassManager mpm;

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl *tlii =
      new TargetLibraryInfoImpl(Triple(M->getTargetTriple()));

  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (disableSimplifyLibCalls)
    tlii->disableAllFunctions();

  mpm.add(new TargetLibraryInfoWrapperPass(*tlii));

  // The DataLayout is already set at the module (in module.cpp,
  // method Module::genLLVMModule())
  // FIXME: Introduce new command line switch default-data-layout to
  // override the module data layout

  // Add internal analysis passes from the target machine.
  mpm.add(createTargetTransformInfoWrapperPass(
      gTargetMachine->getTargetIRAnalysis()));

  // Also set up a manager for the per-function passes.
  legacy::FunctionPassManager fpm(M);

  // Add internal analysis passes from the target machine.
  fpm.add(createTargetTransformInfoWrapperPass(
      gTargetMachine->getTargetIRAnalysis()));

  // If the -strip-debug command line option was specified, add it before
  // anything else.
  if (stripDebug) {
    mpm.add(createStripSymbolsPass(true));
  }

  if (global.params.dllimport != DLLImport::none) {
    mpm.add(createDLLImportRelocationPass());
  }

  // Run per-function passes.
  fpm.doInitialization();
  for (auto &F : *M) {
    fpm.run(F);
  }
  fpm.doFinalization();

  // Run per-module passes.
  mpm.run(*M);

}
////////////////////////////////////////////////////////////////////////////////
// This function runs optimization passes based on command line arguments.
// Returns true if any optimization passes were invoked.
bool new_ldc_optimize_module(llvm::Module *M) {
  // Dont optimise spirv modules because turning GEPs into extracts triggers
  // asserts in the IR -> SPIR-V translation pass. SPIRV doesn't have a target
  // machine, so any optimisation passes that rely on it to provide analysis,
  // like DCE can't be run.
  // The optimisation is supposed to happen between the SPIRV -> native machine
  // code pass of the consumer of the binary.
  // TODO: run rudimentary optimisations to improve IR debuggability.
  if (getComputeTargetType(M) == ComputeBackend::SPIRV)
    return false;

  runOptimizationPasses(M);
  runCodegenPasses(M);
  // Verify the resulting module.
  if (!noVerify) {
    verifyModule(M);
  }

  // Report that we run some passes.
  return true;
}

#endif
////////////////////////////////////////////////////////////////////////////////
// This function calls the fuction which runs optimization passes based on command
// line arguments.  Calls either legacy version using legacy pass manager
// or new version using the new pass managr
// Returns true if any optimization passes were invoked.
bool ldc_optimize_module(llvm::Module *M) {
#if LDC_LLVM_VER < 1400
  return legacy_ldc_optimize_module(M);
#elif LDC_LLVM_VER < 1500
  return passmanager==0 ? legacy_ldc_optimize_module(M)
                        : new_ldc_optimize_module(M);
#else
  return new_ldc_optimize_module(M);
#endif
}


// Verifies the module.
void verifyModule(llvm::Module *m) {
  Logger::println("Verifying module...");
  LOG_SCOPE;
  std::string ErrorStr;
  raw_string_ostream OS(ErrorStr);
  if (llvm::verifyModule(*m, &OS)) {
    error(Loc(), "%s", ErrorStr.c_str());
    fatal();
  }
  Logger::println("Verification passed!");
}

// Output to `hash_os` all optimization settings that influence object code
// output and that are not observable in the IR. This is used to calculate the
// hash use for caching that uniquely identifies the object file output.
void outputOptimizationSettings(llvm::raw_ostream &hash_os) {
  hash_os << optimizeLevel;
  hash_os << willInline();
  hash_os << disableLangSpecificPasses;
  hash_os << disableSimplifyDruntimeCalls;
  hash_os << disableSimplifyLibCalls;
  hash_os << disableGCToStack;
  hash_os << stripDebug;
  hash_os << disableLoopUnrolling;
  hash_os << disableLoopVectorization;
  hash_os << disableSLPVectorization;
}
