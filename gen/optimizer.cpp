//===-- optimizer.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
// This module is compiled into both the compiler and the JIT runtime library
// (with predefined IN_JITRT).
//
//===----------------------------------------------------------------------===//

#ifdef IN_JITRT
#include "runtime/jit-rt/cpp-so/optimizer.h"
#include "runtime/jit-rt/cpp-so/valueparser.h"
#include "runtime/jit-rt/cpp-so/utils.h"
#endif

#include "gen/optimizer.h"

#ifndef IN_JITRT
#include "dmd/errors.h"
#include "gen/logger.h"
#endif

#include "gen/passes/GarbageCollect2Stack.h"
#include "gen/passes/StripExternals.h"
#include "gen/passes/SimplifyDRuntimeCalls.h"
#include "gen/passes/Passes.h"

#ifndef IN_JITRT
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/cl_options_sanitizers.h"
#include "driver/plugins.h"
#include "driver/targetmachine.h"
#endif

#if LDC_LLVM_VER < 1700
#include "llvm/ADT/Triple.h"
#else
#include "llvm/TargetParser/Triple.h"
#endif
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/IPO.h"
#if LDC_LLVM_VER < 1700
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#endif
#include "llvm/Transforms/Instrumentation/MemorySanitizer.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
#include "llvm/Transforms/Instrumentation/InstrProfiling.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Instrumentation/SanitizerCoverage.h"

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

#ifndef IN_JITRT
static cl::opt<cl::boolOrDefault, false, opts::FlagParser<cl::boolOrDefault>>
    enableInlining(
        "inlining", cl::ZeroOrMore,
        cl::desc("(*) Enable function inlining (default in -O2 and higher)"));

static cl::opt<cl::boolOrDefault, false, opts::FlagParser<cl::boolOrDefault>>
    enableCrossModuleInlining(
        "cross-module-inlining", cl::ZeroOrMore, cl::Hidden,
        cl::desc("(*) Enable cross-module function inlining (default disabled)"));
#endif

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

unsigned optLevel() {
  // Use -O2 as a base for the size-optimization levels.
  return optimizeLevel >= 0 ? optimizeLevel : 2;
}

static unsigned sizeLevel() { return optimizeLevel < 0 ? -optimizeLevel : 0; }

// Determines whether or not to run the normal, full inlining pass.
bool willInline() {
#ifdef IN_JITRT
  return false;
#else
  return enableInlining == cl::BOU_TRUE ||
         (enableInlining == cl::BOU_UNSET && optLevel() > 1);
#endif
}

bool willCrossModuleInline() {
#ifdef IN_JITRT
  return false;
#else
  return enableCrossModuleInlining == llvm::cl::BOU_TRUE && willInline();
#endif
}

bool isOptimizationEnabled() { return optimizeLevel != 0; }

llvm::CodeGenOptLevel codeGenOptLevel() {
  // Use same appoach as clang (see lib/CodeGen/BackendUtil.cpp)
  if (optLevel() == 0) {
    return llvm::CodeGenOptLevel::None;
  }
  if (optLevel() >= 3) {
    return llvm::CodeGenOptLevel::Aggressive;
  }
  return llvm::CodeGenOptLevel::Default;
}

std::unique_ptr<TargetLibraryInfoImpl> createTLII(llvm::Module &M) {
  auto tlii = new TargetLibraryInfoImpl(Triple(M.getTargetTriple()));
  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (disableSimplifyLibCalls)
    tlii->disableAllFunctions();
  return std::unique_ptr<TargetLibraryInfoImpl>(tlii);
}

static OptimizationLevel getOptimizationLevel(){
  switch(optimizeLevel) {
    case 0: return OptimizationLevel::O0;
    case 1: return OptimizationLevel::O1;
    case 2: return OptimizationLevel::O2;
    case 3:
    case 4:
    case 5: return OptimizationLevel::O3;
    case -1: return OptimizationLevel::Os;
    case -2: return OptimizationLevel::Oz;
  }
  //This should never be reached
  llvm_unreachable("Unexpected optimizeLevel.");
  return OptimizationLevel::O0;
}

#ifndef IN_JITRT
static void addAddressSanitizerPasses(ModulePassManager &mpm,
                                      OptimizationLevel level ) {
  AddressSanitizerOptions aso;
  aso.CompileKernel = false;
  aso.Recover = opts::isSanitizerRecoveryEnabled(opts::AddressSanitizer);
  aso.UseAfterScope = true;
  aso.UseAfterReturn = opts::fSanitizeAddressUseAfterReturn;

#if LDC_LLVM_VER >= 1600
  mpm.addPass(AddressSanitizerPass(aso));
#else
  mpm.addPass(ModuleAddressSanitizerPass(aso));
#endif
}

static void addMemorySanitizerPass(ModulePassManager &mpm,
                                   FunctionPassManager &fpm,
                                   OptimizationLevel level ) {
  int trackOrigins = fSanitizeMemoryTrackOrigins;
  bool recover = opts::isSanitizerRecoveryEnabled(opts::MemorySanitizer);
  bool kernel = false;
#if LDC_LLVM_VER >= 1600
  mpm.addPass(MemorySanitizerPass(
      MemorySanitizerOptions{trackOrigins, recover, kernel}));
#else
  fpm.addPass(MemorySanitizerPass(
      MemorySanitizerOptions{trackOrigins, recover, kernel}));
#endif

  // MemorySanitizer inserts complex instrumentation that mostly follows
  // the logic of the original code, but operates on "shadow" values.
  // It can benefit from re-running some general purpose optimization passes.
  if (level != OptimizationLevel::O0) {
    fpm.addPass(EarlyCSEPass());
    fpm.addPass(ReassociatePass());
    //FIXME: Fix these parameters
    fpm.addPass(createFunctionToLoopPassAdaptor(LICMPass(128,128,false)));
    fpm.addPass(GVNPass());
    //FIXME: Not sure what to do with these?
    //fpm.addPass(InstructionCombiningPass());
    //fpm.addPass(DeadStoreEliminationPass());
  }
}
static void addThreadSanitizerPass(ModulePassManager &mpm,
                                      OptimizationLevel level ) {
  mpm.addPass(ModuleThreadSanitizerPass());
  mpm.addPass(createModuleToFunctionPassAdaptor(ThreadSanitizerPass()));
}

static void addSanitizerCoveragePass(ModulePassManager &mpm,
                                      OptimizationLevel level ) {
#if LDC_LLVM_VER >= 1600
  mpm.addPass(SanitizerCoveragePass(
      opts::getSanitizerCoverageOptions()));
#else
  mpm.addPass(ModuleSanitizerCoveragePass(
      opts::getSanitizerCoverageOptions()));
#endif
}
// Adds PGO instrumentation generation and use passes.
static void addPGOPasses(ModulePassManager &mpm,
                                      OptimizationLevel level ) {
  if (opts::isInstrumentingForASTBasedPGO()) {
    InstrProfOptions options;
    options.NoRedZone = global.params.disableRedZone;
    if (global.params.datafileInstrProf)
      options.InstrProfileOutput = global.params.datafileInstrProf;
    mpm.addPass(
#if LDC_LLVM_VER < 1800
      InstrProfiling(options)
#else
      InstrProfilingLoweringPass(options)
#endif // LDC_LLVM_VER < 1800
    );
  } else if (opts::isUsingASTBasedPGOProfile()) {
    // We are generating code with PGO profile information available.
    // Do indirect call promotion from -O1
    if (level != OptimizationLevel::O0) {
      mpm.addPass(PGOIndirectCallPromotion());
    }
  }
}
#endif // !IN_JITRT

static void addStripExternalsPass(ModulePassManager &mpm,
                                      OptimizationLevel level ) {

  if (level == OptimizationLevel::O1 || level == OptimizationLevel::O2 ||
      level == OptimizationLevel::O3) {
    mpm.addPass(StripExternalsPass());
    if (verifyEach) {
      mpm.addPass(VerifierPass());
    }
    mpm.addPass(GlobalDCEPass());
  }
}

static void addSimplifyDRuntimeCallsPass(ModulePassManager &mpm,
                                      OptimizationLevel level ) {
  if (level == OptimizationLevel::O2  || level == OptimizationLevel::O3) {
    mpm.addPass(createModuleToFunctionPassAdaptor(SimplifyDRuntimeCallsPass()));
    if (verifyEach) {
      mpm.addPass(VerifierPass());
    }
  }
}

static void addGarbageCollect2StackPass(ModulePassManager &mpm,
                                         OptimizationLevel level ) {
  if (level == OptimizationLevel::O2  || level == OptimizationLevel::O3) {
    mpm.addPass(createModuleToFunctionPassAdaptor(GarbageCollect2StackPass()));
    if (verifyEach) {
      mpm.addPass(VerifierPass());
    }
  }
}

#ifndef IN_JITRT
static llvm::Optional<PGOOptions> getPGOOptions() {
  // FIXME: Do we have these anywhere?
  bool debugInfoForProfiling = false;
  bool pseudoProbeForProfiling = false;
  if (opts::isInstrumentingForIRBasedPGO()) {
    return PGOOptions(
        global.params.datafileInstrProf, "", "",
#if LDC_LLVM_VER >= 1700
        "" /*MemoryProfileUsePath*/, llvm::vfs::getRealFileSystem(),
#endif
        PGOOptions::PGOAction::IRInstr, PGOOptions::CSPGOAction::NoCSAction,
#if LDC_LLVM_VER >= 1900
        PGOOptions::ColdFuncOpt::Default,
#endif
        debugInfoForProfiling, pseudoProbeForProfiling);
  } else if (opts::isUsingIRBasedPGOProfile()) {
    return PGOOptions(
        global.params.datafileInstrProf, "", "",
#if LDC_LLVM_VER >= 1700
        "" /*MemoryProfileUsePath*/, llvm::vfs::getRealFileSystem(),
#endif
        PGOOptions::PGOAction::IRUse, PGOOptions::CSPGOAction::NoCSAction,
#if LDC_LLVM_VER >= 1900
        PGOOptions::ColdFuncOpt::Default,
#endif
        debugInfoForProfiling, pseudoProbeForProfiling);
  } else if (opts::isUsingSampleBasedPGOProfile()) {
    return PGOOptions(
        global.params.datafileInstrProf, "", "",
#if LDC_LLVM_VER >= 1700
        "" /*MemoryProfileUsePath*/, llvm::vfs::getRealFileSystem(),
#endif
        PGOOptions::PGOAction::SampleUse, PGOOptions::CSPGOAction::NoCSAction,
#if LDC_LLVM_VER >= 1900
        PGOOptions::ColdFuncOpt::Default,
#endif
        debugInfoForProfiling, pseudoProbeForProfiling);
  }
#if LDC_LLVM_VER < 1600
  return None;
#else
  return std::nullopt;
#endif
}
#endif // !IN_JITRT

static PipelineTuningOptions getPipelineTuningOptions(unsigned optLevelVal, unsigned sizeLevelVal) {
  PipelineTuningOptions pto;

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

  return pto;
}
/**
 * Adds a set of optimization passes to the given module/function pass
 * managers based on the given optimization and size reduction levels.
 *
 * The selection mirrors Clang behavior and is based on LLVM's
 * PassManagerBuilder.
 */
//Run optimization passes using the new pass manager
void runOptimizationPasses(llvm::Module *M, llvm::TargetMachine *TM) {
  // Create a ModulePassManager to hold and optimize the collection of
  // per-module passes we are about to build.

  unsigned optLevelVal = optLevel();
  unsigned sizeLevelVal = sizeLevel();

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
  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  ModuleAnalysisManager mam;


  PassInstrumentationCallbacks pic;
  PrintPassOptions ppo;
  //FIXME: Where should these come from
  bool debugLogging = false;
  ppo.Indent = false;
  ppo.SkipAnalyses = false;
#if LDC_LLVM_VER < 1600
  StandardInstrumentations si(debugLogging, /*VerifyEach=*/false, ppo);
#else
  StandardInstrumentations si(M->getContext(), debugLogging, /*VerifyEach=*/false, ppo);
#endif

#if LDC_LLVM_VER < 1700
  si.registerCallbacks(pic, &fam);
#else
  si.registerCallbacks(pic, &mam);
#endif

  PassBuilder pb(TM, getPipelineTuningOptions(optLevelVal, sizeLevelVal),
#ifdef IN_JITRT
                 {}, &pic);
#else
                 getPGOOptions(), &pic);
#endif

  // register the target library analysis directly because clang does :)
  auto tlii = createTLII(*M);
  fam.registerPass([&] { return TargetLibraryAnalysis(*tlii); });

  ModulePassManager mpm;

  if (!noVerify) {
    pb.registerPipelineStartEPCallback([&](ModulePassManager &mpm,
                                          OptimizationLevel level) {
      mpm.addPass(VerifierPass());
    });
  }

  // TODO: port over strip-debuginfos pass for -strip-debug

#ifndef IN_JITRT
  pb.registerPipelineStartEPCallback(addPGOPasses);

  if (opts::isSanitizerEnabled(opts::AddressSanitizer)) {
    pb.registerOptimizerLastEPCallback(addAddressSanitizerPasses);
  }

  if (opts::isSanitizerEnabled(opts::MemorySanitizer)) {
    pb.registerOptimizerLastEPCallback(
        [&](ModulePassManager &mpm, OptimizationLevel level) {
          FunctionPassManager fpm;
          addMemorySanitizerPass(mpm, fpm, level);
          mpm.addPass(createModuleToFunctionPassAdaptor(std::move(fpm)));
        });
  }

  if (opts::isSanitizerEnabled(opts::ThreadSanitizer)) {
    pb.registerOptimizerLastEPCallback(addThreadSanitizerPass);
  }

  if (opts::isSanitizerEnabled(opts::CoverageSanitizer)) {
    pb.registerOptimizerLastEPCallback(addSanitizerCoveragePass);
  }
#endif // !IN_JITRT

  if (!disableLangSpecificPasses) {
    if (!disableSimplifyDruntimeCalls) {
      // FIXME: Is this registerOptimizerLastEPCallback correct here
      //(had registerLoopOptimizerEndEPCallback) but that seems wrong
      pb.registerOptimizerLastEPCallback(addSimplifyDRuntimeCallsPass);
    }
    if (!disableGCToStack) {
      // FIXME: This should be checked
      fam.registerPass([&] { return DominatorTreeAnalysis(); });
      mam.registerPass([&] { return CallGraphAnalysis(); });
      // FIXME: Is this registerOptimizerLastEPCallback correct here
      //(had registerLoopOptimizerEndEPCallback) but that seems wrong
      pb.registerOptimizerLastEPCallback(addGarbageCollect2StackPass);
    }
  }

  pb.registerOptimizerLastEPCallback(addStripExternalsPass);

#ifndef IN_JITRT
  registerAllPluginsWithPassBuilder(pb);
#endif

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);


  OptimizationLevel level = getOptimizationLevel();

  if (optLevelVal == 0) {
#ifdef IN_JITRT
    mpm = pb.buildO0DefaultPipeline(level, false);
#else
    mpm = pb.buildO0DefaultPipeline(level, opts::isUsingLTO());
#if LDC_LLVM_VER >= 1700
  } else if (opts::ltoFatObjects && opts::isUsingLTO()) {
    mpm = pb.buildFatLTODefaultPipeline(level,
                                        opts::isUsingThinLTO(),
                                        opts::isUsingThinLTO()
    );
#endif
  } else if (opts::isUsingThinLTO()) {
    mpm = pb.buildThinLTOPreLinkDefaultPipeline(level);
  } else if (opts::isUsingLTO()) {
    mpm = pb.buildLTOPreLinkDefaultPipeline(level);
#endif // !IN_JITRT
  } else {
    mpm = pb.buildPerModuleDefaultPipeline(level);
  }


  mpm.run(*M,mam);
}
////////////////////////////////////////////////////////////////////////////////
// This function runs optimization passes based on command line arguments.
// Returns true if any optimization passes were invoked.
bool ldc_optimize_module(llvm::Module *M, llvm::TargetMachine *TM) {
#ifndef IN_JITRT
  // Dont optimise spirv modules because turning GEPs into extracts triggers
  // asserts in the IR -> SPIR-V translation pass. SPIRV doesn't have a target
  // machine, so any optimisation passes that rely on it to provide analysis,
  // like DCE can't be run.
  // The optimisation is supposed to happen between the SPIRV -> native machine
  // code pass of the consumer of the binary.
  // TODO: run rudimentary optimisations to improve IR debuggability.
  if (getComputeTargetType(M) == ComputeBackend::SPIRV)
    return false;
#endif

  runOptimizationPasses(M, TM);

  // Verify the resulting module.
  if (!noVerify) {
    verifyModule(M);
  }

  // Report that we run some passes.
  return true;
}

#ifdef IN_JITRT
void optimizeModule(const OptimizerSettings &settings, llvm::Module *M,
                    llvm::TargetMachine *TM) {
  if (settings.sizeLevel > 0) {
    optimizeLevel = -settings.sizeLevel;
  } else {
    optimizeLevel = settings.optLevel;
  }

  ldc_optimize_module(M, TM);
}
#endif // IN_JITRT

// Verifies the module.
void verifyModule(llvm::Module *m) {
#ifndef IN_JITRT
  Logger::println("Verifying module...");
  LOG_SCOPE;
#endif
  std::string ErrorStr;
  raw_string_ostream OS(ErrorStr);
  if (llvm::verifyModule(*m, &OS)) {
#ifndef IN_JITRT
    error(Loc(), "%s", ErrorStr.c_str());
    fatal();
#else
    assert(false && "Verification failed!");
#endif
  }
#ifndef IN_JITRT
  Logger::println("Verification passed!");
#endif
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

#ifdef IN_JITRT
void setRtCompileVar(const Context &context, llvm::Module &module,
                     const char *name, const void *init) {
  assert(nullptr != name);
  assert(nullptr != init);
  auto var = module.getGlobalVariable(name);
  if (nullptr != var) {
    auto type = var->getValueType();
    auto initializer =
        parseInitializer(module.getDataLayout(), *type, init,
                         [&](const std::string &str) { fatal(context, str); });
    var->setConstant(true);
    var->setInitializer(initializer);
    var->setLinkage(llvm::GlobalValue::PrivateLinkage);
  }
}
#endif // IN_JITRT
