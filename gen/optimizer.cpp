//===-- optimizer.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/optimizer.h"
#include "errors.h"
#include "gen/cl_helpers.h"
#include "gen/logger.h"
#include "gen/passes/Passes.h"
#include "llvm/LinkAllPasses.h"
#if LDC_LLVM_VER >= 307
#include "llvm/IR/LegacyPassManager.h"
#else
#include "llvm/PassManager.h"
#endif
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/ADT/Triple.h"
#if LDC_LLVM_VER >= 307
#include "llvm/Analysis/TargetTransformInfo.h"
#endif
#include "llvm/IR/Verifier.h"
#if LDC_LLVM_VER >= 307
#include "llvm/Analysis/TargetLibraryInfo.h"
#else
#include "llvm/Target/TargetLibraryInfo.h"
#endif
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

extern llvm::TargetMachine *gTargetMachine;
using namespace llvm;

static cl::opt<signed char> optimizeLevel(
    cl::desc("Setting the optimization level:"), cl::ZeroOrMore,
    clEnumValues(
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

static cl::opt<bool> noVerify("disable-verify",
                              cl::desc("Do not verify result module"),
                              cl::Hidden);

static cl::opt<bool>
    verifyEach("verify-each",
               cl::desc("Run verifier after D-specific and explicitly "
                        "specified optimization passes"),
               cl::Hidden, cl::ZeroOrMore);

static cl::opt<bool>
    disableLangSpecificPasses("disable-d-passes",
                              cl::desc("Disable all D-specific passes"),
                              cl::ZeroOrMore);

static cl::opt<bool> disableSimplifyDruntimeCalls(
    "disable-simplify-drtcalls",
    cl::desc("Disable simplification of druntime calls"), cl::ZeroOrMore);

static cl::opt<bool> disableSimplifyLibCalls(
    "disable-simplify-libcalls",
    cl::desc("Disable simplification of well-known C runtime calls"),
    cl::ZeroOrMore);

static cl::opt<bool> disableGCToStack(
    "disable-gc2stack",
    cl::desc("Disable promotion of GC allocations to stack memory"),
    cl::ZeroOrMore);

static cl::opt<cl::boolOrDefault, false, opts::FlagParser<cl::boolOrDefault>>
    enableInlining(
        "inlining",
        cl::desc("Enable function inlining (default in -O2 and higher)"),
        cl::ZeroOrMore);

static llvm::cl::opt<llvm::cl::boolOrDefault, false,
                     opts::FlagParser<llvm::cl::boolOrDefault>>
    enableCrossModuleInlining(
        "cross-module-inlining",
        llvm::cl::desc("Enable cross-module function inlining (default "
                       "disabled) (LLVM >= 3.7)"),
        llvm::cl::ZeroOrMore, llvm::cl::Hidden);

static cl::opt<bool> unitAtATime("unit-at-a-time", cl::desc("Enable basic IPO"),
                                 cl::init(true));

static cl::opt<bool> stripDebug(
    "strip-debug",
    cl::desc("Strip symbolic debug information before optimization"));

cl::opt<opts::SanitizerCheck> opts::sanitize(
    "sanitize", cl::desc("Enable runtime instrumentation for bug detection"),
    cl::init(opts::None),
    clEnumValues(clEnumValN(opts::AddressSanitizer, "address", "Memory errors"),
                 clEnumValN(opts::MemorySanitizer, "memory", "Memory errors"),
                 clEnumValN(opts::ThreadSanitizer, "thread",
                            "Race detection")));

static cl::opt<bool> disableLoopUnrolling(
    "disable-loop-unrolling",
    cl::desc("Disable loop unrolling in all relevant passes"), cl::init(false));
static cl::opt<bool>
    disableLoopVectorization("disable-loop-vectorization",
                             cl::desc("Disable the loop vectorization pass"),
                             cl::init(false));

static cl::opt<bool>
    disableSLPVectorization("disable-slp-vectorization",
                            cl::desc("Disable the slp vectorization pass"),
                            cl::init(false));

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
#if LDC_LLVM_VER >= 307
  return enableCrossModuleInlining == llvm::cl::BOU_TRUE;
#else
// Cross-module inlining is disabled for <3.7 because we don't emit symbols in
// COMDAT any groups pre-LLVM3.7. With cross-module inlining enabled, without
// COMDAT any there are multiple-def linker errors when linking druntime.
// See supportsCOMDAT().
  return false;
#endif
}

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

static inline void addPass(PassManagerBase &pm, Pass *pass) {
  pm.add(pass);

  if (verifyEach) {
    pm.add(createVerifierPass());
  }
}

static void addStripExternalsPass(const PassManagerBuilder &builder,
                                  PassManagerBase &pm) {
  if (builder.OptLevel >= 1) {
    addPass(pm, createStripExternalsPass());
    addPass(pm, createGlobalDCEPass());
  }
}

static void addSimplifyDRuntimeCallsPass(const PassManagerBuilder &builder,
                                         PassManagerBase &pm) {
  if (builder.OptLevel >= 2 && builder.SizeLevel == 0) {
    addPass(pm, createSimplifyDRuntimeCalls());
  }
}

static void addGarbageCollect2StackPass(const PassManagerBuilder &builder,
                                        PassManagerBase &pm) {
  if (builder.OptLevel >= 2 && builder.SizeLevel == 0) {
    addPass(pm, createGarbageCollect2Stack());
  }
}

static void addAddressSanitizerPasses(const PassManagerBuilder &Builder,
                                      PassManagerBase &PM) {
  PM.add(createAddressSanitizerFunctionPass());
  PM.add(createAddressSanitizerModulePass());
}

static void addMemorySanitizerPass(const PassManagerBuilder &Builder,
                                   PassManagerBase &PM) {
  PM.add(createMemorySanitizerPass());

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

static void addThreadSanitizerPass(const PassManagerBuilder &Builder,
                                   PassManagerBase &PM) {
  PM.add(createThreadSanitizerPass());
}

static void addInstrProfilingPass(legacy::PassManagerBase &mpm) {
#if LDC_WITH_PGO
  if (global.params.genInstrProf) {
    InstrProfOptions options;
    options.NoRedZone = global.params.disableRedZone;
    if (global.params.datafileInstrProf)
      options.InstrProfileOutput = global.params.datafileInstrProf;
#if LDC_LLVM_VER >= 309
    mpm.add(createInstrProfilingLegacyPass(options));
#else
    mpm.add(createInstrProfilingPass(options));
#endif
  }
#endif
}

/**
 * Adds a set of optimization passes to the given module/function pass
 * managers based on the given optimization and size reduction levels.
 *
 * The selection mirrors Clang behavior and is based on LLVM's
 * PassManagerBuilder.
 */
#if LDC_LLVM_VER >= 307
static void addOptimizationPasses(legacy::PassManagerBase &mpm,
                                  legacy::FunctionPassManager &fpm,
#else
static void addOptimizationPasses(PassManagerBase &mpm,
                                  FunctionPassManager &fpm,
#endif
                                  unsigned optLevel, unsigned sizeLevel) {
  if (!noVerify) {
    fpm.add(createVerifierPass());
  }

  PassManagerBuilder builder;
  builder.OptLevel = optLevel;
  builder.SizeLevel = sizeLevel;

  if (willInline()) {
    unsigned threshold = 225;
    if (sizeLevel == 1) { // -Os
      threshold = 75;
    } else if (sizeLevel == 2) { // -Oz
      threshold = 25;
    }
    if (optLevel > 2) {
      threshold = 275;
    }
    builder.Inliner = createFunctionInliningPass(threshold);
  } else {
#if LDC_LLVM_VER >= 400
    builder.Inliner = createAlwaysInlinerLegacyPass();
#else
    builder.Inliner = createAlwaysInlinerPass();
#endif
  }
  builder.DisableUnitAtATime = !unitAtATime;
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

  if (opts::sanitize == opts::AddressSanitizer) {
    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                         addAddressSanitizerPasses);
    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                         addAddressSanitizerPasses);
  }

  if (opts::sanitize == opts::MemorySanitizer) {
    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                         addMemorySanitizerPass);
    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                         addMemorySanitizerPass);
  }

  if (opts::sanitize == opts::ThreadSanitizer) {
    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                         addThreadSanitizerPass);
    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                         addThreadSanitizerPass);
  }

  if (!disableLangSpecificPasses) {
    if (!disableSimplifyDruntimeCalls) {
      builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd,
                           addSimplifyDRuntimeCallsPass);
    }

    if (!disableGCToStack) {
      builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd,
                           addGarbageCollect2StackPass);
    }
  }

  // EP_OptimizerLast does not exist in LLVM 3.0, add it manually below.
  builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                       addStripExternalsPass);

  addInstrProfilingPass(mpm);

  builder.populateFunctionPassManager(fpm);
  builder.populateModulePassManager(mpm);
}

////////////////////////////////////////////////////////////////////////////////
// This function runs optimization passes based on command line arguments.
// Returns true if any optimization passes were invoked.
bool ldc_optimize_module(llvm::Module *M) {
// Create a PassManager to hold and optimize the collection of
// per-module passes we are about to build.
#if LDC_LLVM_VER >= 307
  legacy::
#endif
      PassManager mpm;

#if LDC_LLVM_VER >= 307
  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl *tlii =
      new TargetLibraryInfoImpl(Triple(M->getTargetTriple()));

  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (disableSimplifyLibCalls)
    tlii->disableAllFunctions();

  mpm.add(new TargetLibraryInfoWrapperPass(*tlii));
#else
  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfo *tli = new TargetLibraryInfo(Triple(M->getTargetTriple()));

  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (disableSimplifyLibCalls) {
    tli->disableAllFunctions();
  }

  mpm.add(tli);
#endif

// Add an appropriate DataLayout instance for this module.
#if LDC_LLVM_VER >= 307
// The DataLayout is already set at the module (in module.cpp,
// method Module::genLLVMModule())
// FIXME: Introduce new command line switch default-data-layout to
// override the module data layout
#elif LDC_LLVM_VER == 306
  mpm.add(new DataLayoutPass());
#else
                                    const DataLayout *DL = M->getDataLayout();
                                    assert(DL &&
                                           "DataLayout not set at module");
                                    mpm.add(new DataLayoutPass(*DL));
#endif

#if LDC_LLVM_VER >= 307
  // Add internal analysis passes from the target machine.
  mpm.add(createTargetTransformInfoWrapperPass(
      gTargetMachine->getTargetIRAnalysis()));
#else
  // Add internal analysis passes from the target machine.
  gTargetMachine->addAnalysisPasses(mpm);
#endif

// Also set up a manager for the per-function passes.
#if LDC_LLVM_VER >= 307
  legacy::
#endif
      FunctionPassManager fpm(M);

#if LDC_LLVM_VER >= 307
  // Add internal analysis passes from the target machine.
  fpm.add(createTargetTransformInfoWrapperPass(
      gTargetMachine->getTargetIRAnalysis()));
#elif LDC_LLVM_VER >= 306
  fpm.add(new DataLayoutPass());
  gTargetMachine->addAnalysisPasses(fpm);
#else
                                    fpm.add(new DataLayoutPass(M));
                                    gTargetMachine->addAnalysisPasses(fpm);
#endif

  // If the -strip-debug command line option was specified, add it before
  // anything else.
  if (stripDebug) {
    mpm.add(createStripSymbolsPass(true));
  }

  addOptimizationPasses(mpm, fpm, optLevel(), sizeLevel());

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

// Output to `hash_os` all optimization settings that influence object code output
// and that are not observable in the IR.
// This is used to calculate the hash use for caching that uniquely identifies
// the object file output.
void outputOptimizationSettings(llvm::raw_ostream &hash_os) {
  hash_os << optimizeLevel;
  hash_os << willInline();
  hash_os << disableLangSpecificPasses;
  hash_os << disableSimplifyDruntimeCalls;
  hash_os << disableSimplifyLibCalls;
  hash_os << disableGCToStack;
  hash_os << unitAtATime;
  hash_os << stripDebug;
  hash_os << opts::sanitize;
  hash_os << disableLoopUnrolling;
  hash_os << disableLoopVectorization;
  hash_os << disableSLPVectorization;
}
