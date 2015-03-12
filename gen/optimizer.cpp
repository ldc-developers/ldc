//===-- optimizer.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/optimizer.h"
#include "mars.h"       // error()
#include "gen/cl_helpers.h"
#include "gen/logger.h"
#include "gen/passes/Passes.h"
#include "llvm/LinkAllPasses.h"
#if LDC_LLVM_VER >= 307
#include "llvm/IR/LegacyPassManager.h"
#else
#include "llvm/PassManager.h"
#endif
#if LDC_LLVM_VER >= 303
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#else
#include "llvm/Module.h"
#if LDC_LLVM_VER == 302
#include "llvm/DataLayout.h"
#else
#include "llvm/Target/TargetData.h"
#endif
#endif
#include "llvm/ADT/Triple.h"
#if LDC_LLVM_VER >= 307
#include "llvm/Analysis/TargetTransformInfo.h"
#endif
#if LDC_LLVM_VER >= 305
#include "llvm/IR/Verifier.h"
#else
#include "llvm/Analysis/Verifier.h"
#endif
#if LDC_LLVM_VER >= 307
#include "llvm/Analysis/TargetLibraryInfo.h"
#else
#include "llvm/Target/TargetLibraryInfo.h"
#endif
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#if LDC_LLVM_VER >= 305
#include "llvm/IR/LegacyPassNameParser.h"
#else
#include "llvm/Support/PassNameParser.h"
#endif
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

extern llvm::TargetMachine* gTargetMachine;
using namespace llvm;

// Allow the user to specify specific optimizations to run.
static cl::list<const PassInfo*, bool, PassNameParser>
    passList(
        cl::desc("Running specific optimizations:"),
        cl::Hidden      // to clean up --help output
    );

static cl::opt<signed char> optimizeLevel(
    cl::desc("Setting the optimization level:"),
    cl::ZeroOrMore,
    cl::values(
        clEnumValN(3, "O",  "Equivalent to -O3"),
        clEnumValN(0, "O0", "No optimizations (default)"),
        clEnumValN(1, "O1", "Simple optimizations"),
        clEnumValN(2, "O2", "Good optimizations"),
        clEnumValN(3, "O3", "Aggressive optimizations"),
        clEnumValN(6, "Ofast", "Additionally enables fast-math"),
        clEnumValN(4, "O4", "Link-time optimization"), // Not implemented yet.
        clEnumValN(5, "O5", "Link-time optimization"), // Not implemented yet.
        clEnumValN(-1, "Os", "Like -O2 with extra optimizations for size"),
        clEnumValN(-2, "Oz", "Like -Os but reduces code size further"),
        clEnumValEnd),
    cl::init(0));

static cl::opt<bool>
noVerify("disable-verify",
    cl::desc("Do not verify result module"),
    cl::Hidden);

static cl::opt<bool>
verifyEach("verify-each",
    cl::desc("Run verifier after D-specific and explicitly specified optimization passes"),
    cl::Hidden,
    cl::ZeroOrMore);

static cl::opt<bool>
disableLangSpecificPasses("disable-d-passes",
    cl::desc("Disable all D-specific passes"),
    cl::ZeroOrMore);

static cl::opt<bool>
disableSimplifyDruntimeCalls("disable-simplify-drtcalls",
    cl::desc("Disable simplification of druntime calls"),
    cl::ZeroOrMore);

static cl::opt<bool>
disableSimplifyLibCalls("disable-simplify-libcalls",
    cl::desc("Disable simplification of well-known C runtime calls"),
    cl::ZeroOrMore);

static cl::opt<bool>
disableGCToStack("disable-gc2stack",
    cl::desc("Disable promotion of GC allocations to stack memory"),
    cl::ZeroOrMore);

static cl::opt<opts::BoolOrDefaultAdapter, false, opts::FlagParser>
enableInlining("inlining",
    cl::desc("Enable function inlining (default in -O2 and higher)"),
    cl::ZeroOrMore);

static cl::opt<bool>
unitAtATime("unit-at-a-time",
            cl::desc("Enable basic IPO"),
            cl::init(true));

static cl::opt<bool>
stripDebug("strip-debug",
           cl::desc("Strip symbolic debug information before optimization"));

#if LDC_LLVM_VER >= 303
cl::opt<opts::SanitizerCheck> opts::sanitize("sanitize",
    cl::desc("Enable runtime instrumentation for bug detection"),
    cl::init(opts::None),
    cl::values(
        clEnumValN(opts::AddressSanitizer, "address", "memory errors"),
        clEnumValN(opts::MemorySanitizer, "memory", "memory errors"),
        clEnumValN(opts::ThreadSanitizer, "thread", "race detection"),
        clEnumValEnd));
#endif

#if LDC_LLVM_VER >= 304
static cl::opt<bool>
disableLoopUnrolling("disable-loop-unrolling",
                     cl::desc("Disable loop unrolling in all relevant passes"),
                     cl::init(false));
static cl::opt<bool>
disableLoopVectorization("disable-loop-vectorization",
                     cl::desc("Disable the loop vectorization pass"),
                     cl::init(false));

static cl::opt<bool>
disableSLPVectorization("disable-slp-vectorization",
                        cl::desc("Disable the slp vectorization pass"),
                        cl::init(false));
#endif

static unsigned optLevel() {
    // Use -O2 as a base for the size-optimization levels.
    return optimizeLevel >= 0 ? optimizeLevel : 2;
}

static unsigned sizeLevel() {
    return optimizeLevel < 0 ? -optimizeLevel : 0;
}

// Determines whether or not to run the normal, full inlining pass.
bool willInline() {
    return enableInlining == cl::BOU_TRUE ||
        (enableInlining == cl::BOU_UNSET && optLevel() > 1);
}

llvm::CodeGenOpt::Level codeGenOptLevel() {
    const int opt = optLevel();
    llvm::CodeGenOpt::Level codeGenOptLevel;
    switch (opt)
    {
    case 0:
        codeGenOptLevel = llvm::CodeGenOpt::None;
        break;
    case 1:
        codeGenOptLevel = llvm::CodeGenOpt::Less;
        break;
    case 2:
        codeGenOptLevel = llvm::CodeGenOpt::Default;
        break;
    default:
        codeGenOptLevel = llvm::CodeGenOpt::Aggressive;
        break;
    }
    return codeGenOptLevel;
}

static inline void addPass(PassManagerBase& pm, Pass* pass) {
    pm.add(pass);

    if (verifyEach) pm.add(createVerifierPass());
}

static void addStripExternalsPass(const PassManagerBuilder &builder, PassManagerBase &pm) {
    if (builder.OptLevel >= 1) {
        addPass(pm, createStripExternalsPass());
        addPass(pm, createGlobalDCEPass());
    }
}

static void addSimplifyDRuntimeCallsPass(const PassManagerBuilder &builder, PassManagerBase &pm) {
    if (builder.OptLevel >= 2 && builder.SizeLevel == 0)
        addPass(pm, createSimplifyDRuntimeCalls());
}

static void addGarbageCollect2StackPass(const PassManagerBuilder &builder, PassManagerBase &pm) {
    if (builder.OptLevel >= 2 && builder.SizeLevel == 0)
        addPass(pm, createGarbageCollect2Stack());
}

#if LDC_LLVM_VER >= 303
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
#endif

/**
 * Adds a set of optimization passes to the given module/function pass
 * managers based on the given optimization and size reduction levels.
 *
 * The selection mirrors Clang behavior and is based on LLVM's
 * PassManagerBuilder.
 */
#if LDC_LLVM_VER >= 307
static void addOptimizationPasses(legacy::PassManagerBase &mpm, legacy::FunctionPassManager &fpm,
#else
static void addOptimizationPasses(PassManagerBase &mpm, FunctionPassManager &fpm,
#endif
                                  unsigned optLevel, unsigned sizeLevel) {
    fpm.add(createVerifierPass());                  // Verify that input is correct

    PassManagerBuilder builder;
    builder.OptLevel = optLevel;
    builder.SizeLevel = sizeLevel;

    if (willInline()) {
        unsigned threshold = 225;
        if (sizeLevel == 1)      // -Os
            threshold = 75;
        else if (sizeLevel == 2) // -Oz
            threshold = 25;
        if (optLevel > 2)
            threshold = 275;
        builder.Inliner = createFunctionInliningPass(threshold);
    } else {
        builder.Inliner = createAlwaysInlinerPass();
    }
#if LDC_LLVM_VER < 304
    builder.DisableSimplifyLibCalls = disableSimplifyLibCalls;
#endif
    builder.DisableUnitAtATime = !unitAtATime;
    builder.DisableUnrollLoops = optLevel == 0;

#if LDC_LLVM_VER >= 304
    builder.DisableUnrollLoops = (disableLoopUnrolling.getNumOccurrences() > 0) ?
                                  disableLoopUnrolling : optLevel == 0;

    // This is final, unless there is a #pragma vectorize enable
    if (disableLoopVectorization)
        builder.LoopVectorize = false;
    // If option wasn't forced via cmd line (-vectorize-loops, -loop-vectorize)
    else if (!builder.LoopVectorize)
      builder.LoopVectorize = optLevel > 1 && sizeLevel < 2;

    // When #pragma vectorize is on for SLP, do the same as above
    builder.SLPVectorize =
        disableSLPVectorization ? false : optLevel > 1 && sizeLevel < 2;
#else
    /* builder.Vectorize is set in ctor from command line switch */
#endif

#if LDC_LLVM_VER >= 303
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
#endif

    if (!disableLangSpecificPasses) {
        if (!disableSimplifyDruntimeCalls)
            builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd, addSimplifyDRuntimeCallsPass);

        if (!disableGCToStack)
            builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd, addGarbageCollect2StackPass);
    }

    // EP_OptimizerLast does not exist in LLVM 3.0, add it manually below.
    builder.addExtension(PassManagerBuilder::EP_OptimizerLast, addStripExternalsPass);

    builder.populateFunctionPassManager(fpm);
    builder.populateModulePassManager(mpm);
}

//////////////////////////////////////////////////////////////////////////////////////////
// This function runs optimization passes based on command line arguments.
// Returns true if any optimization passes were invoked.
bool ldc_optimize_module(llvm::Module *M)
{
    // Create a PassManager to hold and optimize the collection of
    // per-module passes we are about to build.
#if LDC_LLVM_VER >= 307
    legacy::
#endif
    PassManager mpm;

#if LDC_LLVM_VER >= 307
    // Add an appropriate TargetLibraryInfo pass for the module's triple.
    TargetLibraryInfoImpl *tlii = new TargetLibraryInfoImpl(Triple(M->getTargetTriple()));

    // The -disable-simplify-libcalls flag actually disables all builtin optzns.
    if (disableSimplifyLibCalls)
      tlii->disableAllFunctions();

    mpm.add(new TargetLibraryInfoWrapperPass(*tlii));
#else
    // Add an appropriate TargetLibraryInfo pass for the module's triple.
    TargetLibraryInfo *tli = new TargetLibraryInfo(Triple(M->getTargetTriple()));

    // The -disable-simplify-libcalls flag actually disables all builtin optzns.
    if (disableSimplifyLibCalls)
      tli->disableAllFunctions();

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
#elif LDC_LLVM_VER == 305
    const DataLayout *DL = M->getDataLayout();
    assert(DL && "DataLayout not set at module");
    mpm.add(new DataLayoutPass(*DL));
#elif LDC_LLVM_VER >= 302
    mpm.add(new DataLayout(M));
#else
    mpm.add(new TargetData(M));
#endif

#if LDC_LLVM_VER >= 307
    // Add internal analysis passes from the target machine.
    mpm.add(createTargetTransformInfoWrapperPass(gTargetMachine->getTargetIRAnalysis()));
#elif LDC_LLVM_VER >= 305
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
    fpm.add(createTargetTransformInfoWrapperPass(gTargetMachine->getTargetIRAnalysis()));
#elif LDC_LLVM_VER >= 306
    fpm.add(new DataLayoutPass());
    gTargetMachine->addAnalysisPasses(fpm);
#elif LDC_LLVM_VER == 305
    fpm.add(new DataLayoutPass(M));
    gTargetMachine->addAnalysisPasses(fpm);
#elif LDC_LLVM_VER >= 302
    fpm.add(new DataLayout(M));
#else
    fpm.add(new TargetData(M));
#endif

    // If the -strip-debug command line option was specified, add it before
    // anything else.
    if (stripDebug)
        mpm.add(createStripSymbolsPass(true));

    bool defaultsAdded = false;
    // Create a new optimization pass for each one specified on the command line
    for (unsigned i = 0; i < passList.size(); ++i) {
        if (optimizeLevel && optimizeLevel.getPosition() < passList.getPosition(i)) {
            addOptimizationPasses(mpm, fpm, optLevel(), sizeLevel());
            defaultsAdded = true;
        }

        const PassInfo *passInf = passList[i];
        Pass *pass = 0;
        if (passInf->getNormalCtor())
            pass = passInf->getNormalCtor()();
        else {
            const char* arg = passInf->getPassArgument(); // may return null
            if (arg)
                error(Loc(), "Can't create pass '-%s' (%s)", arg, pass->getPassName());
            else
                error(Loc(), "Can't create pass (%s)", pass->getPassName());
            llvm_unreachable("pass creation failed");
        }
        if (pass) {
            addPass(mpm, pass);
        }
    }

    // Add the default passes for the specified optimization level.
    if (!defaultsAdded)
        addOptimizationPasses(mpm, fpm, optLevel(), sizeLevel());

    // Run per-function passes.
    fpm.doInitialization();
    for (llvm::Module::iterator F = M->begin(), E = M->end(); F != E; ++F)
        fpm.run(*F);
    fpm.doFinalization();

    // Run per-module passes.
    mpm.run(*M);

    // Verify the resulting module.
    verifyModule(M);

    // Report that we run some passes.
    return true;
}

// Verifies the module.
void verifyModule(llvm::Module* m) {
    if (!noVerify) {
        Logger::println("Verifying module...");
        LOG_SCOPE;
        std::string ErrorStr;
#if LDC_LLVM_VER >= 305
        raw_string_ostream OS(ErrorStr);
        if (llvm::verifyModule(*m, &OS))
#else
        if (llvm::verifyModule(*m, llvm::ReturnStatusAction, &ErrorStr))
#endif
        {
            error(Loc(), "%s", ErrorStr.c_str());
            fatal();
        }
        else {
            Logger::println("Verification passed!");
        }
    }
}
