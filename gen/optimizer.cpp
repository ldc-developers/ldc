#include "gen/optimizer.h"
#include "gen/cl_helpers.h"
#include "gen/irstate.h"
#include "gen/logger.h"

#include "gen/passes/Passes.h"

#include "llvm/PassManager.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Module.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/Verifier.h"
#if LDC_LLVM_VER >= 302
#include "llvm/DataLayout.h"
#else
#include "llvm/Target/TargetData.h"
#endif
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PassNameParser.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "mars.h"       // error()

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
        clEnumValN(2, "O",  "Equivalent to -O2"),
        clEnumValN(0, "O0", "No optimizations (default)"),
        clEnumValN(1, "O1", "Simple optimizations"),
        clEnumValN(2, "O2", "Good optimizations"),
        clEnumValN(3, "O3", "Aggressive optimizations"),
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
disableSimplifyRuntimeCalls("disable-simplify-drtcalls",
    cl::desc("Disable simplification of druntime calls"),
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
#if LDC_LLVM_VER < 302
    const int opt = optLevel();
    // Use same appoach as clang (see lib/CodeGen/BackendUtil.cpp)
    llvm::CodeGenOpt::Level codeGenOptLevel = llvm::CodeGenOpt::Default;
    // Debug info doesn't work properly with CodeGenOpt <> None
    if (global.params.symdebug || !opt) codeGenOptLevel = llvm::CodeGenOpt::None;
    else if (opt >= 3) codeGenOptLevel = llvm::CodeGenOpt::Aggressive;
    return codeGenOptLevel;
#else
    // There's a bug in llvm:LiveInterval::createDeadDef()
    // which prevents use of other values.
    // Happens only with 3.2 trunk.
    return llvm::CodeGenOpt::None;
#endif
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

#if USE_METADATA
static void addGarbageCollect2StackPass(const PassManagerBuilder &builder, PassManagerBase &pm) {
    if (builder.OptLevel >= 2 && builder.SizeLevel == 0)
        addPass(pm, createGarbageCollect2Stack());
}
#endif

/**
 * Adds a set of optimization passes to the given module/function pass
 * managers based on the given optimization and size reduction levels.
 *
 * The selection mirrors Clang behavior and is based on LLVM's
 * PassManagerBuilder.
 */
static void addOptimizationPasses(PassManagerBase &mpm, FunctionPassManager &fpm,
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
    builder.DisableUnitAtATime = !unitAtATime;
    builder.DisableUnrollLoops = optLevel == 0;
    /* builder.Vectorize is set in ctor from command line switch */

    if (!disableLangSpecificPasses) {
        if (!disableSimplifyRuntimeCalls)
            builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd, addSimplifyDRuntimeCallsPass);

#if USE_METADATA
        if (!disableGCToStack)
            Builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd, addGarbageCollect2StackPass);
#endif // USE_METADATA
    }

#if LDC_LLVM_VER >= 301
    // EP_OptimizerLast does not exist in LLVM 3.0, add it manually below.
    builder.addExtension(PassManagerBuilder::EP_OptimizerLast, addStripExternalsPass);
#endif

    builder.populateFunctionPassManager(fpm);
    builder.populateModulePassManager(mpm);

#if LDC_LLVM_VER < 301
    addStripExternalsPass(builder, mpm);
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////
// This function runs optimization passes based on command line arguments.
// Returns true if any optimization passes were invoked.
bool ldc_optimize_module(llvm::Module* m)
{
    // Create a PassManager to hold and optimize the collection of
    // per-module passes we are about to build.
    PassManager mpm;

    // Add an appropriate TargetLibraryInfo pass for the module's triple.
    TargetLibraryInfo *tli = new TargetLibraryInfo(Triple(m->getTargetTriple()));

    // The -disable-simplify-libcalls flag actually disables all builtin optzns.
    if (disableSimplifyRuntimeCalls)
        tli->disableAllFunctions();
    mpm.add(tli);

    // Add an appropriate TargetData instance for this module.
#if LDC_LLVM_VER >= 302
    mpm.add(new DataLayout(m));
#else
    mpm.add(new TargetData(m));
#endif

    // Also set up a manager for the per-function passes.
    FunctionPassManager fpm(m);
#if LDC_LLVM_VER >= 302
    fpm.add(new DataLayout(m));
#else
    fpm.add(new TargetData(m));
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
                error("Can't create pass '-%s' (%s)", arg, pass->getPassName());
            else
                error("Can't create pass (%s)", pass->getPassName());
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
    for (llvm::Module::iterator F = m->begin(), E = m->end(); F != E; ++F)
        fpm.run(*F);
    fpm.doFinalization();

    // Run per-module passes.
    mpm.run(*m);

    // Verify the resulting module.
    verifyModule(m);

    // Report that we run some passes.
    return true;
}

// Verifies the module.
void verifyModule(llvm::Module* m) {
    if (!noVerify) {
        std::string verifyErr;
        Logger::println("Verifying module...");
        LOG_SCOPE;
        if (llvm::verifyModule(*m, llvm::ReturnStatusAction, &verifyErr))
        {
            error("%s", verifyErr.c_str());
            fatal();
        }
        else {
            Logger::println("Verification passed!");
        }
    }
}
