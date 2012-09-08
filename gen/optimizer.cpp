#include "gen/optimizer.h"
#include "gen/cl_helpers.h"
#include "gen/irstate.h"
#include "gen/logger.h"

#include "gen/passes/Passes.h"

#include "llvm/PassManager.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Module.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PassNameParser.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "mars.h"       // error()

#if _MSC_VER >= 1700
#include <algorithm>
#include <functional>
#endif

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
        clEnumValN( 2, "O",  "Equivalent to -O2"),
        clEnumValN( 0, "O0", "No optimizations (default)"),
        clEnumValN( 1, "O1", "Simple optimizations"),
        clEnumValN( 2, "O2", "Good optimizations"),
        clEnumValN( 3, "O3", "Aggressive optimizations"),
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
    cl::desc("Run verifier after each optimization pass"),
    cl::Hidden,
    cl::ZeroOrMore);

static cl::opt<bool>
disableLangSpecificPasses("disable-d-passes",
    cl::desc("Disable D-specific passes in -O<N>"),
    cl::ZeroOrMore);

static cl::opt<bool>
disableSimplifyRuntimeCalls("disable-simplify-drtcalls",
    cl::desc("Disable simplification of runtime calls in -O<N>"),
    cl::ZeroOrMore);

static cl::opt<bool>
disableGCToStack("disable-gc2stack",
    cl::desc("Disable promotion of GC allocations to stack memory in -O<N>"),
    cl::ZeroOrMore);

static cl::opt<bool>
disableInlining("disable-inlining", cl::desc("Do not run the inliner pass"));

static cl::opt<bool>
unitAtATime("unit-at-a-time",
            cl::desc("Enable IPO"),
            cl::init(true));

static cl::opt<bool>
stripDebug("strip-debug",
           cl::desc("Strip debugger symbol info from translation unit"));

// Determine whether or not to run the inliner as part of the default list of
// optimization passes.
// If not explicitly specified, treat as false for -O0-2, and true for -O3.
bool doInline() {
    return !disableInlining && optLevel() > 1;
}

// Determine whether the inliner will be run.
bool willInline() {
    return !disableInlining && optLevel() > 1;
}

// Some extra accessors for the linker: (llvm-ld version only, currently unused?)
unsigned optLevel() {
    return optimizeLevel >= 0 ? optimizeLevel : 2;
}

unsigned sizeLevel() {
    return optimizeLevel < 0 ? -optimizeLevel : 0;
}

bool optimize() {
    return optimizeLevel || doInline() || !passList.empty();
}

llvm::CodeGenOpt::Level codeGenOptLevel() {
#if LDC_LLVM_VER < 302
    const int opt = optLevel();
    // Use same appoach as clang (see lib/CodeGen/BackendUtil.cpp)
    llvm::CodeGenOpt::Level codeGenOptLevel = llvm::CodeGenOpt::Default;
    // Debug info doesn't work properly with CodeGenOpt <> None
    if (global.params.symdebug || !opt) codeGenOptLevel = llvm::CodeGenOpt::None;
    else if (opt >= 3) codeGenOptLevel = llvm::CodeGenOpt::Aggressive;
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
    if (builder.OptLevel >= 1)
        addPass(pm, createStripExternalsPass());
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

/// AddOptimizationPasses - This routine adds optimization passes
/// based on selected optimization level, OptLevel. This routine
/// duplicates llvm-gcc behaviour.
///
/// OptLevel - Optimization Level
static void addOptimizationPasses(PassManagerBase &mpm,FunctionPassManager &fpm,
                                    unsigned optLevel, unsigned sizeLevel) {
    fpm.add(createVerifierPass());                  // Verify that input is correct

    PassManagerBuilder builder;
    builder.OptLevel = optLevel;
    builder.SizeLevel = sizeLevel;

    if (disableInlining) {
        // No inlining pass
    } else if (optLevel > 1) {
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
    builder.DisableSimplifyLibCalls = disableSimplifyRuntimeCalls;
    /* builder.Vectorize is set in ctor from command line switch */

    if (!disableLangSpecificPasses) {
        if (!disableSimplifyRuntimeCalls)
            builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd, addSimplifyDRuntimeCallsPass);

#if USE_METADATA
        Builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd, addGarbageCollect2StackPass);
#endif // USE_METADATA
    }
    builder.addExtension(PassManagerBuilder::EP_OptimizerLast, addStripExternalsPass);

    builder.populateFunctionPassManager(fpm);
    builder.populateModulePassManager(mpm);
}

//////////////////////////////////////////////////////////////////////////////////////////
// This function runs optimization passes based on command line arguments.
// Returns true if any optimization passes were invoked.
bool ldc_optimize_module(llvm::Module* m)
{
    // Early exit if no optimization requested
    if (!optimize())
        return false;

    // Create a PassManager to hold and optimize the collection of passes we are
    // about to build.
    PassManager pm;

    // Add an appropriate TargetLibraryInfo pass for the module's triple.
    TargetLibraryInfo *tli = new TargetLibraryInfo(Triple(m->getTargetTriple()));

    // The -disable-simplify-libcalls flag actually disables all builtin optzns.
    if (disableSimplifyRuntimeCalls)
        tli->disableAllFunctions();
    pm.add(tli);

    // Add an appropriate TargetData instance for this module.
    if (gTargetData)
        pm.add(new TargetData(*gTargetData));

    OwningPtr<FunctionPassManager> fpm;
    if (optimize()) {
        fpm.reset(new FunctionPassManager(m));
        if (gTargetData)
            fpm->add(new TargetData(*gTargetData));
    }

    // If the -strip-debug command line option was specified, add it.  If
    if (stripDebug)
        addPass(pm, createStripSymbolsPass(true));

    bool doOptimize = true;
    // Create a new optimization pass for each one specified on the command line
    for (unsigned i = 0; i < passList.size(); ++i) {
        if (optimizeLevel && optimizeLevel.getPosition() < passList.getPosition(i)) {
            addOptimizationPasses(pm, *fpm, optLevel(), sizeLevel());
            doOptimize = false;
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
            addPass(pm, pass);
        }
    }

    if (doOptimize)
        addOptimizationPasses(pm, *fpm, optLevel(), sizeLevel());

    if (optimize()) {
        fpm->doInitialization();
#if _MSC_VER >= 1700
        // With a recent C++ Standard Library this is the way to go
        for_each(m->begin(), m->end(), bind(&FunctionPassManager::run, &(*fpm), std::placeholders::_1));
#else
        for (llvm::Module::iterator F = m->begin(), E = m->end(); F != E; ++F)
            fpm->run(*F);
#endif
        fpm->doFinalization();
    }

    // Check that the module is well formed on completion of optimization
    if (!noVerify && !verifyEach)
        pm.add(createVerifierPass());

    // Now that we have all of the passes ready, run them.
    pm.run(*m);

    // Verify module again.
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
