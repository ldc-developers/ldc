#include "gen/optimizer.h"
#include "gen/cl_helpers.h"

#include "gen/passes/Passes.h"

#include "llvm/PassManager.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PassNameParser.h"

#include "root.h"       // error()
#include <cstring>      // strcmp();

using namespace llvm;

// Allow the user to specify specific optimizations to run.
static cl::list<const PassInfo*, bool, PassNameParser>
    passList(
        cl::desc("Running specific optimizations:"),
        cl::Hidden      // to clean up --help output
    );

static cl::opt<unsigned char> optimizeLevel(
    cl::desc("Setting the optimization level:"),
    cl::ZeroOrMore,
    cl::values(
        clEnumValN(2, "O",  "Equivalent to -O2"),
        clEnumValN(0, "O0", "No optimizations (default)"),
        clEnumValN(1, "O1", "Simple optimizations"),
        clEnumValN(2, "O2", "Good optimizations"),
        clEnumValN(3, "O3", "Aggressive optimizations"),
        clEnumValN(4, "O4", "Link-time optimization"), //  not implemented?
        clEnumValN(5, "O5", "Link-time optimization"), //  not implemented?
        clEnumValEnd),
    cl::init(0));

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

#ifdef USE_METADATA
static cl::opt<bool>
disableGCToStack("disable-gc2stack",
    cl::desc("Disable promotion of GC allocations to stack memory in -O<N>"),
    cl::ZeroOrMore);

// Not recommended; metadata currently triggers an assert in the backend...
static cl::opt<bool>
disableStripMetaData("disable-strip-metadata",
    cl::desc("Disable default metadata stripping (not recommended)"),
    cl::ZeroOrMore);
#endif

static cl::opt<opts::BoolOrDefaultAdapter, false, opts::FlagParser>
enableInlining("inlining",
    cl::desc("(*) Enable function inlining in -O<N>"),
    cl::ZeroOrMore);

// Determine whether or not to run the inliner as part of the default list of
// optimization passes.
// If not explicitly specified, treat as false for -O0-2, and true for -O3.
bool doInline() {
    return enableInlining == cl::BOU_TRUE
        || (enableInlining == cl::BOU_UNSET && optimizeLevel >= 3);
}

// Determine whether the inliner will be run.
bool willInline() {
    if (doInline())
        return true;
    // It may also have been specified explicitly on the command line as an explicit pass
    typedef cl::list<const PassInfo*, bool, PassNameParser> PL;
    for (PL::iterator I = passList.begin(), E = passList.end(); I != E; ++I) {
        if (!std::strcmp((*I)->getPassArgument(), "inline"))
            return true;
    }
    return false;
}

// Some extra accessors for the linker: (llvm-ld version only, currently unused?)
int optLevel() {
    return optimizeLevel;
}

bool optimize() {
    return optimizeLevel || doInline() || !passList.empty();
}

static void addPass(PassManager& pm, Pass* pass) {
    pm.add(pass);
    
    if (verifyEach) pm.add(createVerifierPass());
}

// this function inserts some or all of the std-compile-opts passes depending on the
// optimization level given.
static void addPassesForOptLevel(PassManager& pm) {
    // -O1
    if (optimizeLevel >= 1)
    {
        //addPass(pm, createStripDeadPrototypesPass());
        addPass(pm, createGlobalDCEPass());
        addPass(pm, createRaiseAllocationsPass());
        addPass(pm, createCFGSimplificationPass());
        if (optimizeLevel == 1)
            addPass(pm, createPromoteMemoryToRegisterPass());
        else
            addPass(pm, createScalarReplAggregatesPass());
        addPass(pm, createGlobalOptimizerPass());
    }

    // -O2
    if (optimizeLevel >= 2)
    {
        addPass(pm, createIPConstantPropagationPass());
        addPass(pm, createDeadArgEliminationPass());
        addPass(pm, createInstructionCombiningPass());
        addPass(pm, createCFGSimplificationPass());
        addPass(pm, createPruneEHPass());
        addPass(pm, createFunctionAttrsPass());
        addPass(pm, createTailCallEliminationPass());
        addPass(pm, createCFGSimplificationPass());
    }

    // -inline
    if (doInline()) {
        addPass(pm, createFunctionInliningPass());

        if (optimizeLevel >= 2) {
            // Run some optimizations to clean up after inlining.
            addPass(pm, createScalarReplAggregatesPass());
            addPass(pm, createInstructionCombiningPass());

            // Inline again, to catch things like foreach delegates
            // passed to inlined opApply's where the function wasn't
            // known during the first inliner pass.
            addPass(pm, createFunctionInliningPass());

            // Run clean-up again.
            addPass(pm, createScalarReplAggregatesPass());
            addPass(pm, createInstructionCombiningPass());
        }
    }

    if (optimizeLevel >= 2 && !disableLangSpecificPasses) {
        if (!disableSimplifyRuntimeCalls)
            addPass(pm, createSimplifyDRuntimeCalls());

#ifdef USE_METADATA
        if (!disableGCToStack) {
            addPass(pm, createGarbageCollect2Stack());
            // Run some clean-up
            addPass(pm, createInstructionCombiningPass());
            addPass(pm, createScalarReplAggregatesPass());
            addPass(pm, createCFGSimplificationPass());
        }
#endif
    }

    // -O3
    if (optimizeLevel >= 3)
    {
        addPass(pm, createArgumentPromotionPass());
        addPass(pm, createTailDuplicationPass());
        addPass(pm, createInstructionCombiningPass());
        addPass(pm, createCFGSimplificationPass());
        addPass(pm, createScalarReplAggregatesPass());
        addPass(pm, createInstructionCombiningPass());
        addPass(pm, createCondPropagationPass());

        addPass(pm, createReassociatePass());
        addPass(pm, createLoopRotatePass());
        addPass(pm, createLICMPass());
        addPass(pm, createLoopUnswitchPass());
        addPass(pm, createInstructionCombiningPass());
        addPass(pm, createIndVarSimplifyPass());
        addPass(pm, createLoopUnrollPass());
        addPass(pm, createInstructionCombiningPass());
        addPass(pm, createGVNPass());
        addPass(pm, createMemCpyOptPass());
        addPass(pm, createSCCPPass());

        addPass(pm, createInstructionCombiningPass());
        addPass(pm, createCondPropagationPass());

        addPass(pm, createDeadStoreEliminationPass());
        addPass(pm, createAggressiveDCEPass());
        addPass(pm, createCFGSimplificationPass());
        addPass(pm, createSimplifyLibCallsPass());
        addPass(pm, createDeadTypeEliminationPass());
        addPass(pm, createConstantMergePass());
    }

    if (optimizeLevel >= 1) {
#if LLVM_REV >= 68940
        addPass(pm, createStripExternalsPass());
#endif
        addPass(pm, createGlobalDCEPass());
    }

    // level -O4 and -O5 are linktime optimizations
}

//////////////////////////////////////////////////////////////////////////////////////////
// This function runs optimization passes based on command line arguments.
// Returns true if any optimization passes were invoked.
bool ldc_optimize_module(llvm::Module* m)
{
    if (!optimize()) {
#ifdef USE_METADATA
        if (!disableStripMetaData) {
            // This one always needs to run if metadata is generated, because
            // the code generator will assert if it's not used.
            ModulePass* stripMD = createStripMetaData();
            stripMD->runOnModule(*m);
            delete stripMD;
        }
#endif
        return false;
    }

    PassManager pm;
    
    if (verifyEach) pm.add(createVerifierPass());
    
    addPass(pm, new TargetData(m));

    bool optimize = optimizeLevel != 0 || doInline();

    unsigned optPos = optimizeLevel != 0
                    ? optimizeLevel.getPosition()
                    : enableInlining.getPosition();

    for (size_t i = 0; i < passList.size(); i++) {
        // insert -O<N> / -enable-inlining in right position
        if (optimize && optPos < passList.getPosition(i)) {
            addPassesForOptLevel(pm);
            optimize = false;
        }

        const PassInfo* pass = passList[i];
        if (PassInfo::NormalCtor_t ctor = pass->getNormalCtor()) {
            addPass(pm, ctor());
        } else {
            const char* arg = pass->getPassArgument(); // may return null
            if (arg)
                error("Can't create pass '-%s' (%s)", arg, pass->getPassName());
            else
                error("Can't create pass (%s)", pass->getPassName());
            assert(0);  // Should be unreachable; root.h:error() calls exit()
        }
    }
    // insert -O<N> / -enable-inlining if specified at the end,
    if (optimize)
        addPassesForOptLevel(pm);

#ifdef USE_METADATA
    if (!disableStripMetaData) {
        // This one is purposely not disabled by disableLangSpecificPasses
        // because the code generator will assert if it's not used.
        addPass(pm, createStripMetaData());
    }
#endif

    pm.run(*m);
    return true;
}
