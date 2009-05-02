#include "gen/optimizer.h"
#include "gen/cl_helpers.h"

#include "gen/passes/Passes.h"

#include "llvm/PassManager.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PassNameParser.h"

#include "root.h"       // error()

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

// Some extra accessors for the linker: (llvm-ld version only, currently unused?)
int optLevel() {
    return optimizeLevel;
}

bool optimize() {
    return optimizeLevel || doInline() || !passList.empty();
}

// this function inserts some or all of the std-compile-opts passes depending on the
// optimization level given.
static void addPassesForOptLevel(PassManager& pm) {
    // -O1
    if (optimizeLevel >= 1)
    {
        //pm.add(createStripDeadPrototypesPass());
        pm.add(createGlobalDCEPass());
        pm.add(createRaiseAllocationsPass());
        pm.add(createCFGSimplificationPass());
        if (optimizeLevel == 1)
            pm.add(createPromoteMemoryToRegisterPass());
        else
            pm.add(createScalarReplAggregatesPass());
        pm.add(createGlobalOptimizerPass());
        pm.add(createGlobalDCEPass());
    }

    // -O2
    if (optimizeLevel >= 2)
    {
        pm.add(createIPConstantPropagationPass());
        pm.add(createDeadArgEliminationPass());
        pm.add(createInstructionCombiningPass());
        pm.add(createCFGSimplificationPass());
        pm.add(createPruneEHPass());
        if (!disableLangSpecificPasses && !disableGCToStack)
            pm.add(createGarbageCollect2Stack());
    }

    // -inline
    if (doInline()) {
        pm.add(createFunctionInliningPass());
        
        if (optimizeLevel >= 2) {
            // Run some optimizations to clean up after inlining.
            pm.add(createScalarReplAggregatesPass());
            pm.add(createInstructionCombiningPass());
            if (!disableLangSpecificPasses && !disableGCToStack)
                pm.add(createGarbageCollect2Stack());
            
            // Inline again, to catch things like foreach delegates
            // passed to inlined opApply's where the function wasn't
            // known during the first inliner pass.
            pm.add(createFunctionInliningPass());
            
            // Run clean-up again.
            pm.add(createScalarReplAggregatesPass());
            pm.add(createInstructionCombiningPass());
            if (!disableLangSpecificPasses && !disableGCToStack)
                pm.add(createGarbageCollect2Stack());
        }
    }

    if (optimizeLevel >= 2 && !disableLangSpecificPasses) {
        if (!disableSimplifyRuntimeCalls)
            pm.add(createSimplifyDRuntimeCalls());
        
        if (!disableGCToStack) {
            // Run some clean-up after the last GC to stack promotion pass.
            pm.add(createScalarReplAggregatesPass());
            pm.add(createInstructionCombiningPass());
            pm.add(createCFGSimplificationPass());
        }
    }

    // -O3
    if (optimizeLevel >= 3)
    {
        pm.add(createArgumentPromotionPass());
        pm.add(createTailDuplicationPass());
        pm.add(createInstructionCombiningPass());
        pm.add(createCFGSimplificationPass());
        pm.add(createScalarReplAggregatesPass());
        pm.add(createInstructionCombiningPass());
        pm.add(createCondPropagationPass());

        pm.add(createTailCallEliminationPass());
        pm.add(createCFGSimplificationPass());
        pm.add(createReassociatePass());
        pm.add(createLoopRotatePass());
        pm.add(createLICMPass());
        pm.add(createLoopUnswitchPass());
        pm.add(createInstructionCombiningPass());
        pm.add(createIndVarSimplifyPass());
        pm.add(createLoopUnrollPass());
        pm.add(createInstructionCombiningPass());
        pm.add(createGVNPass());
        pm.add(createSCCPPass());

        pm.add(createInstructionCombiningPass());
        pm.add(createCondPropagationPass());

        pm.add(createDeadStoreEliminationPass());
        pm.add(createAggressiveDCEPass());
        pm.add(createCFGSimplificationPass());
        pm.add(createSimplifyLibCallsPass());
        pm.add(createDeadTypeEliminationPass());
        pm.add(createConstantMergePass());
    }

    // level -O4 and -O5 are linktime optimizations
}

//////////////////////////////////////////////////////////////////////////////////////////
// This function runs optimization passes based on command line arguments.
// Returns true if any optimization passes were invoked.
bool ldc_optimize_module(llvm::Module* m)
{
    if (!optimize())
        return false;

    PassManager pm;
    pm.add(new TargetData(m));

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
            pm.add(ctor());
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

    pm.run(*m);
    return true;
}
