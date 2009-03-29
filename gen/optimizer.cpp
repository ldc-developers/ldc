#include "gen/optimizer.h"

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

static cl::opt<char> optimizeLevel(
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

static cl::opt<bool> enableInlining("enable-inlining",
    cl::desc("Enable function inlining (in -O<N>, if given)"),
    cl::ZeroOrMore,
    cl::init(false));

// Some accessors for the linker: (llvm-ld version only, currently unused?)
bool doInline() {
    return enableInlining;
}

int optLevel() {
    return optimizeLevel;
}

bool optimize() {
    return optimizeLevel || enableInlining || passList.empty();
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
        pm.add(createPromoteMemoryToRegisterPass());
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
    }

    // -inline
    if (enableInlining) {
        pm.add(createFunctionInliningPass());
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
    if (!enableInlining && optimizeLevel == 0 && passList.empty())
        return false;

    PassManager pm;
    pm.add(new TargetData(m));

    bool optimize = optimizeLevel != 0 || enableInlining;

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
