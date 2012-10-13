#include "gen/optimizer.h"
#include "gen/cl_helpers.h"
#include "gen/logger.h"

#include "gen/passes/Passes.h"

#include "llvm/PassManager.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/Verifier.h"
#if LDC_LLVM_VER >= 302
#include "llvm/DataLayout.h"
#else
#include "llvm/Target/TargetData.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PassNameParser.h"
#include "llvm/Transforms/IPO.h"

#include "mars.h"       // error()
#include "root.h"
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

static cl::opt<opts::BoolOrDefaultAdapter, false, opts::FlagParser>
enableInlining("inlining",
    cl::desc("(*) Enable function inlining in -O<N>"),
    cl::ZeroOrMore);

#if LDC_LLVM_VER >= 301
static cl::opt<bool>
runVectorization("vectorize", cl::desc("Run vectorization passes"));

static cl::opt<bool>
useGVNAfterVectorization("use-gvn-after-vectorization",
    cl::init(false), cl::Hidden,
    cl::desc("Run GVN instead of Early CSE after vectorization passes"));
#endif

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
        // Add alias analysis passes.
        // This is at least required for FunctionAttrs pass.
        addPass(pm, createTypeBasedAliasAnalysisPass());
        addPass(pm, createBasicAliasAnalysisPass());
        //addPass(pm, createStripDeadPrototypesPass());
        addPass(pm, createGlobalDCEPass());
        addPass(pm, createPromoteMemoryToRegisterPass());
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
        addPass(pm, createGVNPass());
    }

    // -inline
    if (doInline()) {
        addPass(pm, createFunctionInliningPass());

        if (optimizeLevel >= 2) {
            // Run some optimizations to clean up after inlining.
            addPass(pm, createScalarReplAggregatesPass());
            addPass(pm, createInstructionCombiningPass());
            // -instcombine + gvn == devirtualization :)
            addPass(pm, createGVNPass());

            // Inline again, to catch things like now nonvirtual
            // function calls, foreach delegates passed to inlined
            // opApply's, etc. where the actual function being called
            // wasn't known during the first inliner pass.
            addPass(pm, createFunctionInliningPass());
        }
    }

    if (optimizeLevel >= 2) {
        if (!disableLangSpecificPasses) {
            if (!disableSimplifyRuntimeCalls)
                addPass(pm, createSimplifyDRuntimeCalls());

#if USE_METADATA
            if (!disableGCToStack)
                addPass(pm, createGarbageCollect2Stack());
#endif // USE_METADATA
        }
        // Run some clean-up passes
        addPass(pm, createInstructionCombiningPass());
        addPass(pm, createScalarReplAggregatesPass());
        addPass(pm, createCFGSimplificationPass());
        addPass(pm, createInstructionCombiningPass());
    }

    // -O3
    if (optimizeLevel >= 3)
    {
        addPass(pm, createArgumentPromotionPass());
        addPass(pm, createSimplifyLibCallsPass());
        addPass(pm, createInstructionCombiningPass());
        addPass(pm, createJumpThreadingPass());
        addPass(pm, createCFGSimplificationPass());
        addPass(pm, createScalarReplAggregatesPass());
        addPass(pm, createInstructionCombiningPass());
        addPass(pm, createConstantPropagationPass());

        addPass(pm, createReassociatePass());
        addPass(pm, createLoopRotatePass());
        addPass(pm, createLICMPass());
        addPass(pm, createLoopUnswitchPass());
        addPass(pm, createInstructionCombiningPass());
        addPass(pm, createIndVarSimplifyPass());
        addPass(pm, createLoopDeletionPass());
        addPass(pm, createLoopUnrollPass());
        addPass(pm, createInstructionCombiningPass());
        addPass(pm, createGVNPass());
        addPass(pm, createMemCpyOptPass());
        addPass(pm, createSCCPPass());

        addPass(pm, createInstructionCombiningPass());
        addPass(pm, createConstantPropagationPass());

        addPass(pm, createDeadStoreEliminationPass());
        addPass(pm, createAggressiveDCEPass());
        addPass(pm, createCFGSimplificationPass());
        addPass(pm, createConstantMergePass());
    }

#if LDC_LLVM_VER >= 301
    // -vectorize
    if (runVectorization)
    {
        addPass(pm, createBBVectorizePass());
        addPass(pm, createInstructionCombiningPass());
        if (optimizeLevel > 1 && useGVNAfterVectorization)
            addPass(pm, createGVNPass());                   // Remove redundancies
        else
            addPass(pm, createEarlyCSEPass());              // Catch trivial redundancies
    }
#endif

    if (optimizeLevel >= 1) {
        addPass(pm, createStripExternalsPass());
        addPass(pm, createGlobalDCEPass());
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

    if (verifyEach) pm.add(createVerifierPass());

#if LDC_LLVM_VER >= 302
    addPass(pm, new DataLayout(m));
#else
    addPass(pm, new TargetData(m));
#endif

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

    pm.run(*m);

    verifyModule(m);

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
