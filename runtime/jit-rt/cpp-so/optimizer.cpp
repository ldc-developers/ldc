//===-- optimizer.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
// Uses some parts from gen/optimizer.cpp which is under the BSD-style LDC
// license.
//
//===----------------------------------------------------------------------===//

#include "optimizer.h"

#include "llvm/Target/TargetMachine.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#if LDC_LLVM_VER < 1700
#include "llvm/ADT/Triple.h"
#else
#include "llvm/TargetParser/Triple.h"
#endif

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"

#include "llvm/Support/CommandLine.h"

#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "context.h"
#include "utils.h"
#include "valueparser.h"

#ifdef LDC_DYNAMIC_COMPILE_USE_CUSTOM_PASSES
#include "Passes.h"
#endif

namespace {
namespace cl = llvm::cl;
cl::opt<bool>
    verifyEach("verify-each", cl::ZeroOrMore, cl::Hidden,
               cl::desc("Run verifier after D-specific and explicitly "
                        "specified optimization passes"));

/// LDC LICENSE START
#ifdef LDC_DYNAMIC_COMPILE_USE_CUSTOM_PASSES
cl::opt<bool>
    disableLangSpecificPasses("disable-d-passes", cl::ZeroOrMore,
                              cl::desc("Disable all D-specific passes"));

cl::opt<bool> disableSimplifyDruntimeCalls(
    "disable-simplify-drtcalls", cl::ZeroOrMore,
    cl::desc("Disable simplification of druntime calls"));

cl::opt<bool> disableSimplifyLibCalls(
    "disable-simplify-libcalls", cl::ZeroOrMore,
    cl::desc("Disable simplification of well-known C runtime calls"));

cl::opt<bool> disableGCToStack(
    "disable-gc2stack", cl::ZeroOrMore,
    cl::desc("Disable promotion of GC allocations to stack memory"));
#endif
/// LDC LICENSE END

cl::opt<bool> stripDebug(
    "strip-debug", cl::ZeroOrMore,
    cl::desc("Strip symbolic debug information before optimization"));

cl::opt<bool> disableLoopUnrolling(
    "disable-loop-unrolling", cl::ZeroOrMore,
    cl::desc("Disable loop unrolling in all relevant passes"));
cl::opt<bool>
    disableLoopVectorization("disable-loop-vectorization", cl::ZeroOrMore,
                             cl::desc("Disable the loop vectorization pass"));

cl::opt<bool>
    disableSLPVectorization("disable-slp-vectorization", cl::ZeroOrMore,
                            cl::desc("Disable the slp vectorization pass"));

/// LDC LICENSE START
#ifdef LDC_DYNAMIC_COMPILE_USE_CUSTOM_PASSES
void addPass(llvm::PassManagerBase &pm, llvm::Pass *pass) {
  pm.add(pass);

  if (verifyEach) {
    pm.add(llvm::createVerifierPass());
  }
}

void addStripExternalsPass(const llvm::PassManagerBuilder &builder,
                           llvm::PassManagerBase &pm) {
  if (builder.OptLevel >= 1) {
    addPass(pm, createStripExternalsPass());
    addPass(pm, llvm::createGlobalDCEPass());
  }
}

void addSimplifyDRuntimeCallsPass(const llvm::PassManagerBuilder &builder,
                                  llvm::PassManagerBase &pm) {
  if (builder.OptLevel >= 2 && builder.SizeLevel == 0) {
    addPass(pm, createSimplifyDRuntimeCalls());
  }
}

void addGarbageCollect2StackPass(const llvm::PassManagerBuilder &builder,
                                 llvm::PassManagerBase &pm) {
  if (builder.OptLevel >= 2 && builder.SizeLevel == 0) {
    addPass(pm, createGarbageCollect2Stack());
  }
}
#endif
/// LDC LICENSE END

// TODO: share this function with compiler
void addOptimizationPasses(llvm::legacy::PassManagerBase &mpm,
                           llvm::legacy::FunctionPassManager &fpm,
                           unsigned optLevel, unsigned sizeLevel) {
  llvm::PassManagerBuilder builder;
  builder.OptLevel = optLevel;
  builder.SizeLevel = sizeLevel;

  // TODO: expose this option from jit
  if (/*willInline()*/ true) {
    auto params = llvm::getInlineParams(optLevel, sizeLevel);
    builder.Inliner = llvm::createFunctionInliningPass(params);
  } else {
    builder.Inliner = llvm::createAlwaysInlinerLegacyPass();
  }

  builder.DisableUnrollLoops = (disableLoopUnrolling.getNumOccurrences() > 0)
                                   ? disableLoopUnrolling
                                   : optLevel == 0;

  if (disableLoopVectorization) {
    builder.LoopVectorize = false;
    // If option wasn't forced via cmd line (-vectorize-loops, -loop-vectorize)
  } else if (!builder.LoopVectorize) {
    builder.LoopVectorize = optLevel > 1 && sizeLevel < 2;
  }

  builder.SLPVectorize =
      disableSLPVectorization ? false : optLevel > 1 && sizeLevel < 2;

  // TODO: sanitizers support in jit?
  // TODO: PGO support in jit?

  /// LDC LICENSE START
#ifdef LDC_DYNAMIC_COMPILE_USE_CUSTOM_PASSES
  if (!disableLangSpecificPasses) {
    if (!disableSimplifyDruntimeCalls) {
      builder.addExtension(llvm::PassManagerBuilder::EP_LoopOptimizerEnd,
                           addSimplifyDRuntimeCallsPass);
    }

    if (!disableGCToStack) {
      builder.addExtension(llvm::PassManagerBuilder::EP_LoopOptimizerEnd,
                           addGarbageCollect2StackPass);
    }
  }

  // EP_OptimizerLast does not exist in LLVM 3.0, add it manually below.
  builder.addExtension(llvm::PassManagerBuilder::EP_OptimizerLast,
                       addStripExternalsPass);
#endif
  /// LDC LICENSE END

  builder.populateFunctionPassManager(fpm);
  builder.populateModulePassManager(mpm);
}

void setupPasses(llvm::TargetMachine &targetMachine,
                 const OptimizerSettings &settings,
                 llvm::legacy::PassManager &mpm,
                 llvm::legacy::FunctionPassManager &fpm) {
  mpm.add(
      new llvm::TargetLibraryInfoWrapperPass(targetMachine.getTargetTriple()));
  mpm.add(llvm::createTargetTransformInfoWrapperPass(
      targetMachine.getTargetIRAnalysis()));
  fpm.add(llvm::createTargetTransformInfoWrapperPass(
      targetMachine.getTargetIRAnalysis()));

  if (stripDebug) {
    mpm.add(llvm::createStripSymbolsPass(true));
  }
  mpm.add(llvm::createStripDeadPrototypesPass());
  mpm.add(llvm::createStripDeadDebugInfoPass());

  addOptimizationPasses(mpm, fpm, settings.optLevel, settings.sizeLevel);
}

struct FuncFinalizer final {
  llvm::legacy::FunctionPassManager &fpm;
  explicit FuncFinalizer(llvm::legacy::FunctionPassManager &_fpm) : fpm(_fpm) {
    fpm.doInitialization();
  }
  ~FuncFinalizer() { fpm.doFinalization(); }
};

void stripComdat(llvm::Module &module) {
  for (auto &&func : module.functions()) {
    func.setComdat(nullptr);
  }
  for (auto &&var : module.globals()) {
    var.setComdat(nullptr);
  }
  module.getComdatSymbolTable().clear();
}

} // anon namespace

void optimizeModule(const Context &context, llvm::TargetMachine &targetMachine,
                    const OptimizerSettings &settings, llvm::Module &module) {
  // There is llvm bug related tp comdat and IR based pgo
  // and anyway comdat is useless at this stage
  stripComdat(module);
  llvm::legacy::PassManager mpm;
  llvm::legacy::FunctionPassManager fpm(&module);
  const auto name = module.getName();
  interruptPoint(context, "Setup passes for module", name.data());
  setupPasses(targetMachine, settings, mpm, fpm);

  // Run per-function passes.
  {
    FuncFinalizer finalizer(fpm);
    for (auto &fun : module) {
      if (fun.isDeclaration()) {
        interruptPoint(context, "Func decl", fun.getName().data());
      } else {
        interruptPoint(context, "Run passes for function",
                       fun.getName().data());
      }
      fpm.run(fun);
    }
  }

  // Run per-module passes.
  interruptPoint(context, "Run passes for module", name.data());
  mpm.run(module);
}

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
