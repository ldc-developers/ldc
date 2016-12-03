#include "optimizer.h"

#include "llvm/Target/TargetMachine.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Verifier.h"

#include "llvm/ADT/Triple.h"

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "context.h"
#include "utils.h"
#include "valueparser.h"

namespace {

void addOptimizationPasses(llvm::legacy::PassManagerBase &mpm,
                           llvm::legacy::FunctionPassManager &fpm,
                           unsigned optLevel, unsigned sizeLevel) {
//  if (!noVerify) {
//    fpm.add(createVerifierPass());
//  }

  llvm::PassManagerBuilder builder;
  builder.OptLevel = optLevel;
  builder.SizeLevel = sizeLevel;

  if (/*willInline()*/true) {
    unsigned threshold = 225;
    if (sizeLevel == 1) { // -Os
      threshold = 75;
    } else if (sizeLevel == 2) { // -Oz
      threshold = 25;
    }
    if (optLevel > 2) {
      threshold = 275;
    }
    builder.Inliner = llvm::createFunctionInliningPass(threshold);
  } else {
    builder.Inliner = llvm::createAlwaysInlinerPass();
  }
//  builder.DisableUnitAtATime = !unitAtATime;
  builder.DisableUnrollLoops = optLevel == 0;

//  builder.DisableUnrollLoops = (disableLoopUnrolling.getNumOccurrences() > 0)
//                                   ? disableLoopUnrolling
//                                   : optLevel == 0;

  // This is final, unless there is a #pragma vectorize enable
  if (/*disableLoopVectorization*/false) {
    builder.LoopVectorize = false;
    // If option wasn't forced via cmd line (-vectorize-loops, -loop-vectorize)
  } else if (!builder.LoopVectorize) {
    builder.LoopVectorize = optLevel > 1 && sizeLevel < 2;
  }

  // When #pragma vectorize is on for SLP, do the same as above
  builder.SLPVectorize =
      /*disableSLPVectorization*/false ? false : optLevel > 1 && sizeLevel < 2;

//  if (opts::sanitize == opts::AddressSanitizer) {
//    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
//                         addAddressSanitizerPasses);
//    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
//                         addAddressSanitizerPasses);
//  }

//  if (opts::sanitize == opts::MemorySanitizer) {
//    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
//                         addMemorySanitizerPass);
//    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
//                         addMemorySanitizerPass);
//  }

//  if (opts::sanitize == opts::ThreadSanitizer) {
//    builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
//                         addThreadSanitizerPass);
//    builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
//                         addThreadSanitizerPass);
//  }

//  if (!disableLangSpecificPasses) {
//    if (!disableSimplifyDruntimeCalls) {
//      builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd,
//                           addSimplifyDRuntimeCallsPass);
//    }

//    if (!disableGCToStack) {
//      builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd,
//                           addGarbageCollect2StackPass);
//    }
//  }

  // EP_OptimizerLast does not exist in LLVM 3.0, add it manually below.
//  builder.addExtension(llvm::PassManagerBuilder::EP_OptimizerLast,
//                       addStripExternalsPass);

//  addInstrProfilingPass(mpm);

  builder.populateFunctionPassManager(fpm);
  builder.populateModulePassManager(mpm);
}

void setupPasses(llvm::TargetMachine &targetMachine,
                 const OptimizerSettings& settings,
                 llvm::legacy::PassManager &mpm,
                 llvm::legacy::FunctionPassManager &fpm) {
  mpm.add(new llvm::TargetLibraryInfoWrapperPass(
            targetMachine.getTargetTriple()));
  mpm.add(llvm::createTargetTransformInfoWrapperPass(
            targetMachine.getTargetIRAnalysis()));
  fpm.add(llvm::createTargetTransformInfoWrapperPass(
            targetMachine.getTargetIRAnalysis()));

  if (/*stripDebug*/true) {
    mpm.add(llvm::createStripSymbolsPass(true));
  }
  mpm.add(llvm::createStripDeadPrototypesPass());
  mpm.add(llvm::createStripDeadDebugInfoPass());

  addOptimizationPasses(mpm, fpm, settings.optLevel, settings.sizeLeve);
}

struct FuncFinalizer final {
  llvm::legacy::FunctionPassManager& fpm;
  explicit FuncFinalizer(llvm::legacy::FunctionPassManager& _fpm):
    fpm(_fpm) {
    fpm.doInitialization();
  }
  ~FuncFinalizer() {
    fpm.doFinalization();
  }

};

} // anon namespace

void optimizeModule(const Context &context,
                    llvm::TargetMachine &targetMachine,
                    const OptimizerSettings &settings,
                    llvm::Module &module)
{
  llvm::legacy::PassManager mpm;
  llvm::legacy::FunctionPassManager fpm(&module);
  const auto name = module.getName();
  interruptPoint(context, "Setup passes for module", name.data());
  setupPasses(targetMachine, settings, mpm, fpm);

  // Run per-function passes.
  {
    FuncFinalizer finalizer(fpm);
    for (auto &fun : module) {
      interruptPoint(context, "Run passes for function", fun.getName().data());
      fpm.run(fun);
    }
  }

  // Run per-module passes.
  interruptPoint(context, "Run passes for module", name.data());
  mpm.run(module);
}

void setRtCompileVar(const Context &context,
                     llvm::Module &module,
                     const char *name,
                     const void *init) {
  assert(nullptr != name);
  assert(nullptr != init);
  auto var = module.getGlobalVariable(name);
  if (nullptr != var) {
    auto type = var->getType()->getElementType();
    auto initializer = parseInitializer(context,
                                        module.getDataLayout(),
                                        type,
                                        init);
    var->setConstant(true);
    var->setInitializer(initializer);
    var->setLinkage(llvm::GlobalValue::PrivateLinkage);
//    auto tempVar = new llvm::GlobalVariable(
//                     module,
//                     type,
//                     true,
//                     llvm::GlobalValue::PrivateLinkage,
//                     initializer,
//                     ".str");
//    llvm::Constant *idxs[] = {zero};
//    auto constPtr = llvm::ConstantExpr::getGetElementPtr(nullptr,
//                                                         tempVar,
//                                                         idxs,
//                                                         true);
//    for (auto&& use: var->uses()) {
//      use->dump();
//      use->getType()->dump();
//      auto i = llvm::cast<llvm::GlobalVariable>(use);
//      i->replaceAllUsesWith(constPtr);
//      i->eraseFromParent();
//    }

//          var->replaceAllUsesWith(initializer);
//          var->eraseFromParent();
  }
}
