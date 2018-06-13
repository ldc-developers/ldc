//===-- optimizer.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "optimizer.h"

#include "llvm/Target/TargetMachine.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include "llvm/ADT/Triple.h"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"

#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "context.h"
#include "utils.h"
#include "valueparser.h"

namespace {
// TODO: share this function with compiler
void addOptimizationPasses(llvm::legacy::PassManagerBase &mpm,
                           llvm::legacy::FunctionPassManager &fpm,
                           unsigned optLevel, unsigned sizeLevel) {
  llvm::PassManagerBuilder builder;
  builder.OptLevel = optLevel;
  builder.SizeLevel = sizeLevel;

  // TODO: expose this option from jit
  if (/*willInline()*/ true) {
#if LDC_LLVM_VER >= 400
    auto params = llvm::getInlineParams(optLevel, sizeLevel);
    builder.Inliner = llvm::createFunctionInliningPass(params);
#else
    builder.Inliner = llvm::createFunctionInliningPass(optLevel, sizeLevel);
#endif
  } else {
#if LDC_LLVM_VER >= 400
    builder.Inliner = llvm::createAlwaysInlinerLegacyPass();
#else
    builder.Inliner = llvm::createAlwaysInlinerPass();
#endif
  }
  builder.DisableUnitAtATime = false;

  // TODO: Expose this option
  builder.DisableUnrollLoops = optLevel == 0;

  // TODO: expose this option
  if (/*disableLoopVectorization*/ false) {
    builder.LoopVectorize = false;
    // If option wasn't forced via cmd line (-vectorize-loops, -loop-vectorize)
  } else if (!builder.LoopVectorize) {
    builder.LoopVectorize = optLevel > 1 && sizeLevel < 2;
  }

  // TODO: expose this option
  builder.SLPVectorize =
      /*disableSLPVectorization*/ false ? false : optLevel > 1 && sizeLevel < 2;

  // TODO: sanitizers support in jit?
  // TODO: lang specific passes support
  // TODO: addStripExternalsPass?
  // TODO: PGO support in jit?

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

  if (/*stripDebug*/ true) {
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
    auto type = var->getType()->getElementType();
    auto initializer =
        parseInitializer(module.getDataLayout(), *type, init,
                         [&](const std::string &str) { fatal(context, str); });
    var->setConstant(true);
    var->setInitializer(initializer);
    var->setLinkage(llvm::GlobalValue::PrivateLinkage);
  }
}
