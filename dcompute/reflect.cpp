//===-- dcompute/reflect.cpp -------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
#include "dcompute/reflect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "gen/logger.h"
using namespace llvm;
namespace {
class DComputeReflect : public ModulePass
{
  static char ID;
  int _target;
  unsigned _version;
public:
  DComputeReflect(int __target, unsigned __version) : ModulePass(ID) {
      _target = __target;
      _version = __version;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
  }
  bool runOnModule(llvm::Module &) override;


};

bool DComputeReflect::runOnModule(llvm::Module& m)
{
    IF_LOG Logger::println("DComputeReflect::runOnModule. _target = %d,_version = %d",_target,_version);
  Function *ReflectFunction = m.getFunction("__dcompute_reflect");
  if (!ReflectFunction)
      return false;
  // Validate _reflect function
  assert(ReflectFunction->isDeclaration() &&
         "__dcompute_reflect function should not have a body");
  assert(ReflectFunction->getReturnType()->isIntegerTy() &&
         "__dcompute_reflect's return type should be bool");
  
  std::vector<Instruction *> ToRemove;
  for (User *U : ReflectFunction->users())
  {
    assert(isa<CallInst>(U) && "Only a call instruction can use __dcompute_reflect");
    CallInst *Reflect = cast<CallInst>(U);
    const Value *targ = Reflect->getArgOperand(0);
    const Value *vers = Reflect->getArgOperand(1);
    assert(isa<ConstantInt>(targ) && isa<ConstantInt>(vers) &&
           "arguments to __dcompute_reflect must be Constant integer types");
    const ConstantInt *ctarg = cast<ConstantInt>(targ);
    const ConstantInt *cvers = cast<ConstantInt>(vers);
    if (ctarg->equalsInt(_target) && (
        cvers->equalsInt(_version) || cvers->isZero()))
    {
        Reflect->replaceAllUsesWith(ConstantInt::get(Reflect->getType(),1));
    }
    else {
        Reflect->replaceAllUsesWith(ConstantInt::get(Reflect->getType(),0));
    }
    ToRemove.push_back(Reflect);
  }
  if (ToRemove.size() == 0)
    return false;
  for (unsigned i = 0, e = ToRemove.size(); i != e; ++i)
    ToRemove[i]->eraseFromParent();
  return true;

}

char DComputeReflect::ID = 0;
};

ModulePass *createDComputeReflectPass(int target, unsigned version)
{
    IF_LOG Logger::println("DComputeReflect create pass target %d, version %d", target,version);
    return new DComputeReflect(target,version);
}
