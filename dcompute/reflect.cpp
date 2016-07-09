//===-- reflect.cpp ---------------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
#include "dcompute/reflect.h"

using namespace llvm;
namespace {
class DComputeReflect : ModulePass
{
  static char ID;
  int target;
  unsigned version;
  DComputeReflect(int _target, unsigned _version) : ModulePass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
  }
  bool runOnModule(Module &) override;


}

bool DComputeReflect::runOnModule(Module& m)
{
  Function *ReflectFunction = M.getFunction("__dcompute_reflect");
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
    assert((Reflect->getNumOperands() == 2) &&
           "Only two operands expect for __dcompute_reflect function");
    const Value *targ = Reflect->getArgOperand(0);
    const Value *vers = Reflect->getArgOperand(1);
    assert(isa<ConstantInt>(targ) && isa<ConstantInt>(vers) &&
           "arguments to __dcompute_reflect must be Constant integer types");
    ConstantInt *ctarg = cast<ConstantInt>(targ);
    ConstantInt *cvers = cast<ConstantInt>(vers);
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
    return new DComputeReflect(target,version);
}
