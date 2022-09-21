//===-- gen/abi-spirv.cpp ---------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/abi.h"
#include "gen/dcompute/druntime.h"
#include "gen/uda.h"
#include "dmd/declaration.h"
#include "gen/tollvm.h"
#include "gen/dcompute/abi-rewrites.h"

struct DirectXTargetABI : TargetABI {

  llvm::CallingConv::ID callingConv(FuncDeclaration *fdecl) override {
    return llvm::CallingConv::C;
  }
  bool passByVal(TypeFunction *, Type *t) override {
    t = t->toBasetype();
    return ((t->ty == TY::Tsarray || t->ty == TY::Tstruct) && t->size() > 64);
  }
  void rewriteFunctionType(IrFuncTy &fty) override { }
  bool returnInArg(TypeFunction *tf, bool) override {
    if (tf->isref())
      return false;
    Type *rt = tf->next->toBasetype();

    if (!isPOD(rt))
      return true;

    // Return structs and static arrays on the stack. The latter is needed
    // because otherwise LLVM tries to actually return the array in a number
    // of physical registers, which leads, depending on the target, to
    // either horrendous codegen or backend crashes.
    return (rt->ty == TY::Tstruct || rt->ty == TY::Tsarray);
  }
  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override { }
  // There are no exceptions at all, so no need for unwind tables.
  bool needsUnwindTables() override {
    return false;
  }
};

TargetABI *createDirectXABI() { return new DirectXTargetABI(); }
