//===-- gen/abi-metal.cpp ---------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/identifier.h"
#include "dmd/nspace.h"
#include "gen/abi/abi.h"
#include "gen/abi/generic.h"
#include "gen/dcompute/druntime.h"
#include "gen/uda.h"
#include "ir/irfuncty.h"
#include "gen/dcompute/abi-rewrites.h"
#include "mtype.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>


using namespace dmd;

struct MetalABI : TargetABI {
    DComputePointerRewrite pointerRewite;
    DcomputeMetalScalarRewrite metalScalarRewrite;

    auto returnInArg(TypeFunction *tf, bool needsThis) -> bool override {
        return false;
    }

    auto passByVal(TypeFunction *tf, Type*t) -> bool override {
        return false;
    }

    void rewriteFunctionType(IrFuncTy &fty) override {
      for (auto arg : fty.args) {
        if (!arg->byref) {
          rewriteArgument(fty, *arg);
        }
      }
    }

    void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
        TargetABI::rewriteArgument(fty, arg);

        if (arg.rewrite) {
            return;
        }

        Type *ty = arg.type->toBasetype();
        std::optional<DcomputePointer> ptr;

        if (ty->ty == TY::Tstruct &&
            (ptr = toDcomputePointer(static_cast<TypeStruct *>(ty)->sym))) {
                pointerRewite.applyTo(arg);
            }

        if (ty->isScalar()) {
          llvm::errs() << "Applying Metal Scalar Rewrite to: " << ty->toChars() << "\n";
          metalScalarRewrite.applyTo(arg);
        }
    }
};

auto createMetalABI() -> TargetABI* { return new MetalABI(); }
