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
#include "ir/irfuncty.h"
#include "gen/dcompute/abi-rewrites.h"
#include "mtype.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>


using namespace dmd;

struct MetalABI : TargetABI {
    DComputePointerRewrite pointerRewite;

    auto returnInArg(TypeFunction *tf, bool needsThis) -> bool override {
        return false;
    }

    auto passByVal(TypeFunction *tf, Type*t) -> bool override {
        return false;
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
    }
};

auto createMetalABI() -> TargetABI* { return new MetalABI(); }
