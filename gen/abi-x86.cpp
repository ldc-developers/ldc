//===-- abi-x86.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "mars.h"
#include "gen/abi-generic.h"
#include "gen/abi.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irfuncty.h"


struct X86TargetABI : TargetABI
{
    CompositeToInt compositeToInt;

    llvm::CallingConv::ID callingConv(LINK l)
    {
        switch (l)
        {
        case LINKc:
        case LINKcpp:
            return llvm::CallingConv::C;
        case LINKd:
        case LINKdefault:
        case LINKpascal:
        case LINKwindows:
            return llvm::CallingConv::X86_StdCall;
        default:
            llvm_unreachable("Unhandled D linkage type.");
        }
    }

    std::string mangleForLLVM(llvm::StringRef name, LINK l)
    {
        switch (l)
        {
        case LINKc:
        case LINKcpp:
        case LINKpascal:
        case LINKwindows:
            return name;
        case LINKd:
        case LINKdefault:
            if (global.params.targetTriple.isOSWindows())
            {
                // Prepend a 0x1 byte to keep LLVM from adding the usual
                // "@<paramsize>" stdcall suffix.
                return ("\1_" + name).str();
            }
            return name;
        default:
            llvm_unreachable("Unhandled D linkage type.");
        }
    }

    bool returnInArg(TypeFunction* tf)
    {
        if (tf->isref)
            return false;

        Type* rt = tf->next->toBasetype();
        // D only returns structs on the stack
        if (tf->linkage == LINKd)
        {
            return rt->ty == Tstruct
                || rt->ty == Tsarray
            ;
        }
        // other ABI's follow C, which is cdouble and creal returned on the stack
        // as well as structs
        else
            return (rt->ty == Tstruct || rt->ty == Tcomplex64 || rt->ty == Tcomplex80);
    }

    bool passByVal(Type* t)
    {
        return t->toBasetype()->ty == Tstruct || t->toBasetype()->ty == Tsarray;
    }

    void rewriteFunctionType(TypeFunction* tf, IrFuncTy &fty)
    {
        Type* rt = fty.ret->type->toBasetype();

        // extern(D)
        if (tf->linkage == LINKd)
        {
            // IMPLICIT PARAMETERS

            // mark this/nested params inreg
            if (fty.arg_this)
            {
                Logger::println("Putting 'this' in register");
                fty.arg_this->attrs.clear()
                                   .add(LDC_ATTRIBUTE(InReg));
            }
            else if (fty.arg_nest)
            {
                Logger::println("Putting context ptr in register");
                fty.arg_nest->attrs.clear()
                                   .add(LDC_ATTRIBUTE(InReg));
            }
            else if (IrFuncTyArg* sret = fty.arg_sret)
            {
                Logger::println("Putting sret ptr in register");
                // sret and inreg are incompatible, but the ABI requires the
                // sret parameter to be in EAX in this situation...
                sret->attrs.add(LDC_ATTRIBUTE(InReg)).remove(LDC_ATTRIBUTE(StructRet));
            }
            // otherwise try to mark the last param inreg
            else if (!fty.args.empty())
            {
                // The last parameter is passed in EAX rather than being pushed on the stack if the following conditions are met:
                //   * It fits in EAX.
                //   * It is not a 3 byte struct.
                //   * It is not a floating point type.

                IrFuncTyArg* last = fty.args.back();
                Type* lastTy = last->type->toBasetype();
                unsigned sz = lastTy->size();

                if (last->byref && !last->isByVal())
                {
                    Logger::println("Putting last (byref) parameter in register");
                    last->attrs.add(LDC_ATTRIBUTE(InReg));
                }
                else if (!lastTy->isfloating() && (sz == 1 || sz == 2 || sz == 4)) // right?
                {
                    // rewrite the struct into an integer to make inreg work
                    if (lastTy->ty == Tstruct || lastTy->ty == Tsarray)
                    {
                        last->rewrite = &compositeToInt;
                        last->ltype = compositeToInt.type(last->type, last->ltype);
                        last->byref = false;
                        // erase previous attributes
                        last->attrs.clear();
                    }
                    last->attrs.add(LDC_ATTRIBUTE(InReg));
                }
            }

            // FIXME: tf->varargs == 1 need to use C calling convention and vararg mechanism to live up to the spec:
            // "The caller is expected to clean the stack. _argptr is not passed, it is computed by the callee."

            // EXPLICIT PARAMETERS

            // reverse parameter order
            // for non variadics
            if (!fty.args.empty() && tf->varargs != 1)
            {
                fty.reverseParams = true;
            }
        }

        // extern(C) and all others
        else
        {
            // RETURN VALUE

            // cfloat -> i64
            if (tf->next->toBasetype() == Type::tcomplex32)
            {
                fty.ret->rewrite = &compositeToInt;
                fty.ret->ltype = compositeToInt.type(fty.ret->type, fty.ret->ltype);
            }

            // IMPLICIT PARAMETERS

            // EXPLICIT PARAMETERS
        }
    }
};

// The public getter for abi.cpp.
TargetABI* getX86TargetABI() {
    return new X86TargetABI;
}
