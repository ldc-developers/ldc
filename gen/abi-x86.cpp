//===-- abi-x86.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "id.h"
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

struct X86TargetABI : TargetABI {
  const bool isOSX;
  const bool isMSVC;
  IntegerRewrite integerRewrite;

  X86TargetABI()
      : isOSX(global.params.isOSX),
        isMSVC(global.params.targetTriple.isWindowsMSVCEnvironment()) {}

  llvm::CallingConv::ID callingConv(llvm::FunctionType *ft, LINK l) override {
    switch (l) {
    case LINKc:
    case LINKcpp:
      return llvm::CallingConv::C;
    case LINKd:
    case LINKdefault:
    case LINKpascal:
    case LINKwindows:
      return ft->isVarArg() ? llvm::CallingConv::C
                            : llvm::CallingConv::X86_StdCall;
    default:
      llvm_unreachable("Unhandled D linkage type.");
    }
  }

  std::string mangleForLLVM(llvm::StringRef name, LINK l) override {
    switch (l) {
    case LINKc:
    case LINKcpp:
    case LINKpascal:
    case LINKwindows:
      return name;
    case LINKd:
    case LINKdefault:
      if (global.params.targetTriple.isOSWindows()) {
        // Prepend a 0x1 byte to keep LLVM from adding the usual
        // "@<paramsize>" stdcall suffix.
        return ("\1_" + name).str();
      }
      return name;
    default:
      llvm_unreachable("Unhandled D linkage type.");
    }
  }

  bool returnInArg(TypeFunction *tf) override {
    if (tf->isref)
      return false;

    Type *rt = tf->next->toBasetype();

    // non-aggregates are returned directly
    if (!isAggregate(rt))
      return false;

    // extern(D): sret exclusively for all structs and static arrays
    if (tf->linkage == LINKd && tf->varargs != 1)
      return rt->ty == Tstruct || rt->ty == Tsarray;

    // extern(C) and all others:

    // special cases for structs
    if (rt->ty == Tstruct) {
      // no sret for magic C++ structs
      if (isMagicCppStruct(rt))
        return false;

      // force sret for non-POD structs
      if (!isPOD(rt, tf->linkage))
        return true;
    }

    if (isOSX || isMSVC) {
      // no sret for remaining aggregates of a power-of-2 size <= 8 bytes
      return !canRewriteAsInt(rt);
    }

    return true;
  }

  bool passByVal(Type *t) override {
    // pass all structs and static arrays with the LLVM byval attribute
    return t->toBasetype()->ty == Tstruct || t->toBasetype()->ty == Tsarray;
  }

  void rewriteFunctionType(TypeFunction *tf, IrFuncTy &fty) override {
    // return value:
    if (!fty.ret->byref) {
      Type *rt = tf->next->toBasetype(); // for sret, rt == void
      if (isAggregate(rt) && !isMagicCppStruct(rt) && canRewriteAsInt(rt) &&
          !integerRewrite.isObsoleteFor(fty.ret->ltype)) {
        fty.ret->rewrite = &integerRewrite;
        fty.ret->ltype = integerRewrite.type(fty.ret->type, fty.ret->ltype);
      }
    }

    // extern(D):
    if (tf->linkage == LINKd && tf->varargs != 1) {

      // try to pass an implicit argument in a register...
      if (fty.arg_this) {
        Logger::println("Putting 'this' in register");
        fty.arg_this->attrs.clear().add(LLAttribute::InReg);
      } else if (fty.arg_nest) {
        Logger::println("Putting context ptr in register");
        fty.arg_nest->attrs.clear().add(LLAttribute::InReg);
      } else if (IrFuncTyArg *sret = fty.arg_sret) {
        Logger::println("Putting sret ptr in register");
        // sret and inreg are incompatible, but the ABI requires the
        // sret parameter to be in EAX in this situation...
        sret->attrs.add(LLAttribute::InReg).remove(LLAttribute::StructRet);
      }

      // ... otherwise try the last argument
      else if (!fty.args.empty()) {
        // The last parameter is passed in EAX rather than being pushed on the
        // stack if the following conditions are met:
        //   * It fits in EAX.
        //   * It is not a 3 byte struct.
        //   * It is not a floating point type.

        IrFuncTyArg *last = fty.args.back();
        Type *lastTy = last->type->toBasetype();
        unsigned sz = lastTy->size();

        if (last->byref) {
          if (!last->isByVal()) {
            Logger::println("Putting last (byref) parameter in register");
            last->attrs.add(LLAttribute::InReg);
          }
        } else if (!lastTy->isfloating() && (sz == 1 || sz == 2 || sz == 4)) {
          // may have to rewrite the aggregate as integer to make inreg work
          if ((lastTy->ty == Tstruct || lastTy->ty == Tsarray) &&
              !integerRewrite.isObsoleteFor(last->ltype)) {
            last->rewrite = &integerRewrite;
            last->ltype = integerRewrite.type(last->type, last->ltype);
            last->byref = false;
            // erase previous attributes
            last->attrs.clear();
          }
          last->attrs.add(LLAttribute::InReg);
        }
      }

      // all other arguments are passed on the stack, don't rewrite

      // reverse parameter order
      if (!fty.args.empty()) {
        fty.reverseParams = true;
      }

      return;
    }

    // extern(C) and all others:

    // Clang does not pass empty structs, while it seems that GCC does,
    // at least on Linux x86. We don't know whether the C compiler will
    // be Clang or GCC, so just assume Clang on OS X and G++ on Linux.
    if (!isOSX) {
      return;
    }

    size_t i = 0;
    while (i < fty.args.size()) {
      Type *type = fty.args[i]->type->toBasetype();
      if (type->ty == Tstruct) {
        // Do not pass empty structs at all for C++ ABI compatibility.
        // Tests with clang reveal that more complex "empty" types, for
        // example a struct containing an empty struct, are not
        // optimized in the same way.
        auto sd = static_cast<TypeStruct *>(type)->sym;
        if (sd->fields.empty()) {
          fty.args.erase(fty.args.begin() + i);
          continue;
        }
      }
      ++i;
    }
  }
};

// The public getter for abi.cpp.
TargetABI *getX86TargetABI() { return new X86TargetABI; }
