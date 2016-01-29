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

struct X86TargetABI : TargetABI {
  const bool isOSX;
  const bool isMSVC;
  bool returnStructsInRegs;
  IntegerRewrite integerRewrite;

  X86TargetABI()
      : isOSX(global.params.targetTriple.isMacOSX()),
        isMSVC(global.params.targetTriple.isWindowsMSVCEnvironment()) {
    using llvm::Triple;
    auto os = global.params.targetTriple.getOS();
    returnStructsInRegs =
        !(os == Triple::Linux || os == Triple::Solaris || os == Triple::NetBSD);
  }

  llvm::CallingConv::ID callingConv(llvm::FunctionType *ft, LINK l,
                                    FuncDeclaration *fdecl = nullptr) override {
    switch (l) {
    case LINKc:
      return llvm::CallingConv::C;
    case LINKcpp:
      return isMSVC && !ft->isVarArg() && fdecl && fdecl->isThis()
                 ? llvm::CallingConv::X86_ThisCall
                 : llvm::CallingConv::C;
    case LINKd:
    case LINKdefault:
    case LINKpascal:
    case LINKwindows:
      return ft->isVarArg() ? llvm::CallingConv::C
                            : llvm::CallingConv::X86_StdCall;
    }
    llvm_unreachable("Unhandled D linkage type.");
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
    }
    llvm_unreachable("Unhandled D linkage type.");
  }

  bool returnInArg(TypeFunction *tf) override {
    if (tf->isref)
      return false;

    Type *rt = tf->next->toBasetype();
    const bool externD = (tf->linkage == LINKd && tf->varargs != 1);

    // non-aggregates and magic C++ structs are returned directly
    if (!isAggregate(rt) || isMagicCppStruct(rt))
      return false;

    // complex numbers
    if (rt->iscomplex()) {
      // extern(D): let LLVM return them directly as LL aggregates
      if (externD)
        return false;
      // extern(C) and all others:
      // * cfloat will be rewritten as 64-bit integer and returned in registers
      // * sret for cdouble and creal
      return rt->ty != Tcomplex32;
    }

    // non-extern(D): some OSs don't return structs in registers at all
    if (!externD && !returnStructsInRegs)
      return true;

    // force sret for non-POD structs
    const bool excludeStructsWithCtor = (isMSVC && tf->linkage == LINKcpp);
    if (!isPOD(rt, excludeStructsWithCtor))
      return true;

    // return aggregates of a power-of-2 size <= 8 bytes in register(s),
    // all others via sret
    return !canRewriteAsInt(rt);
  }

  bool passByVal(Type *t) override {
    // pass all structs and static arrays with the LLVM byval attribute
    return DtoIsInMemoryOnly(t);
  }

  void rewriteFunctionType(TypeFunction *tf, IrFuncTy &fty) override {
    const bool externD = (tf->linkage == LINKd && tf->varargs != 1);

    // return value:
    if (!fty.ret->byref) {
      Type *rt = tf->next->toBasetype(); // for sret, rt == void
      if (isAggregate(rt) && !isMagicCppStruct(rt) && canRewriteAsInt(rt) &&
          // don't rewrite cfloat for extern(D)
          !(externD && rt->ty == Tcomplex32) &&
          !integerRewrite.isObsoleteFor(fty.ret->ltype)) {
        fty.ret->rewrite = &integerRewrite;
        fty.ret->ltype = integerRewrite.type(fty.ret->type, fty.ret->ltype);
      }
    }

    // extern(D): try passing an argument in EAX
    if (externD) {

      // try an implicit argument...
      if (fty.arg_this) {
        Logger::println("Putting 'this' in register");
        fty.arg_this->attrs.add(LLAttribute::InReg);
      } else if (fty.arg_nest) {
        Logger::println("Putting context ptr in register");
        fty.arg_nest->attrs.add(LLAttribute::InReg);
      } else if (IrFuncTyArg *sret = fty.arg_sret) {
        Logger::println("Putting sret ptr in register");
        // sret and inreg are incompatible, but the ABI requires the
        // sret parameter to be in EAX in this situation...
        sret->attrs.remove(LLAttribute::StructRet).add(LLAttribute::InReg);
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

        if (last->byref && !last->isByVal()) {
          Logger::println("Putting last (byref) parameter in register");
          last->attrs.add(LLAttribute::InReg);
        } else if (!lastTy->isfloating() && (sz == 1 || sz == 2 || sz == 4)) {
          // rewrite aggregates as integers to make inreg work
          if (lastTy->ty == Tstruct || lastTy->ty == Tsarray) {
            last->rewrite = &integerRewrite;
            last->ltype = integerRewrite.type(last->type, last->ltype);
            // undo byval semantics applied via passByVal() returning true
            last->byref = false;
            last->attrs.clear();
          }
          last->attrs.add(LLAttribute::InReg);
        }
      }

      // all other arguments are passed on the stack, don't rewrite

      // reverse parameter order
      if (fty.args.size() > 1) {
        fty.reverseParams = true;
      }
    }

    // FIXME: byval params with alignment attributes lead to crashes with MSVC
    if (isMSVC) {
      for (auto arg : fty.args) {
        if (arg->isByVal()) {
          arg->attrs.remove(LLAttribute::Alignment);
        }
      }
    }

    // Clang does not pass empty structs, while it seems that GCC does,
    // at least on Linux x86. We don't know whether the C compiler will
    // be Clang or GCC, so just assume Clang on OS X and G++ on Linux.
    if (externD || !isOSX)
      return;

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
