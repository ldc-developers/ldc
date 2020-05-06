//===-- abi-x86.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/abi-generic.h"
#include "gen/abi.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
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
  IndirectByvalRewrite indirectByvalRewrite;

  X86TargetABI()
      : isOSX(global.params.targetTriple->isMacOSX()),
        isMSVC(global.params.targetTriple->isWindowsMSVCEnvironment()) {
    using llvm::Triple;
    auto os = global.params.targetTriple->getOS();
    returnStructsInRegs =
        !(os == Triple::Linux || os == Triple::Solaris || os == Triple::NetBSD);
  }

  llvm::CallingConv::ID callingConv(LINK l, TypeFunction *tf = nullptr,
                                    FuncDeclaration *fdecl = nullptr) override {
    if (tf && tf->parameterList.varargs == VARARGvariadic)
      return llvm::CallingConv::C;

    switch (l) {
    case LINKc:
    case LINKobjc:
      return llvm::CallingConv::C;
    case LINKcpp:
      return isMSVC && fdecl && fdecl->needThis()
                 ? llvm::CallingConv::X86_ThisCall
                 : llvm::CallingConv::C;
    case LINKd:
    case LINKdefault:
    case LINKpascal:
    case LINKwindows:
      return llvm::CallingConv::X86_StdCall;
    default:
      llvm_unreachable("Unhandled D linkage type.");
    }
  }

  std::string mangleFunctionForLLVM(std::string name, LINK l) override {
    if (global.params.targetTriple->isOSWindows()) {
      if (l == LINKd || l == LINKdefault) {
        // Prepend a 0x1 byte to prevent LLVM from applying MS stdcall mangling:
        // _D… => __D…@<paramssize>, and add extra underscore manually.
        name.insert(0, "\1_");
      } else if (l == LINKcpp && name[0] == '?') {
        // Prepend a 0x1 byte to prevent LLVM from prepending the C underscore
        // for MSVC++ symbols (starting with '?').
        name.insert(0, "\1");
      }
    }
    return name;
  }

  std::string mangleVariableForLLVM(std::string name, LINK l) override {
    if (global.params.targetTriple->isOSWindows() && l == LINKcpp &&
        name[0] == '?') {
      // Prepend a 0x1 byte to prevent LLVM from prepending the C underscore for
      // MSVC++ symbols (starting with '?').
      name.insert(0, "\1");
    }
    return name;
  }

  bool returnInArg(TypeFunction *tf, bool needsThis) override {
    if (tf->isref)
      return false;

    Type *rt = tf->next->toBasetype();
    const bool externD = isExternD(tf);

    // non-aggregates are returned directly
    if (!isAggregate(rt))
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

    const bool isMSVCpp = isMSVC && tf->linkage == LINKcpp;

    // for non-static member functions, MSVC++ enforces sret for all structs
    if (isMSVCpp && needsThis && rt->ty == Tstruct) {
      return true;
    }

    // force sret for non-POD structs
    const bool excludeStructsWithCtor = isMSVCpp;
    if (!isPOD(rt, excludeStructsWithCtor))
      return true;

    // return aggregates of a power-of-2 size <= 8 bytes in register(s),
    // all others via sret
    return !canRewriteAsInt(rt);
  }

  bool passByVal(TypeFunction *tf, Type *t) override {
    // indirectly by-value for non-POD args on Posix
    if (!isMSVC && !isPOD(t))
      return false;

    // pass all structs and static arrays with the LLVM byval attribute
    return DtoIsInMemoryOnly(t);
  }

  void rewriteFunctionType(IrFuncTy &fty) override {
    const bool externD = isExternD(fty.type);

    // return value:
    if (!fty.ret->byref) {
      Type *rt = fty.ret->type->toBasetype(); // for sret, rt == void
      if (isAggregate(rt) && canRewriteAsInt(rt) &&
          // don't rewrite cfloat for extern(D)
          !(externD && rt->ty == Tcomplex32)) {
        integerRewrite.applyToIfNotObsolete(*fty.ret);
      }
    }

    // Posix: non-POD args are passed indirectly by-value
    if (!isMSVC) {
      for (auto arg : fty.args) {
        if (!arg->byref && !isPOD(arg->type))
          indirectByvalRewrite.applyTo(*arg);
      }
    }

    // extern(D): try passing an argument in EAX
    if (externD) {

      // try an implicit argument...
      if (fty.arg_this) {
        Logger::println("Putting 'this' in register");
        fty.arg_this->attrs.addAttribute(LLAttribute::InReg);
      } else if (fty.arg_nest) {
        Logger::println("Putting context ptr in register");
        fty.arg_nest->attrs.addAttribute(LLAttribute::InReg);
      } else if (IrFuncTyArg *sret = fty.arg_sret) {
        Logger::println("Putting sret ptr in register");
        // sret and inreg are incompatible, but the ABI requires the
        // sret parameter to be in EAX in this situation...
        sret->attrs.removeAttribute(LLAttribute::StructRet);
        sret->attrs.addAttribute(LLAttribute::InReg);
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

        if (last->rewrite == &indirectByvalRewrite ||
            (last->byref && !last->isByVal())) {
          Logger::println("Putting last (byref) parameter in register");
          last->attrs.addAttribute(LLAttribute::InReg);
        } else if (!lastTy->isfloating() && (sz == 1 || sz == 2 || sz == 4)) {
          // rewrite aggregates as integers to make inreg work
          if (lastTy->ty == Tstruct || lastTy->ty == Tsarray) {
            integerRewrite.applyTo(*last);
            // undo byval semantics applied via passByVal() returning true
            last->byref = false;
            last->attrs.clear();
          }
          last->attrs.addAttribute(LLAttribute::InReg);
        }
      }
    }

    workaroundIssue1356(fty.args);

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

  void rewriteVarargs(IrFuncTy &fty,
                      std::vector<IrFuncTyArg *> &args) override {
    TargetABI::rewriteVarargs(fty, args);
    workaroundIssue1356(args);
  }

  // FIXME: LDC issue #1356
  // MSVC targets don't support alignment attributes for LL byval args
  void workaroundIssue1356(std::vector<IrFuncTyArg *> &args) const {
    if (isMSVC) {
      for (auto arg : args) {
        if (arg->isByVal())
          arg->attrs.removeAttribute(LLAttribute::Alignment);
      }
    }
  }

  const char *objcMsgSendFunc(Type *ret, IrFuncTy &fty) override {
    // see objc/message.h for objc_msgSend selection rules
    assert(isOSX);
    if (fty.arg_sret) {
      return "objc_msgSend_stret";
    }
    // float, double, long double return
    if (ret && ret->isfloating() && !ret->iscomplex()) {
      return "objc_msgSend_fpret";
    }
    return "objc_msgSend";
  }
};

// The public getter for abi.cpp.
TargetABI *getX86TargetABI() { return new X86TargetABI; }
