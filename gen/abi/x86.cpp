//===-- abi-x86.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/enum.h"
#include "dmd/id.h"
#include "gen/abi/generic.h"
#include "gen/abi/abi.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irfuncty.h"

struct X86TargetABI : TargetABI {
  const bool isDarwin;
  const bool isMSVC;
  bool returnStructsInRegs;
  IntegerRewrite integerRewrite;
  IndirectByvalRewrite indirectByvalRewrite;

  X86TargetABI()
      : isDarwin(global.params.targetTriple->isOSDarwin()),
        isMSVC(global.params.targetTriple->isWindowsMSVCEnvironment()) {
    using llvm::Triple;
    auto os = global.params.targetTriple->getOS();
    returnStructsInRegs =
        !(os == Triple::Linux || os == Triple::Solaris || os == Triple::NetBSD);
  }

  llvm::CallingConv::ID callingConv(LINK l) override {
    switch (l) {
    case LINK::d:
    case LINK::default_:
    case LINK::windows:
      return llvm::CallingConv::X86_StdCall;
    default:
      return llvm::CallingConv::C;
    }
  }
  llvm::CallingConv::ID callingConv(TypeFunction *tf,
                                    bool withThisPtr) override {
    if (tf->parameterList.varargs == VARARGvariadic)
      return llvm::CallingConv::C;

    // MSVC++ passes the `this` pointer in ECX (D: EAX)
    // follow suit, incl. extern(C++) delegates
    if (isMSVC && tf->linkage == LINK::cpp && withThisPtr)
      return llvm::CallingConv::X86_ThisCall;

    return callingConv(tf->linkage);
  }

  std::string mangleFunctionForLLVM(std::string name, LINK l) override {
    if (global.params.targetTriple->isOSWindows()) {
      if (l == LINK::d || l == LINK::default_) {
        // Prepend a 0x1 byte to prevent LLVM from applying MS stdcall mangling:
        // _D… => __D…@<paramssize>, and add extra underscore manually.
        name.insert(0, "\1_");
      } else if (l == LINK::cpp && name[0] == '?') {
        // Prepend a 0x1 byte to prevent LLVM from prepending the C underscore
        // for MSVC++ symbols (starting with '?').
        name.insert(0, "\1");
      }
    }
    return name;
  }

  std::string mangleVariableForLLVM(std::string name, LINK l) override {
    if (global.params.targetTriple->isOSWindows() && l == LINK::cpp &&
        name[0] == '?') {
      // Prepend a 0x1 byte to prevent LLVM from prepending the C underscore for
      // MSVC++ symbols (starting with '?').
      name.insert(0, "\1");
    }
    return name;
  }

  // Helper folding the magic __c_complex_{float,double,real} enums to the basic
  // complex type.
  static Type *getExtraLoweredReturnType(TypeFunction *tf) {
    Type *rt = tf->next;
    if (auto te = rt->isTypeEnum()) {
      auto id = te->sym->ident;
      if (id == Id::__c_complex_float)
        return Type::tcomplex32;
      if (id == Id::__c_complex_double)
        return Type::tcomplex64;
      if (id == Id::__c_complex_real)
        return Type::tcomplex80;
    }
    return rt->toBasetype();
  }

  bool returnInArg(TypeFunction *tf, bool needsThis) override {
    if (tf->isref())
      return false;

    Type *rt = getExtraLoweredReturnType(tf);
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
      return rt->ty != TY::Tcomplex32;
    }

    // non-extern(D): some OSs don't return structs in registers at all
    if (!externD && !returnStructsInRegs)
      return true;

    const bool isMSVCpp = isMSVC && tf->linkage == LINK::cpp;

    // for non-static member functions, MSVC++ enforces sret for all structs
    if (isMSVCpp && needsThis && rt->ty == TY::Tstruct) {
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
    // indirectly by-value for non-POD args (except for MSVC++)
    const bool isMSVCpp = isMSVC && tf->linkage == LINK::cpp;
    if (!isMSVCpp && !isPOD(t))
      return false;

    // pass all structs and static arrays with the LLVM byval attribute
    return DtoIsInMemoryOnly(t);
  }

  void rewriteFunctionType(IrFuncTy &fty) override {
    const bool externD = isExternD(fty.type);

    // return value:
    if (!skipReturnValueRewrite(fty)) {
      Type *rt = getExtraLoweredReturnType(fty.type);
      if (isAggregate(rt) && canRewriteAsInt(rt) &&
          // don't rewrite cfloat for extern(D)
          !(externD && rt->ty == TY::Tcomplex32)) {
        integerRewrite.applyToIfNotObsolete(*fty.ret);
      }
    }

    // non-POD args are passed indirectly by-value (except for MSVC++)
    const bool isMSVCpp = isMSVC && fty.type->linkage == LINK::cpp;
    if (!isMSVCpp) {
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

      // ... otherwise try the first argument
      else if (!fty.args.empty()) {
        // The first parameter is passed in EAX rather than being pushed on the
        // stack if the following conditions are met:
        //   * It fits in EAX.
        //   * It is not a 3 byte struct.
        //   * It is not a floating point type.

        IrFuncTyArg &first = *fty.args[0];
        if (first.rewrite == &indirectByvalRewrite ||
            (first.byref && !first.isByVal())) {
          Logger::println("Putting first (byref) parameter in register");
          first.attrs.addAttribute(LLAttribute::InReg);
        } else {
          Type *firstTy = first.type->toBasetype();
          auto sz = firstTy->size();
          if (!firstTy->isfloating() && (sz == 1 || sz == 2 || sz == 4)) {
            // rewrite aggregates as integers to make inreg work
            if (firstTy->ty == TY::Tstruct || firstTy->ty == TY::Tsarray) {
              integerRewrite.applyTo(first);
              // undo byval semantics applied via passByVal() returning true
              first.byref = false;
              first.attrs.clear();
            }
            first.attrs.addAttribute(LLAttribute::InReg);
          }
        }
      }
    }

    workaroundIssue1356(fty.args);

    // Clang does not pass empty structs, while it seems that GCC does,
    // at least on Linux x86. We don't know whether the C compiler will
    // be Clang or GCC, so just assume Clang on Darwin and G++ on Linux.
    if (externD || !isDarwin)
      return;

    size_t i = 0;
    while (i < fty.args.size()) {
      Type *type = fty.args[i]->type->toBasetype();
      if (type->ty == TY::Tstruct) {
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
        if (arg->isByVal()) {
          // Keep alignment for LLVM 13+, to prevent invalid `movaps` etc.,
          // but limit to 4 (required according to runnable/ldc_cabi1.d).
          auto align4 = llvm::Align(4);
          if (arg->attrs.getAlignment().value_or(align4) > align4)
            arg->attrs.addAlignmentAttr(align4);
        }
      }
    }
  }

  const char *objcMsgSendFunc(Type *ret, IrFuncTy &fty) override {
    // see objc/message.h for objc_msgSend selection rules
    assert(isDarwin);
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
