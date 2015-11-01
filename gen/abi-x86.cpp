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
  IntegerRewrite integerRewrite;

  X86TargetABI() : isOSX(global.params.isOSX) {}

  llvm::CallingConv::ID callingConv(llvm::FunctionType *ft, LINK l) {
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

  std::string mangleForLLVM(llvm::StringRef name, LINK l) {
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

private:
  bool returnOSXStructInArg(TypeStruct *t) {
    // https://developer.apple.com/library/mac/documentation/DeveloperTools/Conceptual/LowLevelABI/Mac_OS_X_ABI_Function_Calls.pdf
    //
    // OS X variation on IA-32 for returning structs, page 57 section on
    // Returning Results:
    //   "Structures 1 or 2 bytes in size are placed in EAX. Structures 4
    //    or 8 bytes in size are placed in: EAX and EDX. Structures of
    //    other sizes are placed at the address supplied by the caller."
    // Non-POD structs (non-C compatible) should always be returned in an
    // arg though (yes, sometimes extern(C) functions return these, but C
    // code does not handle struct lifecycle).
    size_t sz = t->Type::size();
    return !t->sym->isPOD() || (sz != 1 && sz != 2 && sz != 4 && sz != 8);
  }

  bool isMagicCLong(Type *t) {
    // The frontend has magic structs to express the variable-sized C types
    // for C++ mangling purposes. We need to pass them like integers, not
    // on the stack.

    Type *const bt = t->toBasetype();
    if (bt->ty != Tstruct)
      return false;

    Identifier *id = static_cast<TypeStruct *>(bt)->sym->ident;
    return (id == Id::__c_long) || (id == Id::__c_ulong);
  }

public:
  bool returnInArg(TypeFunction *tf) {
    if (tf->isref)
      return false;

    Type *rt = tf->next->toBasetype();
    // D only returns structs on the stack
    if (tf->linkage == LINKd) {
      return rt->ty == Tstruct || rt->ty == Tsarray;
    }
    // other ABI's follow C, which is cdouble and creal returned on the stack
    // as well as structs (except for some OSX cases).
    else {
      if (isMagicCLong(rt))
        return false;

      if (rt->ty == Tstruct) {
        return !isOSX || returnOSXStructInArg((TypeStruct *)rt);
      }
      return (rt->ty == Tsarray || rt->ty == Tcomplex64 ||
              rt->ty == Tcomplex80);
    }
  }

  bool passByVal(Type *t) {
    return t->toBasetype()->ty == Tstruct || t->toBasetype()->ty == Tsarray;
  }

  void rewriteFunctionType(TypeFunction *tf, IrFuncTy &fty) {
    // extern(D)
    if (tf->linkage == LINKd) {
      // IMPLICIT PARAMETERS

      // mark this/nested params inreg
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
      // otherwise try to mark the last param inreg
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
        } else if (!lastTy->isfloating() &&
                   (sz == 1 || sz == 2 || sz == 4)) // right?
        {
          // rewrite the struct into an integer to make inreg work
          if (lastTy->ty == Tstruct || lastTy->ty == Tsarray) {
            last->rewrite = &integerRewrite;
            last->ltype = integerRewrite.type(last->type, last->ltype);
            last->byref = false;
            // erase previous attributes
            last->attrs.clear();
          }
          last->attrs.add(LLAttribute::InReg);
        }
      }

      // FIXME: tf->varargs == 1 need to use C calling convention and vararg
      // mechanism to live up to the spec:
      // "The caller is expected to clean the stack. _argptr is not passed, it
      // is computed by the callee."

      // EXPLICIT PARAMETERS

      // reverse parameter order
      // for non variadics
      if (!fty.args.empty() && tf->varargs != 1) {
        fty.reverseParams = true;
      }
    }

    // extern(C) and all others
    else {
      // RETURN VALUE

      if ((!fty.ret->byref && isMagicCLong(tf->next)) ||
          tf->next->toBasetype() == Type::tcomplex32) {
        // __c_long -> i32, cfloat -> i64
        fty.ret->rewrite = &integerRewrite;
        fty.ret->ltype = integerRewrite.type(fty.ret->type, fty.ret->ltype);
      } else if (isOSX) {
        // value struct returns should be rewritten as an int type to
        // generate correct register usage (matches clang).
        // note: sret functions change ret type to void so this won't
        // trigger for those
        Type *retTy = fty.ret->type->toBasetype();
        if (!fty.ret->byref && retTy->ty == Tstruct) {
          fty.ret->rewrite = &integerRewrite;
          fty.ret->ltype = integerRewrite.type(fty.ret->type, fty.ret->ltype);
        }
      }

      // IMPLICIT PARAMETERS

      // EXPLICIT PARAMETERS

      // Clang does not pass empty structs, while it seems that GCC does,
      // at least on Linux x86. We don't know whether the C compiler will
      // be Clang or GCC, so just assume Clang on OS X and G++ on Linux.
      if (isOSX) {
        size_t i = 0;
        while (i < fty.args.size()) {
          Type *type = fty.args[i]->type->toBasetype();
          if (type->ty == Tstruct) {
            // Do not pass empty structs at all for C++ ABI compatibility.
            // Tests with clang reveal that more complex "empty" types, for
            // example a struct containing an empty struct, are not
            // optimized in the same way.
            StructDeclaration *sd = static_cast<TypeStruct *>(type)->sym;
            if (sd->fields.empty()) {
              fty.args.erase(fty.args.begin() + i);
              continue;
            }
          }
          ++i;
        }
      }

      for (size_t i = 0; i < fty.args.size(); ++i) {
        IrFuncTyArg *arg = fty.args[i];
        if (!arg->byref && isMagicCLong(arg->type)) {
          arg->rewrite = &integerRewrite;
          arg->ltype = integerRewrite.type(arg->type, arg->ltype);
          arg->byref = false;
          arg->attrs.clear();
        }
      }
    }
  }
};

// The public getter for abi.cpp.
TargetABI *getX86TargetABI() { return new X86TargetABI; }
