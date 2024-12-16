//===-- gen/abi-mips64.cpp - MIPS64 ABI description ------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The MIPS64 N32 and N64 ABI can be found here:
// http://techpubs.sgi.com/library/dynaweb_docs/0640/SGI_Developer/books/Mpro_n32_ABI/sgi_html/index.html
//
//===----------------------------------------------------------------------===//

#include "gen/abi/abi.h"
#include "gen/abi/generic.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/tollvm.h"

struct MIPS64TargetABI : TargetABI {
  const bool Is64Bit;

  explicit MIPS64TargetABI(const bool Is64Bit) : Is64Bit(Is64Bit) {}

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    TargetABI::rewriteArgument(fty, arg);
    if (arg.rewrite)
      return;

    // FIXME
  }
};

// The public getter for abi.cpp
TargetABI *getMIPS64TargetABI(bool Is64Bit) {
  return new MIPS64TargetABI(Is64Bit);
}
