//===-- ctfloat.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ctfloat.h"
#include "gen/llvm.h"

using llvm::APFloat;

void CTFloat::toAPFloat(const real_t src, APFloat &dst) {
  if (sizeof(real_t) == 8) {
    dst = APFloat(static_cast<double>(src));
    return;
  }

  assert(sizeof(real_t) > 8 && "real_t < 64 bits?");

  union {
    real_t fp;
    uint64_t bits[(sizeof(real_t) + 7) / 8];
  } u;
  u.fp = src;

#if LDC_LLVM_VER >= 400
  #define PARENS ()
#else
  #define PARENS
#endif

#if __i386__ || __x86_64__
  dst = APFloat(APFloat::x87DoubleExtended PARENS, APInt(80, 2, u.bits));
#elif __aarch64__
  dst = APFloat(APFloat::IEEEquad PARENS, APInt(128, 2, u.bits));
#elif __ppc__ || __ppc64__
  dst = APFloat(APFloat::PPCDoubleDouble PARENS, APInt(128, 2, u.bits));
#else
  llvm_unreachable("Unknown host real_t type for compile-time reals");
#endif
}
