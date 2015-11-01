//===-- warnings.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/warnings.h"
#include "mtype.h"

void warnInvalidPrintfCall(Loc loc, Expression *arguments, size_t nargs) {
  Expression *arg = arguments;

  // make sure first argument is a string literal, or we can't do much
  // TODO make it smarter ?
  if (arg->op != TOKstring)
    return; // assume valid

  StringExp *strexp = static_cast<StringExp *>(arg);

  // not wchar or dhar
  if (strexp->sz != 1) {
    warning(loc, "printf does not support wchar and dchar strings");
    return;
  }

#if 0
    // check the format string
    const char* str = static_cast<char*>(strexp->string);
    for (size_t i = 0; i < strexp->len; ++i)
    {
        // TODO
    }
#endif
}
