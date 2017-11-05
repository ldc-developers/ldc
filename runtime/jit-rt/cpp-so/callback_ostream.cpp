//===-- callback_ostream.cpp ----------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "callback_ostream.h"

void CallbackOstream::write_impl(const char *Ptr, size_t Size) {
  callback(Ptr, Size);
  currentPos += Size;
}

uint64_t CallbackOstream::current_pos() const { return currentPos; }

CallbackOstream::CallbackOstream(CallbackOstream::CallbackT c) : callback(c) {}
