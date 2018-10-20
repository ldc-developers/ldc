//===-- callback_ostream.h - jit support ------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Simple llvm::raw_ostream implementation which sink all input to provided
// callback. It uses llvm::function_ref so user must ensure callback lifetime.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

class CallbackOstream : public llvm::raw_ostream {
  using CallbackT = llvm::function_ref<void(const char *, size_t)>;
  CallbackT callback;
  uint64_t currentPos = 0;

  void write_impl(const char *Ptr, size_t Size) override;

  uint64_t current_pos() const override;

public:
  explicit CallbackOstream(CallbackT c);
};
