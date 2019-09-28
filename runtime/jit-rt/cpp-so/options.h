//===-- options.h - jit support ---------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit runtime - compilation options.
//
//===----------------------------------------------------------------------===//

#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include "slice.h"

bool parseOptions(Slice<Slice<const char>> args,
                  void (*errs)(void *, const char *, size_t),
                  void *errsContext);

#endif // OPTIONS_HPP
