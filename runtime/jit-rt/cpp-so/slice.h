//===-- slice.h - jit support -----------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit runtime - simple memory slice class.
//
//===----------------------------------------------------------------------===//

#ifndef SLICE_HPP
#define SLICE_HPP

#include <cstddef> //size_t

template <typename T> struct Slice final {
  size_t len;
  T *data;
};

#endif // SLICE_HPP
