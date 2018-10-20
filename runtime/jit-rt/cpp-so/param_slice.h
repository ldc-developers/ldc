//===-- param_slice.h - jit support -----------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ParamSlice declaration. Holds pointer into bind parameter and some metadata,
// will be null for placeholders.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef> //size_t
#include <cstdint>

enum ParamType : uint32_t { Simple = 0, Aggregate = 1 };

struct ParamSlice {
  const void *data;
  size_t size;
  ParamType type;
};
