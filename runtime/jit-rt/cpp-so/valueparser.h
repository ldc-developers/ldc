//===-- valueparser.h - jit support -----------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit runtime - value parser.
// Reads data from host process and generates llvm::Constant suitable
// as initializer.
//
//===----------------------------------------------------------------------===//

#pragma once

#if LDC_LLVM_VER < 1700
#include "llvm/ADT/Optional.h"
#else
#include <optional>
namespace llvm {
template <typename T> using Optional = std::optional<T>;
}
#endif
#include "llvm/ADT/STLExtras.h"

namespace llvm {
class Constant;
class Type;
class DataLayout;
}

using ParseInitializerOverride = llvm::Optional<
    llvm::function_ref<llvm::Constant *(llvm::Type &, const void *, size_t)>>;

llvm::Constant *parseInitializer(
    const llvm::DataLayout &dataLayout, llvm::Type &type, const void *data,
    llvm::function_ref<void(const std::string &)> errHandler,
    const ParseInitializerOverride &override = ParseInitializerOverride{});
