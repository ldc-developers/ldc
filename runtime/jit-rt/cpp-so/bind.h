//===-- bind.h - jit support ------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit runtime - support routines for bind, allow to dynamically create
// specialized functions for each bind instance.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "param_slice.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace llvm {
class Constant;
class Type;
class Module;
class Function;
}

using BindOverride = std::optional<
    llvm::function_ref<llvm::Constant *(llvm::Type &, const void *, size_t)>>;

llvm::Function *
bindParamsToFunc(llvm::Module &module, llvm::Function &srcFunc,
                 llvm::Function &exampleFunc,
                 const llvm::ArrayRef<ParamSlice> &params,
                 llvm::function_ref<void(const std::string &)> errHandler,
                 const BindOverride &override = BindOverride{});
