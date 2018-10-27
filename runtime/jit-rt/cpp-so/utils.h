//===-- utils.h - jit support -----------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit runtime - misc routines.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

struct Context;
namespace llvm {
class Module;
}

void fatal(const Context &context, const std::string &reason);
void interruptPoint(const Context &context, const char *desc,
                    const char *object = "");
void verifyModule(const Context &context, llvm::Module &module);
