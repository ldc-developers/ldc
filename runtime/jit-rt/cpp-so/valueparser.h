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

#ifndef VALUEPARSER_H
#define VALUEPARSER_H

namespace llvm {
class Constant;
class Type;
class DataLayout;
}

struct Context;

llvm::Constant *parseInitializer(const Context &context,
                                 const llvm::DataLayout &dataLayout,
                                 llvm::Type *type, const void *data);

#endif // VALUEPARSER_H
