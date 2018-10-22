//===-- gen/cpp-imitating-naming.cpp ----------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include <string>

#include "driver/cl_options-llvm.h"
#include "gen/cpp-imitating-naming.h"

////////////////////////////////////////////////////////////////////////////////
namespace cl = llvm::cl;

////////////////////////////////////////////////////////////////////////////////
static cl::opt<bool>
    cppImitatingNaming("di-imitate-cpp-naming",
                       cl::desc("Imitate C++ type names for debugger"),
                       cl::ZeroOrMore);

////////////////////////////////////////////////////////////////////////////////

extern "C" const char *convertDIdentifierToCPlusPlus(const char *name);

////////////////////////////////////////////////////////////////////////////////

std::string processDITypeName(const std::string &originalName) {
  return cppImitatingNaming
             ? convertDIdentifierToCPlusPlus(originalName.c_str())
             : originalName;
}
