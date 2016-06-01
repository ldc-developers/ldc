//===-- mangling.h --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Tries to centralize functionality for mangling of symbols.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_MANGLING_H
#define LDC_GEN_MANGLING_H

#include <string>
#include "ddmd/globals.h"

class AggregateDeclaration;
class FuncDeclaration;
class VarDeclaration;

std::string getMangledName(FuncDeclaration *fdecl, LINK link);
std::string getMangledName(VarDeclaration *vd);

std::string getMangledInitSymbolName(AggregateDeclaration *aggrdecl);
std::string getMangledVTableSymbolName(AggregateDeclaration *aggrdecl);
std::string getMangledClassInfoSymbolName(AggregateDeclaration *aggrdecl);

#endif // LDC_GEN_MANGLING_H
