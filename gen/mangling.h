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
class ClassDeclaration;
class FuncDeclaration;
class Module;
class VarDeclaration;

/*
 * These functions return a symbol's LLVM mangle.
 * LLVM's codegen performs target-specific postprocessing of these LLVM mangles
 * (for the final object file mangles) unless the LLVM mangle starts with a 0x1
 * byte. The TargetABI gets a chance to tweak the LLVM mangle.
 */

std::string DtoMangledName(FuncDeclaration *fdecl, LINK link);
std::string DtoMangledName(VarDeclaration *vd);

std::string DtoMangledFuncName(std::string baseMangle, LINK link);
std::string DtoMangledVarName(std::string baseMangle, LINK link);

std::string DtoMangledInitSymbolName(AggregateDeclaration *aggrdecl);
std::string DtoMangledVTableSymbolName(AggregateDeclaration *aggrdecl);
std::string DtoMangledClassInfoSymbolName(AggregateDeclaration *aggrdecl);
std::string DtoMangledInterfaceInfosSymbolName(ClassDeclaration *cd);
std::string DtoMangledModuleInfoSymbolName(Module *module);
std::string DtoMangledModuleRefSymbolName(const char *moduleMangle);

#endif // LDC_GEN_MANGLING_H
