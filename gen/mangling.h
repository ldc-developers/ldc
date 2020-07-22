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

#pragma once

#include <string>
#include "dmd/globals.h"

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

std::string getIRMangledName(FuncDeclaration *fdecl, LINK link);
std::string getIRMangledName(VarDeclaration *vd);

std::string getIRMangledFuncName(std::string baseMangle, LINK link);
std::string getIRMangledVarName(std::string baseMangle, LINK link);

std::string getIRMangledAggregateName(AggregateDeclaration *aggrdecl,
                                      const char *suffix = nullptr);
std::string getIRMangledInitSymbolName(AggregateDeclaration *aggrdecl);
std::string getIRMangledVTableSymbolName(AggregateDeclaration *aggrdecl);
std::string getIRMangledClassInfoSymbolName(AggregateDeclaration *aggrdecl);
std::string getIRMangledInterfaceInfosSymbolName(ClassDeclaration *cd);

std::string getIRMangledModuleInfoSymbolName(Module *module);
std::string getIRMangledModuleRefSymbolName(const char *moduleMangle);
