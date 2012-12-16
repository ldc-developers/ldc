//===-- driver/linker.h - Linker invocation ---------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Handles locating and executing the system linker for generating
// libraries/executables.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_LINKER_H
#define LDC_GEN_LINKER_H

#include "llvm/Support/CommandLine.h"
#include <vector>

extern llvm::cl::opt<bool> quiet;

namespace llvm
{
    class Module;
}

/**
 * Links the modules given in MV in to dst.
 * @param dst Destination module.
 * @param MV Vector of modules to link in to destination.
 */
void linkModules(llvm::Module* dst, const std::vector<llvm::Module*>& MV);

/**
 * Link an executable only from object files.
 * @param argv0 the argv[0] value as passed to main
 * @return 0 on success.
 */
int linkObjToBinary(bool sharedLib);

/**
 * Create a static library from object files.
*/
void createStaticLibrary();

/**
 * Delete the executable that was previously linked with linkObjToBinary.
 */
void deleteExecutable();

/**
 * Runs the executable that was previously linked with linkObjToBinary.
 * @return the return status of the executable.
 */
int runExecutable();

#endif // LDC_GEN_LINKER_H
