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

#pragma once

#include "llvm/Support/CommandLine.h" // for llvm::cl::boolOrDefault

namespace llvm {
class Module;
class LLVMContext;
}

template <typename TYPE> struct Array;

/**
 * Indicates whether -link-internally is enabled.
 */
bool useInternalLLDForLinking();

/**
 * Indicates the status of the -static command-line option.
 */
llvm::cl::boolOrDefault linkFullyStatic();

/**
 * Indicates whether the command-line options select debug druntime/Phobos for
 * linking.
 */
bool linkAgainstDebugDefaultLibs();

/**
 * Indicates whether the command-line options select shared druntime/Phobos for
 * linking.
 */
bool linkAgainstSharedDefaultLibs();

/**
 * Indicates whether the internal 'toolchain' (-link-internally and MinGW-w64
 * libs) is to be used for MSVC targets.
 */
bool useInternalToolchainForMSVC();

/**
 * Returns the name of the MS C runtime library to link with.
 */
llvm::StringRef getMscrtLibName();

/**
 * Inserts bitcode files passed on the commandline into a module.
 */
void insertBitcodeFiles(llvm::Module &M, llvm::LLVMContext &Ctx,
                        Array<const char *> &bitcodeFiles);

/**
 * Link an executable only from object files.
 * @return 0 on success.
 */
int linkObjToBinary();

/**
 * Delete the executable that was previously linked with linkObjToBinary.
 */
void deleteExeFile();

/**
 * Runs the executable that was previously linked with linkObjToBinary.
 * @return the return status of the executable.
 */
int runProgram();
