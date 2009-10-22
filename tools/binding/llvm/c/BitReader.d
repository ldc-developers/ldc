// Converted to the D programming language by Tomas Lindquist Olsen 2008
// Original file header:
/*===-- llvm-c/BitReader.h - BitReader Library C Interface ------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMBitReader.a, which          *|
|* implements input of the LLVM bitcode format.                               *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

module llvm.c.BitReader;

import llvm.c.Core;

extern(C):

/* Builds a module from the bitcode in the specified memory buffer, returning a
   reference to the module via the OutModule parameter. Returns 0 on success.
   Optionally returns a human-readable error message via OutMessage. */
int LLVMParseBitcode(LLVMMemoryBufferRef MemBuf,
                     LLVMModuleRef *OutModule, char **OutMessage);

/* Reads a module from the specified path, returning via the OutMP parameter
   a module provider which performs lazy deserialization. Returns 0 on success.
   Optionally returns a human-readable error message via OutMessage. */
int LLVMGetBitcodeModuleProvider(LLVMMemoryBufferRef MemBuf,
                                 LLVMModuleProviderRef *OutMP,
                                 char **OutMessage);
