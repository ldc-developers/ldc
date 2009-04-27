// Converted to the D programming language by Tomas Lindquist Olsen 2008
// Original file header:
/*===-- llvm-c/BitWriter.h - BitWriter Library C Interface ------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMBitWriter.a, which          *|
|* implements output of the LLVM bitcode format.                              *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

module llvm.c.BitWriter;

import llvm.c.Core;

extern(C):

/*===-- Operations on modules ---------------------------------------------===*/

/** Writes a module to an open file descriptor. Returns 0 on success. */
int LLVMWriteBitcodeToFileHandle(LLVMModuleRef M, int Handle);

/** Writes a module to the specified path. Returns 0 on success. */
int LLVMWriteBitcodeToFile(LLVMModuleRef M, /*const*/ char *Path);
