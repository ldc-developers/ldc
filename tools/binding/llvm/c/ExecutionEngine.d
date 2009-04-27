// Converted to the D programming language by Tomas Lindquist Olsen 2008
// Original file header:
/*===-- llvm-c/ExecutionEngine.h - ExecutionEngine Lib C Iface --*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMExecutionEngine.o, which    *|
|* implements various analyses of the LLVM IR.                                *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

module llvm.c.ExecutionEngine;

import llvm.c.Core;

extern(C):

private
{
    struct LLVM_OpaqueGenericValue {}
    struct LLVM_OpaqueExecutionEngine {}
}

typedef LLVM_OpaqueGenericValue* LLVMGenericValueRef;
typedef LLVM_OpaqueExecutionEngine* LLVMExecutionEngineRef;

/*===-- Operations on generic values --------------------------------------===*/

LLVMGenericValueRef LLVMCreateGenericValueOfInt(LLVMTypeRef Ty,
                                                ulong N,
                                                int IsSigned);

LLVMGenericValueRef LLVMCreateGenericValueOfPointer(void *P);

LLVMGenericValueRef LLVMCreateGenericValueOfFloat(LLVMTypeRef Ty, double N);

uint LLVMGenericValueIntWidth(LLVMGenericValueRef GenValRef);

ulong LLVMGenericValueToInt(LLVMGenericValueRef GenVal,
                                         int IsSigned);

void *LLVMGenericValueToPointer(LLVMGenericValueRef GenVal);

double LLVMGenericValueToFloat(LLVMTypeRef TyRef, LLVMGenericValueRef GenVal);

void LLVMDisposeGenericValue(LLVMGenericValueRef GenVal);

/*===-- Operations on execution engines -----------------------------------===*/

int LLVMCreateExecutionEngine(LLVMExecutionEngineRef *OutEE,
                              LLVMModuleProviderRef MP,
                              char **OutError);

int LLVMCreateInterpreter(LLVMExecutionEngineRef *OutInterp,
                          LLVMModuleProviderRef MP,
                          char **OutError);

int LLVMCreateJITCompiler(LLVMExecutionEngineRef *OutJIT,
                          LLVMModuleProviderRef MP,
                          char **OutError);

void LLVMDisposeExecutionEngine(LLVMExecutionEngineRef EE);

void LLVMRunStaticConstructors(LLVMExecutionEngineRef EE);

void LLVMRunStaticDestructors(LLVMExecutionEngineRef EE);

int LLVMRunFunctionAsMain(LLVMExecutionEngineRef EE, LLVMValueRef F,
                          uint ArgC, /*const*/ char * /*const*/ *ArgV,
                          /*const*/ char * /*const*/ *EnvP);

LLVMGenericValueRef LLVMRunFunction(LLVMExecutionEngineRef EE, LLVMValueRef F,
                                    uint NumArgs,
                                    LLVMGenericValueRef *Args);

void LLVMFreeMachineCodeForFunction(LLVMExecutionEngineRef EE, LLVMValueRef F);

void LLVMAddModuleProvider(LLVMExecutionEngineRef EE, LLVMModuleProviderRef MP);

int LLVMRemoveModuleProvider(LLVMExecutionEngineRef EE,
                             LLVMModuleProviderRef MP,
                             LLVMModuleRef *OutMod, char **OutError);

int LLVMFindFunction(LLVMExecutionEngineRef EE, /*const*/ char *Name,
                     LLVMValueRef *OutFn);
