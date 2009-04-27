// Extension of the LLVM C interface for use with D, some things in the
// LLVM 2.2 release are kind sparse or even broken...
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
#ifndef D11_LLVMCEXT_H
#define D11_LLVMCEXT_H

#include "llvm/Type.h"
#include "llvm/Constants.h"
#include "llvm/Support/CFG.h"
#include "llvm/Target/TargetData.h"
#include "llvm-c/Core.h"

#include <sstream>
#include <cstring>

using namespace llvm;
using namespace std;

extern "C"
{

// we need to be able to erase an instruction from its parent
void LLVMEraseFromParent(LLVMValueRef I) {
    unwrap<Instruction>(I)->eraseFromParent();
}

// we need to be able to check if a basic block is terminated
int LLVMIsTerminated(LLVMBasicBlockRef BB) {
    return (unwrap(BB)->getTerminator() != NULL);
}

// we need to be able to check if a basic block has any predecessors
int LLVMHasPredecessors(LLVMBasicBlockRef BB) {
    BasicBlock* B = unwrap(BB);
    return (pred_begin(B) != pred_end(B));
}

// we need to be able to check if a basic block is empty
int LLVMIsBasicBlockEmpty(LLVMBasicBlockRef BB) {
    return unwrap(BB)->empty();
}

// we need to be able to replace all uses of V with W
void LLVMReplaceAllUsesWith(LLVMValueRef V, LLVMValueRef W) {
    unwrap<Value>(V)->replaceAllUsesWith(unwrap<Value>(W));
}

// sometimes it's nice to be able to dump a type, not only values...
void LLVMDumpType(LLVMTypeRef T) {
    unwrap(T)->dump();
}

LLVMValueRef LLVMGetOrInsertFunction(LLVMModuleRef M, char* Name, LLVMTypeRef Type) {
    return wrap(unwrap(M)->getOrInsertFunction(Name, unwrap<FunctionType>(Type)));
}

// being able to determine the "kind" of a value is really useful
unsigned LLVMGetValueKind(LLVMValueRef Value) {
    return unwrap(Value)->getValueID();
}

char* LLVMValueToString(LLVMValueRef v) {
    stringstream ss;
    unwrap(v)->print(ss);
    return strdup(ss.str().c_str());
}

char* LLVMTypeToString(LLVMTypeRef ty) {
    stringstream ss;
    unwrap(ty)->print(ss);
    return strdup(ss.str().c_str());
}

LLVMTypeRef LLVMGetTypeByName(LLVMModuleRef M, char* Name) {
    return wrap(unwrap(M)->getTypeByName(Name));
}

int LLVMIsTypeAbstract(LLVMTypeRef T) {
    return unwrap(T)->isAbstract();
}

} // extern "C"

#endif
