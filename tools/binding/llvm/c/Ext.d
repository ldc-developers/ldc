// Written in the D programming language by Tomas Lindquist Olsen 2008
// Extensions to the LLVM C interface for the D binding.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
module llvm.c.Ext;

import llvm.c.Core;

// taken from llvm/Value.h
/// An enumeration for keeping track of the concrete subclass of Value that
/// is actually instantiated. Values of this enumeration are kept in the
/// Value classes SubclassID field. They are used for concrete type
/// identification.
enum LLVMValueKind : uint
{
    Argument,              /// This is an instance of Argument
    BasicBlock,            /// This is an instance of BasicBlock
    Function,              /// This is an instance of Function
    GlobalAlias,           /// This is an instance of GlobalAlias
    GlobalVariable,        /// This is an instance of GlobalVariable
    UndefValue,            /// This is an instance of UndefValue
    ConstantExpr,          /// This is an instance of ConstantExpr
    ConstantAggregateZero, /// This is an instance of ConstantAggregateNull
    ConstantInt,           /// This is an instance of ConstantInt
    ConstantFP,            /// This is an instance of ConstantFP
    ConstantArray,         /// This is an instance of ConstantArray
    ConstantStruct,        /// This is an instance of ConstantStruct
    ConstantVector,        /// This is an instance of ConstantVector
    ConstantPointerNull,   /// This is an instance of ConstantPointerNull
    InlineAsm,             /// This is an instance of InlineAsm
    Instruction            /// This is an instance of Instruction
}

extern(C)
{
    void LLVMEraseFromParent(LLVMValueRef I);
    int LLVMIsTerminated(LLVMBasicBlockRef BB);
    int LLVMHasPredecessors(LLVMBasicBlockRef BB);
    int LLVMIsBasicBlockEmpty(LLVMBasicBlockRef BB);
    void LLVMReplaceAllUsesWith(LLVMValueRef V, LLVMValueRef W);

    void LLVMOptimizeModule(LLVMModuleRef M, int doinline);
    void LLVMDumpType(LLVMTypeRef T);

    LLVMValueRef LLVMGetOrInsertFunction(LLVMModuleRef M, char* Name, LLVMTypeRef Type);

    /// Return a strdup()ed string which must be free()ed
    char* LLVMValueToString(LLVMValueRef v);
    char* LLVMTypeToString(LLVMTypeRef ty); /// ditto

    LLVMValueKind LLVMGetValueKind(LLVMValueRef Value);

    LLVMTypeRef LLVMGetTypeByName(LLVMModuleRef M, char* Name);
    int LLVMIsTypeAbstract(LLVMTypeRef T);

    alias void function(void* handle, LLVMTypeRef newT) RefineCallback;
    void LLVMRegisterAbstractTypeCallback(LLVMTypeRef T,
                                          void* handle,
                                          RefineCallback callback);
}
