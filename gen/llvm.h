#ifndef GEN_LLVM_H
#define GEN_LLVM_H

#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/CallingConv.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Value.h"
#include "llvm/ParameterAttributes.h"

#include "llvm/Target/TargetData.h"

#include "llvm/Support/IRBuilder.h"
using llvm::IRBuilder;

// for WriteTypeSymbolic
#include "llvm/Assembly/Writer.h"

#define GET_INTRINSIC_DECL(_X) (llvm::Intrinsic::getDeclaration(gIR->module, llvm::Intrinsic:: _X ))

// shortcuts for the common llvm types

#define LLType llvm::Type
#define LLFunctionType llvm::FunctionType
#define LLPointerType llvm::PointerType
#define LLStructType llvm::StructType
#define LLArrayType llvm::ArrayType
#define LLIntegerType llvm::IntegerType
#define LLOpaqueType llvm::OpaqueType

#define LLValue llvm::Value
#define LLGlobalValue llvm::GlobalValue
#define LLGlobalVariable llvm::GlobalVariable
#define LLFunction llvm::Function

#define LLConstant llvm::Constant
#define LLConstantStruct llvm::ConstantStruct
#define LLConstantArray llvm::ConstantArray
#define LLConstantInt llvm::ConstantInt
#define LLConstantFP llvm::ConstantFP

#define LLPATypeHolder llvm::PATypeHolder

#define LLSmallVector llvm::SmallVector

#define APFloat llvm::APFloat

#endif // GEN_LLVM_H
