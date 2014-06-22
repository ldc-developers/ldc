//===-- gen/llvm.h - Common LLVM includes and aliases -----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Pulls in commonly used LLVM headers and provides shorthands for some LLVM
// types.
//
// TODO: Consider removing this file; the aliases mostly make code more
// cumbersome to read for people familiar with LLVM anyway.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_LLVM_H
#define LDC_GEN_LLVM_H

#if LDC_LLVM_VER >= 303
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#if LDC_LLVM_VER >= 305
#include "llvm/IR/DebugInfo.h"
#else
#include "llvm/DebugInfo.h"
#endif
#else
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/CallingConv.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Value.h"
#include "llvm/Attributes.h"
#if LDC_LLVM_VER == 302
#include "llvm/DataLayout.h"
#include "llvm/IRBuilder.h"
#include "llvm/DebugInfo.h"
#else
#include "llvm/Target/TargetData.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Support/IRBuilder.h"
#endif
#endif


#include "gen/llvmcompat.h"

#if LDC_LLVM_VER >= 305
#include "llvm/IR/CallSite.h"
#else
#include "llvm/Support/CallSite.h"
#endif

using llvm::IRBuilder;

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

#define LLCallSite llvm::CallSite

#define LLSmallVector llvm::SmallVector

using llvm::APFloat;
using llvm::APInt;

#endif // LDC_GEN_LLVM_H
