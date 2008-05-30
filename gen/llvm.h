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

#include "llvm/Target/TargetData.h"

#include "llvm/Support/IRBuilder.h"
using llvm::IRBuilder;

#define GET_INTRINSIC_DECL(_X) (llvm::Intrinsic::getDeclaration(gIR->module, llvm::Intrinsic:: _X ))

// shortcuts for the _very_ common llvm types
typedef llvm::Type LLType;
typedef llvm::Value LLValue;
typedef llvm::Constant LLConstant;

#define LLSmallVector llvm::SmallVector

#endif // GEN_LLVM_H
