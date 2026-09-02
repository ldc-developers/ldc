//===-- gen/dcompute/abi-rewrites.h - dcompute ABI rewrites -----*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains ABI rewrites for the dcompute targets SPIR-V and NVPTX
//
//===----------------------------------------------------------------------===//

#pragma once

#include "gen/abi/generic.h"

#include "ir/irfuncty.h"

#include "llvm/IR/IntrinsicsSPIRV.h"

struct DComputePointerRewrite : ABIRewrite {
  LLValue *put(DValue *v, bool isLValueExp, bool) override {
    LLValue *address = DtoLVal(v);
    return DtoLoad(type(v->type), address, ".DComputePointerRewrite_arg");
  }

  LLValue *getLVal(Type *dty, LLValue *v) override {
    LLValue *mem = DtoAlloca(dty, ".DComputePointerRewrite_param_storage");
    LLType *realStructTy = DtoMemType(dty);
    LLValue *field = DtoGEP(realStructTy, mem, 0u, 0u);
    
    if (gIR && gIR->dcomputetarget &&
        (gIR->dcomputetarget->target == DComputeTarget::ID::Vulkan ||
         gIR->dcomputetarget->target == DComputeTarget::ID::OpenCL)) {
      auto ptr = toDcomputePointer(static_cast<TypeStruct *>(dty)->sym);
      llvm::Type *elemTy = DtoType(ptr->type);
      int realAS = gIR->dcomputetarget->mapping[ptr->addrspace];
      
      llvm::Value *ofType = llvm::PoisonValue::get(elemTy);
      llvm::Metadata *md = llvm::ValueAsMetadata::get(ofType);
      llvm::Value *mdVal = llvm::MetadataAsValue::get(gIR->context(), md);
      
      llvm::Function *assignFn = llvm::Intrinsic::getOrInsertDeclaration(
          &gIR->module, llvm::Intrinsic::spv_assign_ptr_type, {v->getType()});
      gIR->ir->CreateCall(assignFn->getFunctionType(), assignFn,
                          {v, mdVal, llvm::ConstantInt::get(llvm::Type::getInt32Ty(gIR->context()), realAS)});
    }
    
    DtoStore(v, field);
    return mem;
  }

  LLType *type(Type *t) override {
    auto ptr = toDcomputePointer(static_cast<TypeStruct *>(t)->sym);
    return ptr->toLLVMType(true);
  }

  void applyTo(IrFuncTyArg &arg, LLType *finalLType = nullptr) override {
    arg.rewrite = this;
    arg.ltype = finalLType ? finalLType : this->type(arg.type);
    arg.byref = false;
    arg.attrs.removeAttribute(llvm::Attribute::Dereferenceable);
    arg.attrs.removeAttribute(llvm::Attribute::NonNull);
  }
};
