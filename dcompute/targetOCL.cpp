//===-- targetOCL.cpp -------------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dcompute/target.h"
#include "dcompute/reflect.h"
namespace {
class TargetOCL : DComputeTarget {
  int ocl_version;
  TargetOCL(Module *_m, int ocl) : m(_m), ocl_version(ocl)
  {}
  void runReflectPass() override {
    auto p = createReflectPass(1,sm_target);
    p->runOnModule(*m);
  }
  void runPointerReplacePass() override {
    int mapping[PSnum] = {0, 1, 2, 3, 4,};
    auto p = createPointerReplacePass(mapping);
    p->runOnModule(*m);
  }
  void addMetadata() override {
    //sopencl.ident?
  }
  void handleKernelFunc(FuncDeclaration *df, llvm::Function *llf) override {
    //TODO: set the calling convention for llf to llvm::CallingConv::SPIR_FUNC
  }
  void handleKernelFunc(FuncDeclaration *df, llvm::Function *llf) override {
    //TODO: set the calling convention for llf to llvm::CallingConv::SPIR_KERNEL
    //TODO: Handle Function attibutes
    
    //mostly copied from clang
      
    // MDNode for the kernel argument address space qualifiers.
    SmallVector<llvm::Metadata *, 8> addressQuals;
    addressQuals.push_back(llvm::MDString::get(Context, "kernel_arg_addr_space"));
    
    // MDNode for the kernel argument access qualifiers (images only).
    SmallVector<llvm::Metadata *, 8> accessQuals;
    accessQuals.push_back(llvm::MDString::get(Context, "kernel_arg_access_qual"));
    
    // MDNode for the kernel argument type names.
    SmallVector<llvm::Metadata *, 8> argTypeNames;
    argTypeNames.push_back(llvm::MDString::get(Context, "kernel_arg_type"));
    
    // MDNode for the kernel argument base type names.
    SmallVector<llvm::Metadata *, 8> argBaseTypeNames;
    argBaseTypeNames.push_back(llvm::MDString::get(Context, "kernel_arg_base_type"));
    
    // MDNode for the kernel argument type qualifiers.
    SmallVector<llvm::Metadata *, 8> argTypeQuals;
    argTypeQuals.push_back(llvm::MDString::get(Context, "kernel_arg_type_qual"));
    
    // MDNode for the kernel argument names.
    SmallVector<llvm::Metadata*, 8> argNames;
    for(int i = 0; i < *df->parameters.dim; i++) {
      std::string typeQuals;
      std::string baseTyName;
      std::string tyName;
      VarDeclaration * v = *df->parameters[i];
      TemplateInstance * t = v->isInstantiated();
      if (t && !strcmp(t->tempdecl->ident->string, "dcompute.types.Pointer") {
        //We have a pointer in an address space
        //struct Pointer(uint addrspace, T)
        int addrspace = (TemplateValueParameter*)(t->tdargs[0])->/* gah what goes here?*/
        addressQuals.push_back(llvm::ConstantAsMetadata::get(IntegerType::getInt32Ty(),addrspace,false));
        //tyName = T.stringof ~ "*"
        //baseTyName = tyName
        //typeQuals = ((T == const U, U) || addrspace == Constant) ? "const" : "";
        // there is no volatile or restrict in D
        //TODO: deal with Pipes and Images. (they are global pointers)
      } else {
        //tyName = T.stringof ~ "*"
        //baseTyName = tyName
        //typeQuals = ((T == const U, U) || addrspace == Constant) ? "const" : "";
        // there is no volatile or restrict in D
      }
      // Adding the type and base type to the metadata.
      assert(!tyName.empty() && "Empty type name");
      argTypeNames.push_back(llvm::MDString::get(Context, tyName));
      assert(!baseTyName.empty() && "Empty base type name");
      argBaseTypeNames.push_back(llvm::MDString::get(Context, baseTyName));
      argTypeQuals.push_back(llvm::MDString::get(Context, typeQuals));
    }
    kernelMDArgs.push_back(llvm::MDNode::get(Context, addressQuals));
    kernelMDArgs.push_back(llvm::MDNode::get(Context, accessQuals));
    kernelMDArgs.push_back(llvm::MDNode::get(Context, argTypeNames));
    kernelMDArgs.push_back(llvm::MDNode::get(Context, argBaseTypeNames));
    kernelMDArgs.push_back(llvm::MDNode::get(Context, argTypeQuals));
    ///-------------------------------
    ///TODO: Handle Function attibutes
    ///-------------------------------
    llvm::MDNode *kernelMDNode = llvm::MDNode::get(Context, kernelMDArgs);
    llvm::NamedMDNode *OpenCLKernelMetadata = llm->getOrInsertNamedMetadata("opencl.kernels");
    OpenCLKernelMetadata->addOperand(kernelMDNode);

  }
}
}

DComputeTarget *createOCLTarget(Module *_m, int oclver) {
  return new TargetOCL(_m,oclver)
}
