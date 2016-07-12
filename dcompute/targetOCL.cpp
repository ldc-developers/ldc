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
#include "template.h"
#include "dcompute/abi-ocl.h"
namespace {
class TargetOCL : public DComputeTarget {
public:
    TargetOCL(llvm::LLVMContext &c, int oclversion) : DComputeTarget(c,oclversion)
  {
      _ir = new IRState("dcomputeTargetCUDA",ctx);
      _ir->module.setTargetTriple( global.params.is64bit ? "sipr64-unknown-unknown" : "sipr-unknown-unknown");
      //TODO: does this need to be changed
#if LDC_LLVM_VER >= 308
      _ir->module.setDataLayout(*gDataLayout);
#else
      _ir->module.setDataLayout(gDataLayout->getStringRepresentation());
#endif
      abi = createOCLABI();
      binSuffix="code.spv";
  }
  void runReflectPass() override {
    auto p = createDComputeReflectPass(1,tversion);
    p->runOnModule(_ir->module);
  }
  /*void runPointerReplacePass() override {
    int mapping[PSnum] = {0, 1, 2, 3, 4,};
    auto p = createPointerReplacePass(mapping);
    p->runOnModule(_ir.module);
  }*/
  void addMetadata() override {
    //opencl.ident?
    //spirv.Source // debug only
    //stuff from clang's CGSPIRMetadataAdder.cpp
    //opencl.enable.FP_CONTRACT
  }
  void handleNonKernelFunc(FuncDeclaration *df, llvm::Function *llf) override {
    //TODO: set the calling convention for llf to llvm::CallingConv::SPIR_FUNC
  }
  void handleKernelFunc(FuncDeclaration *df, llvm::Function *llf) override {
    //TODO: set the calling convention for llf to llvm::CallingConv::SPIR_KERNEL
    //TODO: Handle Function attibutes
    
    //mostly copied from clang
      llvm::SmallVector<llvm::Metadata *, 8> kernelMDArgs;
    // MDNode for the kernel argument address space qualifiers.
      llvm::SmallVector<llvm::Metadata *, 8> addressQuals;
    addressQuals.push_back(llvm::MDString::get(ctx, "kernel_arg_addr_space"));
    
    // MDNode for the kernel argument access qualifiers (images only).
    llvm::SmallVector<llvm::Metadata *, 8> accessQuals;
    accessQuals.push_back(llvm::MDString::get(ctx, "kernel_arg_access_qual"));
    
    // MDNode for the kernel argument type names.
    llvm::SmallVector<llvm::Metadata *, 8> argTypeNames;
    argTypeNames.push_back(llvm::MDString::get(ctx, "kernel_arg_type"));
    
    // MDNode for the kernel argument base type names.
    llvm::SmallVector<llvm::Metadata *, 8> argBaseTypeNames;
    argBaseTypeNames.push_back(llvm::MDString::get(ctx, "kernel_arg_base_type"));
    
    // MDNode for the kernel argument type qualifiers.
    llvm::SmallVector<llvm::Metadata *, 8> argTypeQuals;
    argTypeQuals.push_back(llvm::MDString::get(ctx, "kernel_arg_type_qual"));
    
    // MDNode for the kernel argument names.
    llvm::SmallVector<llvm::Metadata*, 8> argNames;
    for(int i = 0; i < df->parameters->dim; i++) {
      std::string typeQuals;
      std::string baseTyName;
      std::string tyName;
      VarDeclarations *vs = df->parameters;
        VarDeclaration *v = (*vs)[i];
      TemplateInstance *t = v->isInstantiated();
      if (t && !strcmp(t->tempdecl->ident->string, "dcompute.types.Pointer")) {
        //We have a pointer in an address space
        //struct Pointer(uint addrspace, T)
        //int addrspace = (TemplateValueParameter*)(t->tdargs[0])->/* gah what goes here?*/
        //addressQuals.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::IntegerType::get(ctx,32),addrspace)));
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
      argTypeNames.push_back(llvm::MDString::get(ctx, tyName));
      assert(!baseTyName.empty() && "Empty base type name");
      argBaseTypeNames.push_back(llvm::MDString::get(ctx, baseTyName));
      argTypeQuals.push_back(llvm::MDString::get(ctx, typeQuals));
    }
    kernelMDArgs.push_back(llvm::MDNode::get(ctx, addressQuals));
    kernelMDArgs.push_back(llvm::MDNode::get(ctx, accessQuals));
    kernelMDArgs.push_back(llvm::MDNode::get(ctx, argTypeNames));
    kernelMDArgs.push_back(llvm::MDNode::get(ctx, argBaseTypeNames));
    kernelMDArgs.push_back(llvm::MDNode::get(ctx, argTypeQuals));
    ///-------------------------------
    ///TODO: Handle Function attibutes
    ///-------------------------------
    llvm::MDNode *kernelMDNode = llvm::MDNode::get(ctx, kernelMDArgs);
    llvm::NamedMDNode *OpenCLKernelMetadata = _ir->module.getOrInsertNamedMetadata("opencl.kernels");
    OpenCLKernelMetadata->addOperand(kernelMDNode);

  }
};
}
DComputeTarget *createOCLTarget(llvm::LLVMContext& c, int oclver) {
    return new TargetOCL(c,oclver);
}
