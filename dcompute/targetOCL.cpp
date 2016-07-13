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
#include "gen/logger.h"
#include "dcompute/util.h"
#include <cstring>
namespace {
class TargetOCL : public DComputeTarget {
public:
    TargetOCL(llvm::LLVMContext &c, int oclversion) : DComputeTarget(c,oclversion)
  {
    _ir = new IRState("dcomputeTargetOCL",ctx);
    _ir->module.setTargetTriple( global.params.is64bit ? "sipr64-unknown-unknown" : "sipr-unknown-unknown");
    std::string dl;
    if (global.params.is64bit) {
        dl = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";
    } else {
        dl = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";
    }
    _ir->module.setDataLayout(dl);

    abi = createOCLABI();
    binSuffix="spv";
    int _mapping[PSnum] = {0, 1, 2, 3, 4,};
    memcpy(mapping,_mapping,sizeof(_mapping));
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
    
  }
  void handleKernelFunc(FuncDeclaration *df, llvm::Function *llf) override {
    //TODO: Handle Function attibutes
    
    //mostly copied from clang
      llvm::SmallVector<llvm::Metadata *, 8> kernelMDArgs;
      kernelMDArgs.push_back(llvm::ConstantAsMetadata::get(llf));
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
    IF_LOG Logger::println("TargetOCL::handleKernelFunc");
    LOG_SCOPE
    // MDNode for the kernel argument names.
    llvm::SmallVector<llvm::Metadata*, 8> argNames;
    if (df->parameters) {
        for(int i = 0; i < df->parameters->dim; i++) {
            std::string typeQuals;
            std::string baseTyName;
            std::string tyName;
            std::string accessQual = "none";
            int addrspace = 0;
            VarDeclarations *vs = df->parameters;
            VarDeclaration *v = (*vs)[i];
            Type *ty = v->type;
            IF_LOG Logger::println("TargetOCL::handleKernelFunc: proceesing parameter(%s,type=%s)",v->toPrettyChars(),ty->toPrettyChars());

            if (ty->ty == Tstruct) {
                TemplateInstance *t = ((TypeStruct*)(v->type))->sym->isInstantiated();
                if (t && isFromDCompute_Types(((TypeStruct*)(v->type))->sym)) {
                    IF_LOG Logger::println("From dcompute.types");
                    if (!strcmp(t->tempdecl->ident->string, "Pointer")) {
                        //We have a pointer in an address space
                        //struct Pointer(uint addrspace, T)
                        Expression *exp = isExpression((*t->tiargs)[0]);
                        Type * t1 = isType((*t->tiargs)[1]);
                        addrspace = exp->toInteger();
                    
                        //tyName = T.stringof ~ "*"
                        tyName = t1->toChars()+std::string("*");
                        baseTyName = tyName;
                        //typeQuals = ((T == const U, U) || addrspace == Constant) ? "const" : "";
                        typeQuals = (t1->mod & (MODconst | MODimmutable)|| addrspace == 3) ? "const" : "";
                        // there is no volatile or restrict in D
                    
                } else {
                     //TODO: deal with Pipes and Images. (they are global pointers)
                }
            }

            } else {
                //tyName = T.stringof ~ "*"
                tyName = v->type->toChars();
                baseTyName = tyName;
                //typeQuals = ((T == const U, U) || addrspace == Constant) ? "const" : "";
                typeQuals = v->storage_class & (STCconst | STCimmutable) ? "const" : "";
                // there is no volatile or restrict in D
            }
            // Adding the type and base type to the metadata.
            assert(!tyName.empty() && "Empty type name");
            argTypeNames.push_back(llvm::MDString::get(ctx, tyName));
            assert(!baseTyName.empty() && "Empty base type name");
            argBaseTypeNames.push_back(llvm::MDString::get(ctx, baseTyName));
            argTypeQuals.push_back(llvm::MDString::get(ctx, typeQuals));
            accessQuals.push_back(llvm::MDString::get(ctx, accessQual));
            addressQuals.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::IntegerType::get(ctx,32),addrspace)));
        }
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
