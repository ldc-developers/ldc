//===-- dcompute/targetOCL.cpp
//---------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dcompute/target.h"
#include "dcompute/util.h"
#include "template.h"
#include "dcompute/abi.h"
#include "gen/logger.h"
#include "llvm/Transforms/Scalar.h"
#include <cstring>

// from SPIRVInternal.h
#define SPIR_TARGETTRIPLE32 "spir-unknown-unknown"
#define SPIR_TARGETTRIPLE64 "spir64-unknown-unknown"
#define SPIR_DATALAYOUT32                                                      \
  "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32"                             \
  "-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32"                         \
  "-v32:32:32-v48:64:64-v64:64:64-v96:128:128"                                 \
  "-v128:128:128-v192:256:256-v256:256:256"                                    \
  "-v512:512:512-v1024:1024:1024"
#define SPIR_DATALAYOUT64                                                      \
  "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32"                             \
  "-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32"                         \
  "-v32:32:32-v48:64:64-v64:64:64-v96:128:128"                                 \
  "-v128:128:128-v192:256:256-v256:256:256"                                    \
  "-v512:512:512-v1024:1024:1024"

namespace {
class TargetOCL : public DComputeTarget {
public:
  TargetOCL(llvm::LLVMContext &c, int oclversion)
      : DComputeTarget(c, oclversion) {
    _ir = new IRState("dcomputeTargetOCL", ctx);
    _ir->module.setTargetTriple(global.params.is64bit ? SPIR_TARGETTRIPLE64
                                                      : SPIR_TARGETTRIPLE32);

    _ir->module.setDataLayout(global.params.is64bit ? SPIR_DATALAYOUT64
                                                    : SPIR_DATALAYOUT32);
    target = 1;
    abi = createOCLABI();
    binSuffix = "spv";
    int _mapping[MAX_NUM_TARGET_ADDRSPACES] = {
        0, 1, 2, 3, 4,
    };
    memcpy(mapping, _mapping, sizeof(_mapping));
  }
  void setGTargetMachine() override {
    gTargetMachine = nullptr;
  }
  void addMetadata() override {
    // opencl.ident?
    // spirv.Source // debug only
    // stuff from clang's CGSPIRMetadataAdder.cpp
    // opencl.used.extensions
    // opencl.used.optional.core.features
    // opencl.compiler.options
    // opencl.enable.FP_CONTRACT
    llvm::Metadata *SPIRVerElts[] = {
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1)),
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 2))};
    llvm::NamedMDNode *SPIRVerMD =
        _ir->module.getOrInsertNamedMetadata("opencl.spir.version");
    SPIRVerMD->addOperand(llvm::MDNode::get(ctx, SPIRVerElts));

    // Add OpenCL version
    llvm::Metadata *OCLVerElts[] = {
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(ctx), tversion / 100)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(ctx), (tversion % 100) / 10))};
    llvm::NamedMDNode *OCLVerMD =
        _ir->module.getOrInsertNamedMetadata("opencl.ocl.version");
    OCLVerMD->addOperand(llvm::MDNode::get(ctx, OCLVerElts));
  }
  void handleNonKernelFunc(FuncDeclaration *df, llvm::Function *llf) override {}
  void handleKernelFunc(FuncDeclaration *df, llvm::Function *llf) override {
    // TODO: Handle Function attibutes

    // mostly copied from clang
    llvm::SmallVector<llvm::Metadata *, 8> kernelMDArgs;
    kernelMDArgs.push_back(llvm::ConstantAsMetadata::get(llf));
    // MDNode for the kernel argument address space qualifiers.
    llvm::SmallVector<llvm::Metadata *, 8> addressQuals;
    addressQuals.push_back(llvm::MDString::get(ctx, "kernel_arg_addr_space"));

    // MDNode for the kernel argument access qualifiers (memory objects only).
    llvm::SmallVector<llvm::Metadata *, 8> accessQuals;
    accessQuals.push_back(llvm::MDString::get(ctx, "kernel_arg_access_qual"));

    // MDNode for the kernel argument type names.
    llvm::SmallVector<llvm::Metadata *, 8> argTypeNames;
    argTypeNames.push_back(llvm::MDString::get(ctx, "kernel_arg_type"));

    // MDNode for the kernel argument base type names.
    llvm::SmallVector<llvm::Metadata *, 8> argBaseTypeNames;
    argBaseTypeNames.push_back(
        llvm::MDString::get(ctx, "kernel_arg_base_type"));

    // MDNode for the kernel argument type qualifiers.
    llvm::SmallVector<llvm::Metadata *, 8> argTypeQuals;
    argTypeQuals.push_back(llvm::MDString::get(ctx, "kernel_arg_type_qual"));
    IF_LOG Logger::println("TargetOCL::handleKernelFunc");
    LOG_SCOPE
    // MDNode for the kernel argument names.
    llvm::SmallVector<llvm::Metadata *, 8> argNames;
    if (df->parameters) {
      for (int i = 0; i < df->parameters->dim; i++) {
        std::string typeQuals;
        std::string baseTyName;
        std::string tyName;
        std::string accessQual = "none";
        int addrspace = 0;
        VarDeclarations *vs = df->parameters;
        VarDeclaration *v = (*vs)[i];
        Type *ty = v->type;
        IF_LOG Logger::println(
            "TargetOCL::handleKernelFunc: proceesing parameter(%s,type=%s)",
            v->toPrettyChars(), ty->toPrettyChars());

        if (ty->ty == Tstruct) {
          TemplateInstance *t =
              ((TypeStruct *)(v->type))->sym->isInstantiated();
          if (t && isFromDCompute_Types(((TypeStruct *)(v->type))->sym)) {
            IF_LOG Logger::println("From dcompute.types");
            if (!strcmp(t->tempdecl->ident->string, "Pointer")) {
              // We have a pointer in an address space
              // struct Pointer(uint addrspace, T)
              Expression *exp = isExpression((*t->tiargs)[0]);
              Type *t1 = isType((*t->tiargs)[1]);
              addrspace = exp->toInteger();

              // baseTyName is the underlying type, tyName is an alias (if any).
              // We currently dont have enough info
              // to determine the alias's name.
              // tyName = T.stringof ~ "*"
              tyName = t1->toChars() + std::string("*");
              //FIXME for vector types
              if (tyName.substr(0,4) == "byte") {
                tyName = "char" + tyName.substr(4);
              } else if (tyName.substr(0,5) == "ubyte") {
                tyName = "uchar" + tyName.substr(5);
              }
              baseTyName = tyName;
              // typeQuals = ((T == const U, U) || addrspace == Constant) ?
              // "const" : "";
              typeQuals =
                  (t1->mod & (MODconst | MODimmutable) || addrspace == 3)
                      ? "const"
                      : "";
              // there is no volatile or restrict in D

            } else {
              // TODO: deal with Pipes and Images. (they are global pointers)
            }
          }

        } else {
          // tyName = T.stringof
          tyName = v->type->toChars();
          if (tyName.substr(0,4) == "byte") {
              tyName = "char" + tyName.substr(4);
          } else if (tyName.substr(0,5) == "ubyte") {
              tyName = "uchar" + tyName.substr(5);
          }

          baseTyName = tyName;
          // typeQuals = ((T == const U, U) || addrspace == Constant) ? "const"
          // : "";
          typeQuals =
              v->storage_class & (STCconst | STCimmutable) ? "const" : "";
          // there is no volatile or restrict in D
        }
        // Adding the type and base type to the metadata.
        assert(!tyName.empty() && "Empty type name");
        argTypeNames.push_back(llvm::MDString::get(ctx, tyName));
        assert(!baseTyName.empty() && "Empty base type name");
        argBaseTypeNames.push_back(llvm::MDString::get(ctx, baseTyName));
        argTypeQuals.push_back(llvm::MDString::get(ctx, typeQuals));
        accessQuals.push_back(llvm::MDString::get(ctx, accessQual));
        addressQuals.push_back(
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::IntegerType::get(ctx, 32), addrspace)));
      }
    }

    kernelMDArgs.push_back(llvm::MDNode::get(ctx, addressQuals));
    kernelMDArgs.push_back(llvm::MDNode::get(ctx, accessQuals));
    kernelMDArgs.push_back(llvm::MDNode::get(ctx, argTypeNames));
    kernelMDArgs.push_back(llvm::MDNode::get(ctx, argBaseTypeNames));
    kernelMDArgs.push_back(llvm::MDNode::get(ctx, argTypeQuals));
    ///-------------------------------
    /// TODO: Handle Function attibutes
    ///-------------------------------
    llvm::MDNode *kernelMDNode = llvm::MDNode::get(ctx, kernelMDArgs);
    llvm::NamedMDNode *OpenCLKernelMetadata =
        _ir->module.getOrInsertNamedMetadata("opencl.kernels");
    OpenCLKernelMetadata->addOperand(kernelMDNode);
  }
};
}
DComputeTarget *createOCLTarget(llvm::LLVMContext &c, int oclver) {
  return new TargetOCL(c, oclver);
}
