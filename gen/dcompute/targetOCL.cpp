//===-- gen/dcomputetargetOCL.cpp -----------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// Parts of this file are adapted from CodeGenFunction.cpp (Clang, LLVM).
// Therefore, this file is distributed under the LLVM license.
// See the LICENSE file for details.
//===----------------------------------------------------------------------===//

#if LDC_LLVM_SUPPORTED_TARGET_SPIRV

#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/template.h"
#include "gen/abi-spirv.h"
#include "gen/dcompute/target.h"
#include "gen/dcompute/druntime.h"
#include "gen/logger.h"
#include "llvm/Transforms/Scalar.h"
#include <cstring>
#include <string>

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
      : DComputeTarget(c, oclversion, OpenCL, "ocl", "spv", createSPIRVABI(),
                       // Map from nomimal DCompute address space to OpenCL
                       // address. For OpenCL this is a no-op.
                       {{0, 1, 2, 3, 4}}) {

    _ir = new IRState("dcomputeTargetOCL", ctx);
    _ir->module.setTargetTriple(global.params.is64bit ? SPIR_TARGETTRIPLE64
                                                      : SPIR_TARGETTRIPLE32);

    _ir->module.setDataLayout(global.params.is64bit ? SPIR_DATALAYOUT64
                                                    : SPIR_DATALAYOUT32);
    _ir->dcomputetarget = this;
  }

  // Adapted from clang
  void addMetadata() override {
    // opencl.ident?
    // spirv.Source // debug only
    // stuff from clang's CGSPIRMetadataAdder.cpp
    // opencl.used.extensions
    // opencl.used.optional.core.features
    // opencl.compiler.options
    // opencl.enable.FP_CONTRACT (-ffast-math)
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
  enum KernArgMD {
      KernArgMD_addr_space,
      KernArgMD_access_qual,
      KernArgMD_type,
      KernArgMD_base_type,
      KernArgMD_type_qual,
      KernArgMD_name,
      count_KernArgMD
  };
  void addKernelMetadata(FuncDeclaration *fd, llvm::Function *llf) override {
    // By the time we get here the ABI should have rewritten the function
    // type so that the magic types in ldc.dcompute are transformed into
    // what the LLVM backend expects.

    unsigned i = 0;
    // TODO: Handle Function attibutes
    llvm::SmallVector<llvm::Metadata *, 8> kernelMDArgs;
    kernelMDArgs.push_back(llvm::ConstantAsMetadata::get(llf));
    // MDNode for the kernel argument address space qualifiers.

    std::array<llvm::SmallVector<llvm::Metadata *, 8>,count_KernArgMD> paramArgs;
    std::array<const char*,count_KernArgMD> args = {{
      "kernel_arg_addr_space",
      "kernel_arg_access_qual",
      "kernel_arg_type",
      "kernel_arg_type_qual",
      "kernel_arg_base_type",
      "kernel_arg_name"
    }};
      
    for (auto md : args) {
      paramArgs[i].push_back(llvm::MDString::get(ctx, md));
      i++;
    }

    VarDeclarations *vs = fd->parameters;
    for (i = 0; i < vs->length; i++) {
      VarDeclaration *v = (*vs)[i];
      decodeTypes(paramArgs, v);
    }

    for (auto &md : paramArgs)
      kernelMDArgs.push_back(llvm::MDNode::get(ctx, md));
    ///-------------------------------
    /// TODO: Handle Function attibutes
    ///-------------------------------
    llvm::MDNode *kernelMDNode = llvm::MDNode::get(ctx, kernelMDArgs);
    llvm::NamedMDNode *OpenCLKernelMetadata =
        _ir->module.getOrInsertNamedMetadata("opencl.kernels");
    OpenCLKernelMetadata->addOperand(kernelMDNode);
  }

  std::string mod2str(MOD mod) {
    return mod & (MODconst | MODimmutable) ? "const" : "";
  }

  std::string basicTypeToString(Type *t) {
    std::stringstream ss;
    auto ty = t->ty;
    if (ty == Tint8)
      ss << "char";
    else if (ty == Tuns8)
      ss << "uchar";
    else if (ty == Tvector) {
      TypeVector *vec = static_cast<TypeVector *>(t);
      auto size = vec->size(Loc());
      auto basety = vec->basetype->ty;
      if (basety == Tint8)
        ss << "char";
      else if (basety == Tuns8)
        ss << "uchar";
      else
        ss << vec->basetype->toChars();
      ss << (int)size;
    } else
      ss << t->toChars();
    return ss.str();
  }

  void decodeTypes(std::array<llvm::SmallVector<llvm::Metadata *, 8>,count_KernArgMD> attrs,
                   VarDeclaration *v)
  {
    llvm::Optional<DcomputePointer> ptr;
    std::string typeQuals;
    std::string baseTyName;
    std::string tyName;
    std::string accessQual = "none";
    int addrspace = 0;
    if (v->type->ty == Tstruct &&
        (ptr = toDcomputePointer(static_cast<TypeStruct *>(v->type)->sym))) {
      addrspace = ptr->addrspace;
      tyName = basicTypeToString(ptr->type) + "*";
      baseTyName = tyName;
      // there is no volatile or restrict (yet) in D
      typeQuals = mod2str(ptr->type->mod);
      // TODO: Images and Pipes They are global pointers to opaques
    } else {
      tyName = basicTypeToString(v->type);
      baseTyName = tyName;
      typeQuals = mod2str(v->type->mod);
    }

    attrs[KernArgMD_addr_space].push_back( // i32 addrspace
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::IntegerType::get(ctx, 32), addrspace)));
    attrs[KernArgMD_access_qual].push_back(llvm::MDString::get(ctx, accessQual));
    attrs[KernArgMD_type].push_back(llvm::MDString::get(ctx, tyName));
    attrs[KernArgMD_base_type].push_back(llvm::MDString::get(ctx, baseTyName));
    attrs[KernArgMD_type_qual].push_back(llvm::MDString::get(ctx, typeQuals));
    attrs[KernArgMD_name].push_back(llvm::MDString::get(ctx, v->ident->toChars()));
  }
};
} // anonymous namespace.

DComputeTarget *createOCLTarget(llvm::LLVMContext &c, int oclver) {
  return new TargetOCL(c, oclver);
}

#endif // LDC_LLVM_SUPPORTED_TARGET_SPIRV
