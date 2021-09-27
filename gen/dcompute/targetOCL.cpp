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
#include "dmd/module.h"
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
  bool usedImage;
public:
  TargetOCL(llvm::LLVMContext &c, int oclversion)
      : DComputeTarget(c, oclversion, OpenCL, "ocl", "spv", createSPIRVABI(),
                       // Map from nomimal DCompute address space to OpenCL
                       // address. For OpenCL this is a no-op.
                       {{0, 1, 2, 3, 4}}) {
    const bool is64 = global.params.targetTriple->isArch64Bit();

    _ir = new IRState("dcomputeTargetOCL", ctx);
    _ir->module.setTargetTriple(is64 ? SPIR_TARGETTRIPLE64
                                     : SPIR_TARGETTRIPLE32);

    _ir->module.setDataLayout(is64 ? SPIR_DATALAYOUT64 : SPIR_DATALAYOUT32);
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
    {
      llvm::Metadata *SPIRVerElts[] = {
        llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1)),
        llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 2))};
      llvm::NamedMDNode *SPIRVerMD =
        _ir->module.getOrInsertNamedMetadata("opencl.spir.version");
      SPIRVerMD->addOperand(llvm::MDNode::get(ctx, SPIRVerElts));
    }
    // Add OpenCL version
    {
      llvm::Metadata *OCLVerElts[] = {
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(ctx), tversion / 100)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(ctx), (tversion % 100) / 10))};
      llvm::NamedMDNode *OCLVerMD =
        _ir->module.getOrInsertNamedMetadata("opencl.ocl.version");
      OCLVerMD->addOperand(llvm::MDNode::get(ctx, OCLVerElts));
    }
    if (usedImage) {
      llvm::NamedMDNode *OCLUsedCoreFeatures =
        _ir->module.getOrInsertNamedMetadata("opencl.used.optional.core.features");
      llvm::Metadata *OCLUsedCoreFeaturesElts[] = {
        llvm::MDString::get(ctx,"cl_images")
      };
      OCLUsedCoreFeatures->addOperand(llvm::MDNode::get(ctx, OCLUsedCoreFeaturesElts));
    }
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

    std::array<llvm::SmallVector<llvm::Metadata *, 8>,count_KernArgMD> paramArgs;
    std::array<llvm::StringRef,count_KernArgMD> args = {{
      "kernel_arg_addr_space",
      "kernel_arg_access_qual",
      "kernel_arg_type",
      "kernel_arg_type_qual",
      "kernel_arg_base_type",
      "kernel_arg_name"
    }};

    VarDeclarations *vs = fd->parameters;
    for (i = 0; i < vs->length; i++) {
      VarDeclaration *v = (*vs)[i];
      decodeTypes(paramArgs, v);
    }
    for (i = 0; i < count_KernArgMD; i++) {
      llf->setMetadata(args[i],llvm::MDNode::get(ctx,paramArgs[i]));
    }
  }

  std::string mod2str(MOD mod) {
    return mod & (MODconst | MODimmutable) ? "const" : "";
  }

  std::string basicTypeToString(Type *t) {
    std::stringstream ss;
    auto ty = t->ty;
    if (ty == TY::Tint8)
      ss << "char";
    else if (ty == TY::Tuns8)
      ss << "uchar";
    else if (ty == TY::Tvector) {
      TypeVector *vec = static_cast<TypeVector *>(t);
      auto size = vec->size(Loc());
      auto basety = vec->basetype->ty;
      if (basety == TY::Tint8)
        ss << "char";
      else if (basety == TY::Tuns8)
        ss << "uchar";
      else
        ss << vec->basetype->toChars();
      ss << (int)size;
    } else
      ss << t->toChars();
    return ss.str();
  }

  void decodeTypes(std::array<llvm::SmallVector<llvm::Metadata *, 8>,count_KernArgMD>& attrs,
                   VarDeclaration *v)
  {
    llvm::Optional<DcomputePointer> ptr;
    std::string typeQuals;
    std::string baseTyName;
    std::string tyName;
    std::string accessQual = "none";
    int addrspace = 0;
    if (v->type->ty == TY::Tstruct &&
        (ptr = toDcomputePointer(static_cast<TypeStruct *>(v->type)->sym))) {
      addrspace = ptr->addrspace;
      tyName = basicTypeToString(ptr->type);

      auto ts = ptr->type->isTypeStruct();
      if (ts && isFromLDC_OpenCL(ts->sym)) {
        // TODO: Pipes
        auto name = std::string(ts->toChars());
        if (!usedImage && name.rfind("ldc.opencl.image",0) == 0)
          usedImage = true;
        // parse access qualifiers from ldc.opencl.*_XX_t types like ldc.opencl.image1d_ro_t
        // 4 == length of "XX_t"
        name = name.substr(name.length()-4, 2);
        if (name == "ro")
          accessQual = "read_only";
        else if (name == "wo")
          accessQual = "write_only";
        else if (name == "rw")
          accessQual = "read_write";
        // ldc.opencl.{sampler_t, reserve_id_t} do not get an access qualifier
      } else {
        // things like e.g. GlobalPointer!opencl.image1d_[ro,wo,rw]_t/sampler_t must not have the additional *
        // but all others e.g. GlobalPointer!int must
        tyName += "*";
      }
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
