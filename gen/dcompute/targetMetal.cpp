//===-- gen/dcompute/targetCUDA.cpp ---------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "declaration.h"
#include "gen/dcompute/druntime.h"
#include "mtype.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <optional>
#include <vector>
#include "dmd/identifier.h"

#if LDC_LLVM_SUPPORTED_TARGET_AArch64

#include "gen/dcompute/target.h"
#include "gen/abi/targets.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Scalar.h"
#include "driver/targetmachine.h"
#include <cstring>

namespace {
class TargetMetal : public DComputeTarget {
public:
  TargetMetal(llvm::LLVMContext &c, int version)
      : DComputeTarget(
            c, version, ID::Metal, "metal", "air", createMetalABI(),

            // DCompute Order: [Private, Global, Shared, Constant, Generic]
            // AIR equivalents: Private=0, Device/Global=1, Threadgroup/Shared=3, Constant=2
            {{0, 1, 3, 2, 0}}) {
    const bool is64 = global.params.targetTriple->isArch64Bit();

    _ir = new IRState("dcomputeTargetMetal", ctx);
    // TODO: need to find 32-bit triple
    auto tripleString = "air64_v28-apple-macosx26.0.0";

    // std::string targTripleStr = is64 ? SPIR_TARGETTRIPLE64
    //                                   : SPIR_TARGETTRIPLE32;
    #if LDC_LLVM_VER >= 2100
        llvm::Triple targTriple = llvm::Triple(tripleString);
    #else
        std::string targTriple = tripleString;
    #endif
        _ir->module.setTargetTriple(targTriple);

        llvm::StringRef dataLayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64"
            "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-"
            "v512:512:512-v1024:1024:1024-n8:16:32";

        auto floatABI = ::FloatABI::Hard;
        // targetMachine = createTargetMachine(
        //         targTriple,
        //         is64 ? "" : "",
        //         "", {},
        //         is64 ? ExplicitBitness::M64 : ExplicitBitness::M32, floatABI,
        //         llvm::Reloc::Static, llvm::CodeModel::Medium, codeGenOptLevel(), false);
        _ir->module.setDataLayout(is64 ? dataLayout: /* TODO: need to find 32-bit data layout */dataLayout);
        _ir->dcomputetarget = this;
  }

  void addMetadata() override {
    llvm::NamedMDNode *airVersion = _ir->module.getOrInsertNamedMetadata("air.version");
    llvm::Metadata *major = llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), 2));
    llvm::Metadata *minor = llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), 8));
    llvm::Metadata *patch = llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), 0));
    std::array<llvm::Metadata*, 3> arr = {major, minor, patch};
    airVersion->addOperand(llvm::MDTuple::get(ctx, arr));

    llvm::NamedMDNode *airLangVersion = _ir->module.getOrInsertNamedMetadata("air.language_version");
    std::array<llvm::Metadata*, 4> langArr = {
        llvm::MDString::get(ctx, "Metal"),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), 4)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), 0)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), 0))
    };
    airLangVersion->addOperand(llvm::MDTuple::get(ctx, langArr));
  }

  void addKernelMetadata(FuncDeclaration *df, llvm::Function *llf, StructLiteralExp *_unused_) override {
    llvm::errs() << "\n\nAdding kernel metadata...............\n\n";
    llvm::NamedMDNode *kernels = _ir->module.getOrInsertNamedMetadata("air.kernel");

    std::vector<llvm::Metadata *> kernelMetadataArguments;
    kernelMetadataArguments.push_back(llvm::ConstantAsMetadata::get(llf));

    // XXX: unknown, not sure why we need this, Metal backend expects it
    kernelMetadataArguments.push_back(
      llvm::MDNode::get(ctx, {})
    );

    std::vector<llvm::Metadata *> argumentMetadata = addArgumentMetadata(df, llf);

    kernelMetadataArguments.push_back(
      llvm::MDNode::get(ctx, argumentMetadata)
    );

    llvm::MDTuple *kernelTuple = llvm::MDTuple::get(ctx, kernelMetadataArguments);

    kernels->addOperand(kernelTuple);
  }

  auto addArgumentMetadata(FuncDeclaration *df, llvm::Function *llf) -> std::vector<llvm::Metadata *> {
    std::vector<llvm::Metadata *> kernelMetadataArguments;
    int locationIndex = 0;

    for(auto &arg: llf->args()) {
        std::vector<llvm::Metadata *> argumentMetadata;

        argumentMetadata.push_back(
            llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(
                    llvm::IntegerType::get(ctx, 32), locationIndex)));

        argumentMetadata.push_back(llvm::MDString::get(ctx, "air.buffer"));
        argumentMetadata.push_back(llvm::MDString::get(ctx, "air.location_index"));
        argumentMetadata.push_back(llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), locationIndex)
        ));

        // XXX: unknown, not sure why we need this, Metal backend expects it
        argumentMetadata.push_back(llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), 1)
        ));

        argumentMetadata.push_back(llvm::MDString::get(ctx, "air.read_write"));

        argumentMetadata.push_back(llvm::MDString::get(ctx, "air.address_space"));

        if (arg.getType()->isPointerTy()){
          unsigned addressSpace = arg.getType()->getPointerAddressSpace();
          argumentMetadata.push_back(llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), addressSpace)
          ));
        } else {
          argumentMetadata.push_back(llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), 0)
          ));
        }

        VarDeclaration *vd = (*df->parameters)[locationIndex];
        addArgumentTypeInformation(vd, argumentMetadata);

        if (!argumentMetadata.empty()) {
            kernelMetadataArguments.push_back(llvm::MDTuple::get(ctx, argumentMetadata));
        }

        locationIndex++;
    }

    return kernelMetadataArguments;
  }

  void addArgumentTypeInformation(VarDeclaration *vd, std::vector<llvm::Metadata *> &argumentMetadata) {
    Type *type = nullptr;
    std::optional<DcomputePointer> ptr;
    if (vd->type->ty == TY::Tstruct && (ptr = toDcomputePointer(static_cast<TypeStruct *>(vd->type)->sym))){
      type = ptr->type;
    } else {
      type = vd->type;
    }

    argumentMetadata.push_back(llvm::MDString::get(ctx, "air.arg_type_size"));
    argumentMetadata.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), dmd::size(type, vd->loc))
            ));

    argumentMetadata.push_back(llvm::MDString::get(ctx, "air.arg_type_align_size"));
    argumentMetadata.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 32), type->alignsize())
            ));

    argumentMetadata.push_back(llvm::MDString::get(ctx, "air.arg_type_name"));
    // TODO: check if using char needed instead of int8 as in ocl target implementation
    argumentMetadata.push_back(llvm::MDString::get(ctx, basicTypeToString(type)));

    argumentMetadata.push_back(llvm::MDString::get(ctx, "air.arg_name"));
    argumentMetadata.push_back(llvm::MDString::get(ctx, vd->ident->toChars()));
  }

  auto basicTypeToString(Type *t) -> std::string {
    std::stringstream ss;
    auto ty = t->ty;
    if (ty == TY::Tint8) {
      ss << "char";
    } else if (ty == TY::Tuns8) {
      ss << "uchar";
    } else {
      ss << t->toChars();
    }

    return ss.str();
  }

};
} // anonymous namespace.

auto createMetalTarget(llvm::LLVMContext &c, int version) -> DComputeTarget * {
  return new TargetMetal(c, version);
};

#endif // LDC_LLVM_SUPPORTED_TARGET_AArch64
