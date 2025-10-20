//===-- gen/dcomputetargetOCL.cpp -----------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// Parts of this file are adapted from CodeGenFunction.cpp (Clang, LLVM).
// Therefore, this file is distributed under the LLVM license.
// See the LICENSE file for details.
//===----------------------------------------------------------------------===//

#if LDC_LLVM_SUPPORTED_TARGET_SPIRV && LDC_LLVM_VER >= 2100

#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/template.h"
#include "dmd/mangle.h"
#include "dmd/module.h"
#include "gen/abi/targets.h"
#include "gen/dcompute/target.h"
#include "gen/dcompute/druntime.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "driver/targetmachine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Target/TargetMachine.h"
#include <cstring>
#include <string>

using namespace dmd;

namespace {
class TargetVulkan : public DComputeTarget {
public:
  TargetVulkan(llvm::LLVMContext &c, int ver)
      : DComputeTarget(c, ver, ID::Vulkan, "vulkan", "spv", createSPIRVVulkanABI(),
                       {{0, 1, 2, 3, 4}}) {

    _ir = new IRState("dcomputeTargetVulkan", ctx);
    // "spirv-vulkan-foo"? foo = library, pixel, etc
    std::string targTriple = "spirv1.6-unknown-vulkan1.3-compute";
    _ir->module.setTargetTriple(llvm::Triple(targTriple));

    auto floatABI = ::FloatABI::Hard;
    targetMachine = createTargetMachine(
            targTriple, "spirv", "", {},
            ExplicitBitness::None, floatABI,
            llvm::Reloc::Static, llvm::CodeModel::Medium, codeGenOptLevel(), false);

    _ir->module.setDataLayout(targetMachine->createDataLayout());
     
    _ir->dcomputetarget = this;
  }

  void addMetadata() override {}

  llvm::AttrBuilder buildKernAttrs(StructLiteralExp *kernAttr) {
    auto b = llvm::AttrBuilder(ctx);
    b.addAttribute("hlsl.shader", "compute");
    Expressions* elts = static_cast<ArrayLiteralExp*>((*(kernAttr->elements))[0])->elements;
    std::string numthreads = "";
    numthreads += std::to_string((*elts)[0]->toInteger()) + ",";
    numthreads += std::to_string((*elts)[1]->toInteger()) + ",";
    numthreads += std::to_string((*elts)[2]->toInteger());

    b.addAttribute("hlsl.numthreads", numthreads);
    //  ?  "hlsl.wavesize"="8,128,64"
    //  ?  "hlsl.export"
    return b;
  }
  llvm::Function *buildFunction(FuncDeclaration *fd) {
    auto *void_func_void = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),{}, false);
    auto linkage = llvm::GlobalValue::LinkageTypes::ExternalLinkage;
    auto name = llvm::Twine(mangleExact(fd)) + llvm::Twine("_kernel");
    auto *f = llvm::Function::Create(void_func_void, linkage, name, _ir->module);
    f->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
    return f;
  }
  llvm::Type *buildArgType(llvm::Function *llf, llvm::SmallVector<llvm::Type *, 8> &args, llvm::StringRef name) {
    IF_LOG {
      Logger::cout() << "buildArgType: " << *llf << std::endl;
    }
    llvm::FunctionType *tf = llf->getFunctionType();
    for (unsigned int i = 0; i < tf->getNumParams(); i++) {
      llvm::Type *t = tf->getParamType(i);
      if (t->isPointerTy())
        t = getI64Type(); // FIXME: 32 bit pointers on 32 but systems?
      args[i] = t;
    }

    IF_LOG {
      for (auto *arg : args) {
        Logger::cout() << *arg;
      }
    }
    return llvm::StructType::create(ctx, args, name);
  }
  llvm::TargetExtType *buildTargetType(llvm::Type *argType) {
    // TODO: Do we need to bother with a "spirv.Layout" here?
    //auto *layout = llvm::TargetExtType::get(ctx, "spirv.Layout", ElemType,{});
    auto * ArrayType = llvm::ArrayType::get(argType, 0);
    return llvm::TargetExtType::get(ctx, "spirv.VulkanBuffer",
                                    {ArrayType},
                                    {12/*StorageClass*/, 0 /*isWritable*/});
  }

  llvm::Value *buildIntrinsicCall(IRBuilder<>& builder, llvm::StringRef dbg,llvm::StringRef name,
                                     llvm::ArrayRef<llvm::Type *> types, llvm::ArrayRef<llvm::Value *> args) {
    IF_LOG {
      Logger::println("buildIntrinsicCall: %s", name.data());
    }
    LOG_SCOPE
    llvm::Function *intrinsic = llvm::Intrinsic::getOrInsertDeclaration(&_ir->module,
                                                   llvm::Intrinsic::lookupIntrinsicID(name),
                                                   types);
    IF_LOG {
      Logger::cout() << "intrinsic = " << *intrinsic << std::endl;
      Logger::println("args:");
      LOG_SCOPE
      for (auto* arg : args) {
        Logger::cout() << *arg << std::endl;
      }
    }
    
    return builder.CreateCall(intrinsic->getFunctionType(), intrinsic, args, dbg);
  }

  void addKernelMetadata(FuncDeclaration *fd, llvm::Function *llf, StructLiteralExp *kernAttr) override {
    // Fake being HLSL
    llvm::Function *f = buildFunction(fd);
    f->addFnAttrs(buildKernAttrs(kernAttr));

    llvm::SmallVector<llvm::Type *, 8> argTypes(llf->getFunctionType()->getNumParams());
    auto name = llvm::Twine(mangleExact(fd)) + llvm::Twine("_args");
    auto *argType = buildArgType(llf, argTypes, name.str());
    llvm::Type *targetType = buildTargetType(argType);
  
    auto bb = llvm::BasicBlock::Create(ctx, "", f);
    llvm::IRBuilder<> builder(ctx);
    builder.SetInsertPoint(bb);

    llvm::Value *i32zero = llvm::ConstantInt::get(getI32Type(), 0, false);
    llvm::Value *i32one  = llvm::ConstantInt::get(getI32Type(), 1, false);
    llvm::Value *i1false = llvm::ConstantInt::get(llvm::Type::getInt1Ty(ctx),  0, false);
    
    // We can't use `DtoConstCString` here because it ends up in the wrong address space, So we use
    // `getCachedStringLiteral` directly with an explicitly supplied addrspace of `0`.
    // FIXME: call should have `notnull` attribute on pointer?
    auto *handle = buildIntrinsicCall(builder, "handle","llvm.spv.resource.handlefrombinding",
                                      {targetType},
                                      {i32zero, i32zero, i32one, i32zero, i1false, _ir->getCachedStringLiteral(name.str(), 0) });
    auto *p11 = llvm::PointerType::get(ctx, 11);
    auto *pointer = buildIntrinsicCall(builder, "pointer", "llvm.spv.resource.getpointer",
                                       {p11, targetType}, {handle, i32one});
    llvm::FunctionType *tf = llf->getFunctionType();
    IF_LOG  {
      Logger::cout() << "load pointer: " << *pointer << std::endl;
      Logger::cout() << _ir->module.getDataLayout().getABITypeAlign(argType).value() << std::endl;
      Logger::cout() << tf->getParamType(0)->getTypeID() << std::endl;
      Logger::cout() << "done" << std::endl;
    }
    LOG_SCOPE
    llvm::SmallVector<llvm::Value *, 8> args(tf->getNumParams());

    auto *arg = builder.CreateAlignedLoad(argType, pointer, _ir->module.getDataLayout().getABITypeAlign(argType), false);
    IF_LOG {
    //  Logger::cout() << "load elements from " << *arg << std::endl;
    //  Logger::cout() << "of type " << *argType << std::endl;
    }
    for (unsigned int i = 0; i < tf->getNumParams(); i++) {
      args[i] = builder.CreateExtractValue(arg, {i});
      llvm::Type *t = tf->getParamType(i);
      if (t->isPointerTy())
        args[i] = builder.CreateIntToPtr(args[i],t);
    }

    builder.CreateCall(llf->getFunctionType(), llf, args);
    builder.CreateRetVoid();
    IF_LOG Logger::cout() << *f << std::endl;
  }

};
} // anonymous namespace.

DComputeTarget *createVulkanTarget(llvm::LLVMContext &c, int ver) {
  return new TargetVulkan(c, ver);
}

#endif // LDC_LLVM_SUPPORTED_TARGET_SPIRV
