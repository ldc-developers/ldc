//===-- bind.cpp ----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "bind.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "valueparser.h"

namespace {
enum {
  SmallParamsCount = 5
};

llvm::FunctionType *getDstFuncType(llvm::FunctionType &srcType,
                                   const llvm::ArrayRef<Slice> &params) {
  assert(!srcType.isVarArg());
  llvm::SmallVector<llvm::Type*, SmallParamsCount> newParams;
  const auto srcParamsCount = srcType.params().size();
  assert(params.size() == srcParamsCount);
  for (size_t i = 0; i < srcParamsCount; ++i) {
    if (params[i].data == nullptr) {
      newParams.push_back(srcType.getParamType(static_cast<unsigned>(i)));
    }
  }
  auto retType = srcType.getReturnType();
  return llvm::FunctionType::get(retType, newParams, /*isVarArg*/false);
}

llvm::Function *createBindFunc(llvm::Module &module,
                               llvm::Function &srcFunc,
                               llvm::FunctionType &funcType,
                               const llvm::ArrayRef<Slice> &params) {
  auto newFunc = llvm::Function::Create(
                   &funcType, llvm::GlobalValue::ExternalLinkage, "\1.jit_bind",
                   &module);

  newFunc->setCallingConv(srcFunc.getCallingConv());
  auto srcAttributes = srcFunc.getAttributes();
  newFunc->addAttributes(llvm::AttributeList::ReturnIndex,
                         srcAttributes.getRetAttributes());
  newFunc->addAttributes(llvm::AttributeList::FunctionIndex,
                         srcAttributes.getFnAttributes());
  unsigned dstInd = 0;
  for (size_t i = 0; i < params.size(); ++i) {
    if (params[i].data == nullptr) {
      newFunc->addAttributes(llvm::AttributeList::FirstArgIndex + dstInd,
                             srcAttributes.getParamAttributes(
                               static_cast<unsigned>(i)));
      ++dstInd;
    }
  }
  assert(dstInd == funcType.getNumParams());
  return newFunc;
}

void doBind(llvm::Module &module, llvm::Function &dstFunc,
            llvm::Function &srcFunc, const llvm::ArrayRef<Slice> &params,
            llvm::function_ref<void(const std::string &)> errHandler) {
  auto& context = dstFunc.getContext();
  auto bb = llvm::BasicBlock::Create(context, "", &dstFunc);

  llvm::IRBuilder<> builder(context);
  builder.SetInsertPoint(bb);
  llvm::SmallVector<llvm::Value*, SmallParamsCount> args;
  auto currentArg = dstFunc.arg_begin();
  for (size_t i = 0; i < params.size(); ++i) {
    auto type = srcFunc.getFunctionType()->getParamType(static_cast<unsigned>(i));
    llvm::Value* arg = nullptr;
    if (params[i].data == nullptr) {
      arg = currentArg;
      ++currentArg;
    } else {
      auto &layout = module.getDataLayout();
      auto stackArg = builder.CreateAlloca(type);
      stackArg->setAlignment(layout.getABITypeAlignment(type));
      const auto& param = params[i];
      auto init = parseInitializer(layout, *type, param.data, errHandler);
      builder.CreateStore(init, stackArg);
      arg = builder.CreateLoad(stackArg);
    }
    assert(arg != nullptr);
    args.push_back(arg);
  }
  assert(currentArg == dstFunc.arg_end());

  auto ret = builder.CreateCall(&srcFunc, args);
  ret->setCallingConv(srcFunc.getCallingConv());
  if (dstFunc.getReturnType()->isVoidTy()) {
    builder.CreateRetVoid();
  } else {
    builder.CreateRet(ret);
  }
}
}

llvm::Function *bindParamsToFunc(
    llvm::Module &module, llvm::Function &srcFunc,
    const llvm::ArrayRef<Slice> &params,
    llvm::function_ref<void(const std::string &)> errHandler) {
  auto srcType = srcFunc.getFunctionType();
  auto dstType = getDstFuncType(*srcType, params);

  auto newFunc = createBindFunc(module, srcFunc, *dstType, params);
  doBind(module, *newFunc, srcFunc, params, errHandler);
  return newFunc;
}
