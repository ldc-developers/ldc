//===-- bind.cpp ----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "bind.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "valueparser.h"

namespace {
enum { SmallParamsCount = 5 };

llvm::FunctionType *getDstFuncType(llvm::FunctionType &srcType,
                                   const llvm::ArrayRef<ParamSlice> &params) {
  assert(!srcType.isVarArg());
  llvm::SmallVector<llvm::Type *, SmallParamsCount> newParams;
  const auto srcParamsCount = srcType.params().size();
  assert(params.size() == srcParamsCount);
  for (size_t i = 0; i < srcParamsCount; ++i) {
    if (params[i].data == nullptr) {
      newParams.push_back(srcType.getParamType(static_cast<unsigned>(i)));
    }
  }
  auto retType = srcType.getReturnType();
  return llvm::FunctionType::get(retType, newParams, /*isVarArg*/ false);
}

llvm::Function *createBindFunc(llvm::Module &module, llvm::Function &srcFunc,
                               llvm::Function &exampleFunc,
                               llvm::FunctionType &funcType,
                               const llvm::ArrayRef<ParamSlice> &params) {
  auto newFunc = llvm::Function::Create(
      &funcType, llvm::GlobalValue::ExternalLinkage, "\1.jit_bind", &module);

  newFunc->setCallingConv(srcFunc.getCallingConv());
  //  auto srcAttributes = srcFunc.getAttributes();
  //  newFunc->addAttributes(llvm::AttributeList::ReturnIndex,
  //                         srcAttributes.getRetAttributes());
  //  newFunc->addAttributes(llvm::AttributeList::FunctionIndex,
  //                         srcAttributes.getFnAttributes());
  //  unsigned dstInd = 0;
  //  for (size_t i = 0; i < params.size(); ++i) {
  //    if (params[i].data == nullptr) {
  //      newFunc->addAttributes(llvm::AttributeList::FirstArgIndex + dstInd,
  //                             srcAttributes.getParamAttributes(
  //                               static_cast<unsigned>(i)));
  //      ++dstInd;
  //    }
  //  }
  //  assert(dstInd == funcType.getNumParams());
  newFunc->setAttributes(exampleFunc.getAttributes());
  return newFunc;
}

llvm::Value *
allocParam(llvm::IRBuilder<> &builder, llvm::Type &srcType,
           const llvm::DataLayout &layout, const ParamSlice &param,
           llvm::function_ref<void(const std::string &)> errHandler,
           const BindOverride &override) {
  if (param.type == ParamType::Aggregate && srcType.isPointerTy()) {
    auto elemType = srcType.getPointerElementType();
    auto stackArg = builder.CreateAlloca(elemType);
    stackArg->setAlignment(layout.getABITypeAlign(elemType));
    auto init =
        parseInitializer(layout, *elemType, param.data, errHandler, override);
    builder.CreateStore(init, stackArg);
    return stackArg;
  }
  auto stackArg = builder.CreateAlloca(&srcType);
  stackArg->setAlignment(layout.getABITypeAlign(&srcType));
  auto init =
      parseInitializer(layout, srcType, param.data, errHandler, override);
  builder.CreateStore(init, stackArg);
  return builder.CreateLoad(stackArg);
}

void doBind(llvm::Module &module, llvm::Function &dstFunc,
            llvm::Function &srcFunc, const llvm::ArrayRef<ParamSlice> &params,
            llvm::function_ref<void(const std::string &)> errHandler,
            const BindOverride &override) {
  auto &context = dstFunc.getContext();
  auto bb = llvm::BasicBlock::Create(context, "", &dstFunc);

  llvm::IRBuilder<> builder(context);
  builder.SetInsertPoint(bb);
  llvm::SmallVector<llvm::Value *, SmallParamsCount> args;
  auto currentArg = dstFunc.arg_begin();
  auto funcType = srcFunc.getFunctionType();
  auto &layout = module.getDataLayout();
  for (size_t i = 0; i < params.size(); ++i) {
    llvm::Value *arg = nullptr;
    const auto &param = params[i];
    if (param.data == nullptr) {
      arg = currentArg;
      ++currentArg;
    } else {
      auto type = funcType->getParamType(static_cast<unsigned>(i));
      arg = allocParam(builder, *type, layout, param, errHandler, override);
    }
    assert(arg != nullptr);
    args.push_back(arg);
  }
  assert(currentArg == dstFunc.arg_end());

  auto ret = builder.CreateCall(&srcFunc, args);
  if (!srcFunc.isDeclaration()) {
    ret->addAttribute(llvm::AttributeList::FunctionIndex,
                      llvm::Attribute::AlwaysInline);
  }
  ret->setCallingConv(srcFunc.getCallingConv());
  ret->setAttributes(srcFunc.getAttributes());
  if (dstFunc.getReturnType()->isVoidTy()) {
    builder.CreateRetVoid();
  } else {
    builder.CreateRet(ret);
  }
}
}

llvm::Function *
bindParamsToFunc(llvm::Module &module, llvm::Function &srcFunc,
                 llvm::Function &exampleFunc,
                 const llvm::ArrayRef<ParamSlice> &params,
                 llvm::function_ref<void(const std::string &)> errHandler,
                 const BindOverride &override) {
  auto srcType = srcFunc.getFunctionType();
  auto dstType = getDstFuncType(*srcType, params);

  auto newFunc = createBindFunc(module, srcFunc, exampleFunc, *dstType, params);
  doBind(module, *newFunc, srcFunc, params, errHandler, override);
  return newFunc;
}
