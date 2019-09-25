//===-- valueparser.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "valueparser.h"

#include <cassert>
#include <cstdint>
#include <string>

#include "utils.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/raw_ostream.h"

namespace {

void checkOverrideType(
    llvm::Type &type, llvm::Constant &val,
    const llvm::function_ref<void(const std::string &)> &errHandler) {
  auto retType = val.getType();
  if (retType != &type) {
    std::string str;
    llvm::raw_string_ostream ss(str);
    ss << "Override type mismatch, expected \"";
    type.print(ss);
    ss << "\", got \"";
    retType->print(ss);
    ss << "\"";
    ss.flush();
    errHandler(str);
  }
}

template <typename T>
llvm::Constant *
callOverride(const ParseInitializerOverride &override, llvm::Type &type,
             const T &val,
             const llvm::function_ref<void(const std::string &)> &errHandler) {
  if (override) {
    auto ptr = reinterpret_cast<const char *>(&val);
    auto ret = (*override)(type, ptr, sizeof(val));
    if (ret != nullptr) {
      checkOverrideType(type, *ret, errHandler);
    }
    return ret;
  }
  return nullptr;
}

llvm::Constant *
getBool(llvm::LLVMContext &context, const void *data, llvm::Type &type,
        const llvm::function_ref<void(const std::string &)> &errHandler,
        const ParseInitializerOverride &override) {
  assert(nullptr != data);
  const bool val = *static_cast<const bool *>(data);
  if (auto ret = callOverride(override, type, val, errHandler)) {
    return ret;
  }
  return llvm::ConstantInt::get(context, llvm::APInt(1, (val ? 1 : 0), true));
}

template <typename T>
llvm::Constant *
getInt(llvm::LLVMContext &context, const void *data, llvm::Type &type,
       const llvm::function_ref<void(const std::string &)> &errHandler,
       const ParseInitializerOverride &override) {
  assert(nullptr != data);
  const T val = *static_cast<const T *>(data);
  if (auto ret = callOverride(override, type, val, errHandler)) {
    return ret;
  }
  return llvm::ConstantInt::get(context, llvm::APInt(sizeof(T) * 8, val, true));
}

template <typename T>
llvm::Constant *
getFloat(llvm::LLVMContext &context, const void *data, llvm::Type &type,
         const llvm::function_ref<void(const std::string &)> &errHandler,
         const ParseInitializerOverride &override) {
  assert(nullptr != data);
  const T val = *static_cast<const T *>(data);
  if (auto ret = callOverride(override, type, val, errHandler)) {
    return ret;
  }
  return llvm::ConstantFP::get(context, llvm::APFloat(val));
}

llvm::Constant *
getPtr(llvm::LLVMContext &context, const void *data, llvm::Type &type,
       const llvm::function_ref<void(const std::string &)> &errHandler,
       const ParseInitializerOverride &override) {
  assert(nullptr != data);
  const auto val = *static_cast<const uintptr_t *>(data);
  if (auto ret = callOverride(override, type, val, errHandler)) {
    return ret;
  }
  return llvm::ConstantExpr::getIntToPtr(
      llvm::ConstantInt::get(context, llvm::APInt(sizeof(val) * 8, val)),
      &type);
}

llvm::Constant *
getStruct(const void *data, const llvm::DataLayout &dataLayout,
          llvm::Type &type,
          const llvm::function_ref<void(const std::string &)> &errHandler,
          const ParseInitializerOverride &override) {
  assert(nullptr != data);
  if (override) {
    auto size = dataLayout.getTypeStoreSize(&type);
    auto ptr = static_cast<const char *>(data);
    auto ret = (*override)(type, ptr, size);
    if (ret != nullptr) {
      checkOverrideType(type, *ret, errHandler);
      return ret;
    }
  }
  auto stype = llvm::cast<llvm::StructType>(&type);
  auto slayout = dataLayout.getStructLayout(stype);
  auto numElements = stype->getNumElements();
  llvm::SmallVector<llvm::Constant *, 16> elements(numElements);
  for (unsigned i = 0; i < numElements; ++i) {
    const auto elemType = stype->getElementType(i);
    const auto elemOffset = slayout->getElementOffset(i);
    const auto elemPtr = static_cast<const char *>(data) + elemOffset;
    elements[i] =
        parseInitializer(dataLayout, *elemType, elemPtr, errHandler, override);
  }
  return llvm::ConstantStruct::get(stype, elements);
}
}

llvm::Constant *
parseInitializer(const llvm::DataLayout &dataLayout, llvm::Type &type,
                 const void *data,
                 llvm::function_ref<void(const std::string &)> errHandler,
                 const ParseInitializerOverride &override) {
  assert(nullptr != data);
  auto &llcontext = type.getContext();
  if (type.isIntegerTy()) {
    const auto width = type.getIntegerBitWidth();
    switch (width) {
    case 1:
      return getBool(llcontext, data, type, errHandler, override);
    case 8:
      return getInt<uint8_t>(llcontext, data, type, errHandler, override);
    case 16:
      return getInt<uint16_t>(llcontext, data, type, errHandler, override);
    case 32:
      return getInt<uint32_t>(llcontext, data, type, errHandler, override);
    case 64:
      return getInt<uint64_t>(llcontext, data, type, errHandler, override);
    default:
      errHandler(std::string("Invalid int bit width: ") +
                 std::to_string(width));
      return nullptr;
    }
  }
  if (type.isFloatingPointTy()) {
    const auto width = type.getPrimitiveSizeInBits();
    switch (width) {
    case 32:
      return getFloat<float>(llcontext, data, type, errHandler, override);
    case 64:
      return getFloat<double>(llcontext, data, type, errHandler, override);
    default:
      errHandler(std::string("Invalid fp bit width: ") + std::to_string(width));
      return nullptr;
    }
  }
  if (type.isPointerTy()) {
    return getPtr(llcontext, data, type, errHandler, override);
  }
  if (type.isStructTy()) {
    return getStruct(data, dataLayout, type, errHandler, override);
  }
  if (type.isArrayTy()) {
    auto elemType = type.getArrayElementType();
    const auto step = dataLayout.getTypeAllocSize(elemType);
    const auto numElements = type.getArrayNumElements();
    llvm::SmallVector<llvm::Constant *, 16> elements(numElements);
    for (uint64_t i = 0; i < numElements; ++i) {
      const auto elemPtr = static_cast<const char *>(data) + step * i;
      elements[i] = parseInitializer(dataLayout, *elemType, elemPtr, errHandler,
                                     override);
    }
    return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(&type),
                                    elements);
  }
  std::string tname;
  llvm::raw_string_ostream os(tname);
  type.print(os, true);
  errHandler(std::string("Unhandled type: ") + os.str());
  return nullptr;
}
