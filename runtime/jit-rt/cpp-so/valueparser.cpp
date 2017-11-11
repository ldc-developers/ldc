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
template <typename T>
llvm::ConstantInt *getInt(llvm::LLVMContext &context, const void *data) {
  assert(nullptr != data);
  const T val = *static_cast<const T *>(data);
  return llvm::ConstantInt::get(context, llvm::APInt(sizeof(T) * 8, val, true));
}

template <typename T>
llvm::ConstantFP *getFloat(llvm::LLVMContext &context, const void *data) {
  assert(nullptr != data);
  const T val = *static_cast<const T *>(data);
  return llvm::ConstantFP::get(context, llvm::APFloat(val));
}

llvm::Constant *getPtr(llvm::LLVMContext &context, llvm::Type *targetType,
                       const void *data) {
  assert(nullptr != targetType);
  assert(nullptr != data);
  const auto val = *static_cast<const uintptr_t *>(data);
  return llvm::ConstantExpr::getIntToPtr(
      llvm::ConstantInt::get(context, llvm::APInt(sizeof(val) * 8, val)),
      targetType);
}
}

llvm::Constant *parseInitializer(const Context &context,
                                 const llvm::DataLayout &dataLayout,
                                 llvm::Type *type, const void *data) {
  assert(nullptr != type);
  assert(nullptr != data);
  auto &llcontext = type->getContext();
  if (type->isIntegerTy()) {
    const auto width = type->getIntegerBitWidth();
    switch (width) {
    case 8:
      return getInt<uint8_t>(llcontext, data);
    case 16:
      return getInt<uint16_t>(llcontext, data);
    case 32:
      return getInt<uint32_t>(llcontext, data);
    case 64:
      return getInt<uint64_t>(llcontext, data);
    default:
      fatal(context,
            std::string("Invalid int bit width: ") + std::to_string(width));
    }
  }
  if (type->isFloatingPointTy()) {
    const auto width = type->getPrimitiveSizeInBits();
    switch (width) {
    case 32:
      return getFloat<float>(llcontext, data);
    case 64:
      return getFloat<double>(llcontext, data);
    default:
      fatal(context,
            std::string("Invalid fp bit width: ") + std::to_string(width));
    }
  }
  if (type->isPointerTy()) {
    return getPtr(llcontext, type, data);
  }
  if (type->isStructTy()) {
    auto stype = llvm::cast<llvm::StructType>(type);
    auto slayout = dataLayout.getStructLayout(stype);
    auto numElements = stype->getNumElements();
    llvm::SmallVector<llvm::Constant *, 16> elements(numElements);
    for (unsigned i = 0; i < numElements; ++i) {
      const auto elemType = stype->getElementType(i);
      const auto elemOffset = slayout->getElementOffset(i);
      const auto elemPtr = static_cast<const char *>(data) + elemOffset;
      elements[i] = parseInitializer(context, dataLayout, elemType, elemPtr);
    }
    return llvm::ConstantStruct::get(stype, elements);
  }
  if (type->isArrayTy()) {
    auto elemType = type->getArrayElementType();
    const auto step = dataLayout.getTypeAllocSize(elemType);
    const auto numElements = type->getArrayNumElements();
    llvm::SmallVector<llvm::Constant *, 16> elements(numElements);
    for (uint64_t i = 0; i < numElements; ++i) {
      const auto elemPtr = static_cast<const char *>(data) + step * i;
      elements[i] = parseInitializer(context, dataLayout, elemType, elemPtr);
    }
    return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(type),
                                    elements);
  }
  std::string tname;
  llvm::raw_string_ostream os(tname);
  type->print(os, true);
  fatal(context, std::string("Unhandled type: ") + os.str());
  return nullptr;
}
