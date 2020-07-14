//===-- gen/attributes.h - Attribute abstractions ---------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "gen/llvm.h"

using LLAttribute = llvm::Attribute::AttrKind;
#if LDC_LLVM_VER >= 500
  using LLAttributeSet = llvm::AttributeList;
#else
  using LLAttributeSet = llvm::AttributeSet;
#endif

class AttrSet {
  LLAttributeSet set;

  AttrSet &add(unsigned index, const llvm::AttrBuilder &builder);

public:
  AttrSet() = default;
  AttrSet(const LLAttributeSet &nativeSet) : set(nativeSet) {}
  AttrSet(const AttrSet &base, unsigned index, LLAttribute attribute);

#if LDC_LLVM_VER >= 500
  static const unsigned FirstArgIndex = LLAttributeSet::FirstArgIndex;
#else
  static const unsigned FirstArgIndex = 1;
#endif

  static AttrSet
  extractFunctionAndReturnAttributes(const llvm::Function *function);

  AttrSet &addToParam(unsigned paramIndex, const llvm::AttrBuilder &builder) {
    return add(paramIndex + FirstArgIndex, builder);
  }
  AttrSet &addToFunction(const llvm::AttrBuilder &builder) {
    return add(LLAttributeSet::FunctionIndex, builder);
  }
  AttrSet &addToReturn(const llvm::AttrBuilder &builder) {
    return add(LLAttributeSet::ReturnIndex, builder);
  }
  AttrSet &merge(const AttrSet &other);

  operator LLAttributeSet &() { return set; }
  operator const LLAttributeSet &() const { return set; }
};
