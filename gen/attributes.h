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
using LLAttributeList = llvm::AttributeList;

class AttrSet {
  LLAttributeList set;

  AttrSet &add(unsigned index, const llvm::AttrBuilder &builder);

public:
  AttrSet() = default;
  AttrSet(const LLAttributeList &nativeSet) : set(nativeSet) {}
  AttrSet(const AttrSet &base, unsigned index, LLAttribute attribute);

  static AttrSet
  extractFunctionAndReturnAttributes(const llvm::Function *function);

  AttrSet &addToParam(unsigned paramIndex, const llvm::AttrBuilder &builder) {
    return add(LLAttributeList::FirstArgIndex + paramIndex, builder);
  }
  AttrSet &addToFunction(const llvm::AttrBuilder &builder) {
    return add(LLAttributeList::FunctionIndex, builder);
  }
  AttrSet &addToReturn(const llvm::AttrBuilder &builder) {
    return add(LLAttributeList::ReturnIndex, builder);
  }
  AttrSet &merge(const AttrSet &other);

  operator LLAttributeList &() { return set; }
  operator const LLAttributeList &() const { return set; }
};
