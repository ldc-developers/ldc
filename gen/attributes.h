//===-- gen/attributes.h - Attribute abstractions ---------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_ATTRIBUTES_H
#define LDC_GEN_ATTRIBUTES_H

#include "gen/llvm.h"

using LLAttribute = llvm::Attribute::AttrKind;
#if LDC_LLVM_VER >= 500
  using LLAttributeSet = llvm::AttributeList;
#else
  using LLAttributeSet = llvm::AttributeSet;
#endif

class AttrBuilder {
  llvm::AttrBuilder builder;

public:
  AttrBuilder() = default;

  bool hasAttributes() const;
  bool contains(LLAttribute attribute) const;

  AttrBuilder &clear();
  AttrBuilder &add(LLAttribute attribute);
  AttrBuilder &remove(LLAttribute attribute);
  AttrBuilder &merge(const AttrBuilder &other);

  AttrBuilder &addAlignment(unsigned alignment);
  AttrBuilder &addByVal(unsigned alignment);
  AttrBuilder &addDereferenceable(unsigned size);

  operator llvm::AttrBuilder &() { return builder; }
  operator const llvm::AttrBuilder &() const { return builder; }
};

class AttrSet {
  LLAttributeSet set;

  AttrSet &add(unsigned index, const AttrBuilder &builder);

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

  AttrSet &addToParam(unsigned paramIndex, const AttrBuilder &builder) {
    return add(paramIndex + FirstArgIndex, builder);
  }
  AttrSet &addToFunction(const AttrBuilder &builder) {
    return add(LLAttributeSet::FunctionIndex, builder);
  }
  AttrSet &addToReturn(const AttrBuilder &builder) {
    return add(LLAttributeSet::ReturnIndex, builder);
  }
  AttrSet &merge(const AttrSet &other);

  operator LLAttributeSet &() { return set; }
  operator const LLAttributeSet &() const { return set; }
};

#endif
