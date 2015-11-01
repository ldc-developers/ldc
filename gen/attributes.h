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

class AttrBuilder
{
    llvm::AttrBuilder builder;

public:
    AttrBuilder() = default;

    bool hasAttributes() const;
    bool contains(LLAttribute attribute) const;

    AttrBuilder& clear();
    AttrBuilder& add(LLAttribute attribute);
    AttrBuilder& remove(LLAttribute attribute);
    AttrBuilder& merge(const AttrBuilder& other);

    operator llvm::AttrBuilder&() { return builder; }
    operator const llvm::AttrBuilder&() const { return builder; }
};

class AttrSet
{
    llvm::AttributeSet set;

public:
    AttrSet() = default;
    static AttrSet extractFunctionAndReturnAttributes(const llvm::Function* function);

    AttrSet& add(unsigned index, const AttrBuilder& builder);

    operator llvm::AttributeSet&() { return set; }
    operator const llvm::AttributeSet&() const { return set; }
};

#endif
