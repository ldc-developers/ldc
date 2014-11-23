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

#include <vector>

struct AttrBuilder
{
    // A: basic attribute type
#if LDC_LLVM_VER >= 303
    typedef llvm::Attribute::AttrKind A;
#elif LDC_LLVM_VER == 302
    typedef llvm::Attributes::AttrVal A;
#else
    typedef llvm::Attributes A;
#endif

    // field for actual builder
#if LDC_LLVM_VER >= 302
    llvm::AttrBuilder attrs;
#else
    llvm::Attributes attrs;
#endif

    bool hasAttributes() const;
    bool contains(A attribute) const;

    AttrBuilder& clear();
    AttrBuilder& add(A attribute);
    AttrBuilder& remove(A attribute);
};

struct AttrSet
{
    std::vector<AttrBuilder> entries;

    void reserve(size_t length);
    AttrSet& add(size_t index, const AttrBuilder& builder);

#if LDC_LLVM_VER >= 303
    llvm::AttributeSet toNativeSet() const;
#else
    llvm::AttrListPtr toNativeSet() const;
#endif
};

// LDC_ATTRIBUTE(name) helper macro returning:
// * an AttrBuilder::A (enum) value for LLVM 3.2+,
// * or an llvm::Attribute::AttrConst value for LLVM 3.1,
//   which can be implicitly converted to AttrBuilder::A
//   (i.e., llvm::Attributes)
#if LDC_LLVM_VER >= 303
#define LDC_ATTRIBUTE(name) llvm::Attribute::name
#elif LDC_LLVM_VER == 302
#define LDC_ATTRIBUTE(name) llvm::Attributes::name
#else
#define LDC_ATTRIBUTE(name) llvm::Attribute::name
#endif

#endif
