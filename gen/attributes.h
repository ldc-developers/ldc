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

#include <map>

struct AttrBuilder
{
    // A: basic attribute type
    // B: builder type
#if LDC_LLVM_VER >= 303
    typedef llvm::Attribute::AttrKind A;
    typedef llvm::AttrBuilder B;
#elif LDC_LLVM_VER == 302
    typedef llvm::Attributes::AttrVal A;
    typedef llvm::AttrBuilder B;
#else
    typedef llvm::Attributes A;
    typedef llvm::Attributes B;
#endif

    B attrs;

    AttrBuilder() {}
    AttrBuilder(const B& attrs) : attrs(attrs) {}

    bool hasAttributes() const;
    bool contains(A attribute) const;

    AttrBuilder& clear();
    AttrBuilder& add(A attribute);
    AttrBuilder& remove(A attribute);
    AttrBuilder& merge(const AttrBuilder& other);
};

struct AttrSet
{
#if LDC_LLVM_VER >= 303
    typedef llvm::AttributeSet NativeSet;
    NativeSet entries;
#else
    typedef llvm::AttrListPtr NativeSet;
    std::map<unsigned, AttrBuilder> entries;
#endif

    AttrSet() {}
    static AttrSet extractFunctionAndReturnAttributes(const llvm::Function* function);

    AttrSet& add(unsigned index, const AttrBuilder& builder);

    NativeSet toNativeSet() const;
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
