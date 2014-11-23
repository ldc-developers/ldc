//===-- attributes.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/attributes.h"
#include "gen/irstate.h"

bool AttrBuilder::hasAttributes() const {
#if LDC_LLVM_VER >= 302
    return attrs.hasAttributes();
#else
    return attrs.Raw() != 0;
#endif
}

bool AttrBuilder::contains(A attribute) const {
#if LDC_LLVM_VER >= 303
    return attrs.contains(attribute);
#elif LDC_LLVM_VER == 302
    return attrs.hasAttribute(attribute);
#else
    return (attrs & attribute).Raw() != 0;
#endif
}

AttrBuilder& AttrBuilder::clear() {
#if LDC_LLVM_VER >= 302
    attrs.clear();
#else
    attrs = A(0);
#endif
    return *this;
}

AttrBuilder& AttrBuilder::add(A attribute) {
#if LDC_LLVM_VER >= 302
    // never set 'None' explicitly
    if (attribute)
        attrs.addAttribute(attribute);
#else
    attrs |= attribute;
#endif
    return *this;
}

AttrBuilder& AttrBuilder::remove(A attribute) {
#if LDC_LLVM_VER >= 302
    // never remove 'None' explicitly
    if (attribute)
        attrs.removeAttribute(attribute);
#else
    attrs &= ~attribute;
#endif
    return *this;
}


void AttrSet::reserve(size_t length) {
    entries.reserve(length);
}

AttrSet& AttrSet::add(size_t index, const AttrBuilder& builder) {
    if (builder.hasAttributes()) {
        if (index >= entries.size())
            entries.resize(index + 1);
        entries[index] = builder;
    }
    return *this;
}

#if LDC_LLVM_VER >= 303

llvm::AttributeSet AttrSet::toNativeSet() const {
    llvm::AttributeSet set;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].hasAttributes()) {
            AttrBuilder a = entries[i];
            set = set.addAttributes(gIR->context(), i, llvm::AttributeSet::get(gIR->context(), i, a.attrs));
        }
    }
    return set;
}

#else

llvm::AttrListPtr AttrSet::toNativeSet() const {
    std::vector<llvm::AttributeWithIndex> attrsWithIndex;
    attrsWithIndex.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].hasAttributes()) {
#if LDC_LLVM_VER == 302
            AttrBuilder a = entries[i];
            attrsWithIndex.push_back(llvm::AttributeWithIndex::get(i,
                llvm::Attributes::get(gIR->context(), a.attrs)));
#else
            attrsWithIndex.push_back(llvm::AttributeWithIndex::get(i, entries[i].attrs));
#endif
        }
    }
#if LDC_LLVM_VER == 302
    return llvm::AttrListPtr::get(gIR->context(), llvm::ArrayRef<llvm::AttributeWithIndex>(attrsWithIndex));
#else
    return llvm::AttrListPtr::get(attrsWithIndex.begin(), attrsWithIndex.end());
#endif
}

#endif
