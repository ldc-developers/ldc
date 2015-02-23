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


AttrSet& AttrSet::add(unsigned index, AttrBuilder builder) {
    if (builder.hasAttributes())
        entries[index] = builder;
    return *this;
}

#if LDC_LLVM_VER >= 303

llvm::AttributeSet AttrSet::toNativeSet() const {
    llvm::AttributeSet set;

    typedef std::map<unsigned, AttrBuilder>::const_iterator I;
    for (I it = entries.begin(); it != entries.end(); ++it) {
        unsigned index = it->first;
        AttrBuilder builder = it->second;
        if (!builder.hasAttributes())
            continue;

        llvm::AttributeSet as = llvm::AttributeSet::get(gIR->context(),
            index, builder.attrs);
        set = set.addAttributes(gIR->context(), index, as);
    }

    return set;
}

#else

llvm::AttrListPtr AttrSet::toNativeSet() const {
    if (entries.empty())
        return llvm::AttrListPtr();

    std::vector<llvm::AttributeWithIndex> attrsWithIndex;
    attrsWithIndex.reserve(entries.size());

    typedef std::map<unsigned, AttrBuilder>::const_iterator I;
    for (I it = entries.begin(); it != entries.end(); ++it) {
        unsigned index = it->first;
        AttrBuilder builder = it->second;
        if (!builder.hasAttributes())
            continue;

#if LDC_LLVM_VER == 302
        attrsWithIndex.push_back(llvm::AttributeWithIndex::get(index,
            llvm::Attributes::get(gIR->context(), builder.attrs)));
#else
        attrsWithIndex.push_back(llvm::AttributeWithIndex::get(index, builder.attrs));
#endif
    }

#if LDC_LLVM_VER == 302
    return llvm::AttrListPtr::get(gIR->context(), llvm::ArrayRef<llvm::AttributeWithIndex>(attrsWithIndex));
#else
    return llvm::AttrListPtr::get(attrsWithIndex.begin(), attrsWithIndex.end());
#endif
}

#endif
