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

bool AttrBuilder::hasAttributes() const
{
#if LDC_LLVM_VER >= 302
    return attrs.hasAttributes();
#else
    return attrs.Raw() != 0;
#endif
}

bool AttrBuilder::contains(A attribute) const
{
#if LDC_LLVM_VER >= 303
    return attrs.contains(attribute);
#elif LDC_LLVM_VER == 302
    return attrs.hasAttribute(attribute);
#else
    return (attrs & attribute).Raw() != 0;
#endif
}

AttrBuilder& AttrBuilder::clear()
{
#if LDC_LLVM_VER >= 302
    attrs.clear();
#else
    attrs = A(0);
#endif
    return *this;
}

AttrBuilder& AttrBuilder::add(A attribute)
{
#if LDC_LLVM_VER >= 302
    // never set 'None' explicitly
    if (attribute)
        attrs.addAttribute(attribute);
#else
    attrs |= attribute;
#endif
    return *this;
}

AttrBuilder& AttrBuilder::remove(A attribute)
{
#if LDC_LLVM_VER >= 302
    // never remove 'None' explicitly
    if (attribute)
        attrs.removeAttribute(attribute);
#else
    attrs &= ~attribute;
#endif
    return *this;
}

AttrBuilder& AttrBuilder::merge(const AttrBuilder& other)
{
#if LDC_LLVM_VER >= 303
    attrs.merge(other.attrs);
#elif LDC_LLVM_VER == 302
    AttrBuilder mutableCopy = other;
    attrs.addAttributes(llvm::Attributes::get(gIR->context(), mutableCopy.attrs));
#else
    attrs |= other.attrs;
#endif
    return *this;
}


AttrSet AttrSet::extractFunctionAndReturnAttributes(const llvm::Function* function)
{
    AttrSet set;

#if LDC_LLVM_VER >= 303
    NativeSet old = function->getAttributes();
    llvm::AttributeSet existingAttrs[] = { old.getFnAttributes(), old.getRetAttributes() };
    set.entries = llvm::AttributeSet::get(gIR->context(), existingAttrs);
#else
    unsigned fnIndex = ~0u;
    unsigned retIndex = 0;

#if LDC_LLVM_VER == 302
    #define ADD_ATTRIBS(i, a) \
    if (a.Raw()) \
        set.entries[i].attrs.addAttributes(a);
#else
    #define ADD_ATTRIBS(i, a) \
    if (a.Raw()) \
        set.entries[i].attrs = a;
#endif

    ADD_ATTRIBS(fnIndex, function->getAttributes().getFnAttributes());
    ADD_ATTRIBS(retIndex, function->getAttributes().getRetAttributes());

    #undef ADD_ATTRIBS
#endif

    return set;
}

AttrSet& AttrSet::add(unsigned index, const AttrBuilder& builder)
{
    if (builder.hasAttributes())
    {
#if LDC_LLVM_VER >= 303
        AttrBuilder mutableBuilderCopy = builder;
        llvm::AttributeSet as = llvm::AttributeSet::get(
            gIR->context(), index, mutableBuilderCopy.attrs);
        entries = entries.addAttributes(gIR->context(), index, as);
#else
        entries[index].merge(builder);
#endif
    }
    return *this;
}

AttrSet::NativeSet AttrSet::toNativeSet() const
{
#if LDC_LLVM_VER >= 303
    return entries;
#else
    if (entries.empty())
        return NativeSet();

    std::vector<llvm::AttributeWithIndex> attrsWithIndex;
    attrsWithIndex.reserve(entries.size());

    typedef std::map<unsigned, AttrBuilder>::const_iterator I;
    for (I it = entries.begin(); it != entries.end(); ++it)
    {
        unsigned index = it->first;
        const AttrBuilder& builder = it->second;
        if (!builder.hasAttributes())
            continue;

#if LDC_LLVM_VER == 302
        AttrBuilder mutableBuilderCopy = builder;
        attrsWithIndex.push_back(llvm::AttributeWithIndex::get(index,
            llvm::Attributes::get(gIR->context(), mutableBuilderCopy.attrs)));
#else
        attrsWithIndex.push_back(llvm::AttributeWithIndex::get(index, builder.attrs));
#endif
    }

#if LDC_LLVM_VER == 302
    return NativeSet::get(gIR->context(), attrsWithIndex);
#else
    return NativeSet::get(attrsWithIndex.begin(), attrsWithIndex.end());
#endif
#endif
}
