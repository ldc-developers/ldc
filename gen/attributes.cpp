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
    return attrs.hasAttributes();
}

bool AttrBuilder::contains(A attribute) const
{
    return attrs.contains(attribute);
}

AttrBuilder& AttrBuilder::clear()
{
    attrs.clear();
    return *this;
}

AttrBuilder& AttrBuilder::add(A attribute)
{
    // never set 'None' explicitly
    if (attribute)
        attrs.addAttribute(attribute);
    return *this;
}

AttrBuilder& AttrBuilder::remove(A attribute)
{
    // never remove 'None' explicitly
    if (attribute)
        attrs.removeAttribute(attribute);
    return *this;
}

AttrBuilder& AttrBuilder::merge(const AttrBuilder& other)
{
    attrs.merge(other.attrs);
    return *this;
}


AttrSet AttrSet::extractFunctionAndReturnAttributes(const llvm::Function* function)
{
    AttrSet set;

    NativeSet old = function->getAttributes();
    llvm::AttributeSet existingAttrs[] = { old.getFnAttributes(), old.getRetAttributes() };
    set.entries = llvm::AttributeSet::get(gIR->context(), existingAttrs);

    return set;
}

AttrSet& AttrSet::add(unsigned index, const AttrBuilder& builder)
{
    if (builder.hasAttributes())
    {
        AttrBuilder mutableBuilderCopy = builder;
        llvm::AttributeSet as = llvm::AttributeSet::get(
            gIR->context(), index, mutableBuilderCopy.attrs);
        entries = entries.addAttributes(gIR->context(), index, as);
    }
    return *this;
}

AttrSet::NativeSet AttrSet::toNativeSet() const
{
    return entries;
}
