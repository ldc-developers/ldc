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
    return builder.hasAttributes();
}

bool AttrBuilder::contains(LLAttribute attribute) const
{
    return builder.contains(attribute);
}

AttrBuilder& AttrBuilder::clear()
{
    builder.clear();
    return *this;
}

AttrBuilder& AttrBuilder::add(LLAttribute attribute)
{
    // never set 'None' explicitly
    if (attribute)
        builder.addAttribute(attribute);
    return *this;
}

AttrBuilder& AttrBuilder::remove(LLAttribute attribute)
{
    // never remove 'None' explicitly
    if (attribute)
        builder.removeAttribute(attribute);
    return *this;
}

AttrBuilder& AttrBuilder::merge(const AttrBuilder& other)
{
    builder.merge(other.builder);
    return *this;
}


AttrSet AttrSet::extractFunctionAndReturnAttributes(const llvm::Function* function)
{
    AttrSet r;

    llvm::AttributeSet old = function->getAttributes();
    llvm::AttributeSet existingAttrs[] = { old.getFnAttributes(), old.getRetAttributes() };
    r.set = llvm::AttributeSet::get(gIR->context(), existingAttrs);

    return r;
}

AttrSet& AttrSet::add(unsigned index, const AttrBuilder& builder)
{
    if (builder.hasAttributes())
    {
        llvm::AttributeSet as = llvm::AttributeSet::get(
            gIR->context(), index, builder);
        set = set.addAttributes(gIR->context(), index, as);
    }
    return *this;
}
