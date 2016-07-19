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

bool AttrBuilder::hasAttributes() const { return builder.hasAttributes(); }

bool AttrBuilder::contains(LLAttribute attribute) const {
  return builder.contains(attribute);
}

AttrBuilder &AttrBuilder::clear() {
  builder.clear();
  return *this;
}

AttrBuilder &AttrBuilder::add(LLAttribute attribute) {
  // never set 'None' explicitly
  if (attribute) {
    builder.addAttribute(attribute);
  }
  return *this;
}

AttrBuilder &AttrBuilder::remove(LLAttribute attribute) {
  // never remove 'None' explicitly
  if (attribute) {
    builder.removeAttribute(attribute);
  }
  return *this;
}

AttrBuilder &AttrBuilder::merge(const AttrBuilder &other) {
  builder.merge(other.builder);
  return *this;
}

AttrBuilder &AttrBuilder::addAlignment(unsigned alignment) {
  builder.addAlignmentAttr(alignment);
  return *this;
}

AttrBuilder &AttrBuilder::addByVal(unsigned alignment) {
  builder.addAttribute(LLAttribute::ByVal);
  if (alignment != 0) {
    builder.addAlignmentAttr(alignment);
  }
  return *this;
}

AttrBuilder &AttrBuilder::addDereferenceable(unsigned size) {
  builder.addDereferenceableAttr(size);
  return *this;
}


AttrSet::AttrSet(const AttrSet &base, unsigned index, LLAttribute attribute)
    : set(base.set.addAttribute(gIR->context(), index, attribute)) {}

AttrSet
AttrSet::extractFunctionAndReturnAttributes(const llvm::Function *function) {
  AttrSet r;

  llvm::AttributeSet old = function->getAttributes();
  llvm::AttributeSet existingAttrs[] = {old.getFnAttributes(),
                                        old.getRetAttributes()};
  r.set = llvm::AttributeSet::get(gIR->context(), existingAttrs);

  return r;
}

AttrSet &AttrSet::add(unsigned index, const AttrBuilder &builder) {
  if (builder.hasAttributes()) {
    auto as = llvm::AttributeSet::get(gIR->context(), index, builder);
    set = set.addAttributes(gIR->context(), index, as);
  }
  return *this;
}

AttrSet &AttrSet::merge(const AttrSet &other) {
  auto &os = other.set;
  for (unsigned i = 0; i < os.getNumSlots(); ++i) {
    unsigned index = os.getSlotIndex(i);
    set = set.addAttributes(gIR->context(), index, os.getSlotAttributes(i));
  }
  return *this;
}
