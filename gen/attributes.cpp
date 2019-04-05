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

AttrSet::AttrSet(const AttrSet &base, unsigned index, LLAttribute attribute)
    : set(base.set.addAttribute(gIR->context(), index, attribute)) {}

AttrSet
AttrSet::extractFunctionAndReturnAttributes(const llvm::Function *function) {
  auto old = function->getAttributes();
#if LDC_LLVM_VER >= 500
  return {LLAttributeSet::get(gIR->context(), old.getFnAttributes(),
                              old.getRetAttributes(), {})};
#else
  llvm::AttributeSet existingAttrs[] = {old.getFnAttributes(),
                                        old.getRetAttributes()};
  return {LLAttributeSet::get(gIR->context(), existingAttrs)};
#endif
}

AttrSet &AttrSet::add(unsigned index, const llvm::AttrBuilder &builder) {
  if (builder.hasAttributes()) {
#if LDC_LLVM_VER >= 500
    set = set.addAttributes(gIR->context(), index, builder);
#else
    auto as = LLAttributeSet::get(gIR->context(), index, builder);
    set = set.addAttributes(gIR->context(), index, as);
#endif
  }
  return *this;
}

AttrSet &AttrSet::merge(const AttrSet &other) {
  auto &os = other.set;
#if LDC_LLVM_VER >= 500
  set = LLAttributeSet::get(gIR->context(), {set,os});
#else
  for (unsigned i = 0; i < os.getNumSlots(); ++i) {
    unsigned index = os.getSlotIndex(i);
    set = set.addAttributes(gIR->context(), index, os.getSlotAttributes(i));
  }
#endif
  return *this;
}
