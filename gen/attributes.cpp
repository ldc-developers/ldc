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
#if LDC_LLVM_VER >= 1400
    : set(base.set.addAttributeAtIndex(gIR->context(), index, attribute)) {}
#else
    : set(base.set.addAttribute(gIR->context(), index, attribute)) {}
#endif

AttrSet
AttrSet::extractFunctionAndReturnAttributes(const llvm::Function *function) {
  auto old = function->getAttributes();
  return {LLAttributeList::get(gIR->context(),
#if LDC_LLVM_VER >= 1400
                               old.getFnAttrs(),
                               old.getRetAttrs(),
#else
                               old.getFnAttributes(),
                               old.getRetAttributes(),
#endif
                               {})};
}

AttrSet &AttrSet::add(unsigned index, const llvm::AttrBuilder &builder) {
  if (builder.hasAttributes()) {
#if LDC_LLVM_VER >= 1400
    set = set.addAttributesAtIndex(gIR->context(), index, builder);
#else
    set = set.addAttributes(gIR->context(), index, builder);
#endif
  }
  return *this;
}

AttrSet &AttrSet::merge(const AttrSet &other) {
  auto &os = other.set;
  set = LLAttributeList::get(gIR->context(), {set, os});
  return *this;
}
