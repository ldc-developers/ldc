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
  return {LLAttributeList::get(gIR->context(), old.getFnAttributes(),
                               old.getRetAttributes(), {})};
}

AttrSet &AttrSet::add(unsigned index, const llvm::AttrBuilder &builder) {
  if (builder.hasAttributes()) {
    set = set.addAttributes(gIR->context(), index, builder);
  }
  return *this;
}

AttrSet &AttrSet::merge(const AttrSet &other) {
  auto &os = other.set;
  set = LLAttributeList::get(gIR->context(), {set, os});
  return *this;
}
