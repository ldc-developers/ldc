//===-- irdsymbol.cpp -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "gen/logger.h"
#include "ir/irdsymbol.h"
#include "ir/irvar.h"

// Callbacks for constructing/destructing Dsymbol.ir member.
void* newIrDsymbol() { return static_cast<void*>(new IrDsymbol()); }
void deleteIrDsymbol(void* sym) { delete static_cast<IrDsymbol*>(sym); }

std::vector<IrDsymbol *> IrDsymbol::list;

void IrDsymbol::resetAll() {
  Logger::println("resetting %llu Dsymbols",
                  static_cast<unsigned long long>(list.size()));

  for (auto s : list) {
    s->reset();
  }
}

IrDsymbol::IrDsymbol() : irData(nullptr) {
  list.push_back(this);
}

IrDsymbol::IrDsymbol(const IrDsymbol &s) {
  list.push_back(this);
  irData = s.irData;
  m_type = s.m_type;
  m_state = s.m_state;
}

IrDsymbol::~IrDsymbol() {
  if (this == list.back()) {
    list.pop_back();
    return;
  }

  auto it = std::find(list.rbegin(), list.rend(), this).base();
  // base() returns the iterator _after_ the found position
  list.erase(--it);
}

void IrDsymbol::reset() {
  irData = nullptr;
  m_type = Type::NotSet;
  m_state = State::Initial;
}

void IrDsymbol::setResolved() {
  if (m_state < Resolved) {
    m_state = Resolved;
  }
}

void IrDsymbol::setDeclared() {
  if (m_state < Declared) {
    m_state = Declared;
  }
}

void IrDsymbol::setDefined() {
  if (m_state < Defined) {
    m_state = Defined;
  }
}
