//===-- ldcbindings.h -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "expression.h"
#include <cstdint>

class InlineAsmStatement;

using uint = uint32_t;

// Classes
IntegerExp *createIntegerExp(const Loc &loc, dinteger_t value, Type *type);
IntegerExp *createIntegerExp(dinteger_t value);
EqualExp *createEqualExp(TOK, const Loc &, Expression *, Expression *);
CmpExp *createCmpExp(TOK, const Loc &, Expression *, Expression *);
ShlExp *createShlExp(const Loc &, Expression *, Expression *);
ShrExp *createShrExp(const Loc &, Expression *, Expression *);
UshrExp *createUshrExp(const Loc &, Expression *, Expression *);
LogicalExp *createLogicalExp(const Loc &, TOK op, Expression *, Expression *);
OrExp *createOrExp(const Loc &, Expression *, Expression *);
AndExp *createAndExp(const Loc &, Expression *, Expression *);
XorExp *createXorExp(const Loc &, Expression *, Expression *);
ModExp *createModExp(const Loc &, Expression *, Expression *);
MulExp *createMulExp(const Loc &, Expression *, Expression *);
DivExp *createDivExp(const Loc &, Expression *, Expression *);
AddExp *createAddExp(const Loc &, Expression *, Expression *);
MinExp *createMinExp(const Loc &, Expression *, Expression *);
RealExp *createRealExp(const Loc &, real_t, Type *);
NotExp *createNotExp(const Loc &, Expression *);
ComExp *createComExp(const Loc &, Expression *);
NegExp *createNegExp(const Loc &, Expression *);
AddrExp *createAddrExp(const Loc &, Expression *);
DsymbolExp *createDsymbolExp(const Loc &, Dsymbol *, bool = false);
Expression *createExpression(const Loc &loc, TOK op, int size);
InlineAsmStatement *createInlineAsmStatement(const Loc &loc, Token *tokens);
TypeDelegate *createTypeDelegate(Type *t);
TypeIdentifier *createTypeIdentifier(const Loc &loc, Identifier *ident);

Strings *createStrings();

// Structs
//Loc createLoc(const char * filename, uint linnum, uint charnum);

/*
 * Define bindD<Type>::create(...) templated functions, to create D objects in templated code (class type is template parameter).
 * Used e.g. in toir.cpp
 */
template <class T> struct bindD {
  template <typename... Args> T *create(Args...) {
    assert(0 && "newD<> not implemented for this type");
  }
};
#define NEWD_TEMPLATE(T)                                                       \
  template <> struct bindD<T> {                                                \
    template <typename... Args> static T *create(Args... args) {               \
      return create##T(args...);                                               \
    }                                                                          \
  };
NEWD_TEMPLATE(ShlExp)
NEWD_TEMPLATE(ShrExp)
NEWD_TEMPLATE(UshrExp)
NEWD_TEMPLATE(LogicalExp)
NEWD_TEMPLATE(OrExp)
NEWD_TEMPLATE(AndExp)
NEWD_TEMPLATE(XorExp)
NEWD_TEMPLATE(ModExp)
NEWD_TEMPLATE(MulExp)
NEWD_TEMPLATE(DivExp)
NEWD_TEMPLATE(AddExp)
NEWD_TEMPLATE(MinExp)
