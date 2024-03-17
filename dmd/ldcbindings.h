//===-- ldcbindings.h -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "root/array.h"
#include "tokens.h"

class AddrExp;
class CommaExp;
class Dsymbol;
class DsymbolExp;
class Expression;
class GccAsmStatement;
class InlineAsmStatement;
struct OutBuffer;
class Parameter;

Array<Parameter *> *createParameters();
Array<Expression *> *createExpressions();

OutBuffer *createOutBuffer();

// for gen/asmstmt.cpp only:
InlineAsmStatement *createInlineAsmStatement(const Loc &loc, Token *tokens);
GccAsmStatement *createGccAsmStatement(const Loc &loc, Token *tokens);

// for gen/asm-x86.h only:
Expression *createExpressionForIntOp(const Loc &loc, TOK op, Expression *e1, Expression *e2);
Expression *createExpression(const Loc &loc, EXP op);
DsymbolExp *createDsymbolExp(const Loc &loc, Dsymbol *s);
AddrExp *createAddrExp(const Loc &loc, Expression *e);

// for gen/toir.cpp only:
CommaExp *createCommaExp(const Loc &loc, Expression *e1, Expression *e2, bool generated = true);
