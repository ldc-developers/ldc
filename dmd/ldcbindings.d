//===-- ldcbindings.d -----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module dmd.ldcbindings;

import dmd.arraytypes;
import dmd.dsymbol;
import dmd.expression;
import dmd.globals;
import dmd.location;
import dmd.common.outbuffer;
import dmd.statement;
import dmd.tokens;

extern (C++):

Parameters* createParameters() { return new Parameters(); }
Expressions* createExpressions() { return new Expressions(); }

OutBuffer* createOutBuffer() { return new OutBuffer(); }

InlineAsmStatement createInlineAsmStatement(const ref Loc loc, Token* tokens) { return new InlineAsmStatement(loc, tokens); }
GccAsmStatement createGccAsmStatement(const ref Loc loc, Token* tokens) { return new GccAsmStatement(loc, tokens); }

Expression createExpressionForIntOp(const ref Loc loc, TOK op, Expression e1, Expression e2)
{
    switch (op)
    {
        case TOK.add:
            return e2 ? new AddExp(loc, e1, e2) : e1;
        case TOK.min:
            return e2 ? new MinExp(loc, e1, e2) : new NegExp(loc, e1);
        case TOK.mul:
            return new MulExp(loc, e1, e2);
        case TOK.div:
            return new DivExp(loc, e1, e2);
        case TOK.mod:
            return new ModExp(loc, e1, e2);
        case TOK.leftShift:
            return new ShlExp(loc, e1, e2);
        case TOK.rightShift:
            return new ShrExp(loc, e1, e2);
        case TOK.unsignedRightShift:
            return new UshrExp(loc, e1, e2);
        case TOK.not:
            return new NotExp(loc, e1);
        case TOK.tilde:
            return new ComExp(loc, e1);
        case TOK.orOr:
            return new LogicalExp(loc, EXP.orOr, e1, e2);
        case TOK.andAnd:
            return new LogicalExp(loc, EXP.andAnd, e1, e2);
        case TOK.or:
            return new OrExp(loc, e1, e2);
        case TOK.and:
            return new AndExp(loc, e1, e2);
        case TOK.xor:
            return new XorExp(loc, e1, e2);
        case TOK.equal:
            return new EqualExp(EXP.equal, loc, e1, e2);
        case TOK.notEqual:
            return new EqualExp(EXP.notEqual, loc, e1, e2);
        case TOK.greaterThan:
            return new CmpExp(EXP.greaterThan, loc, e1, e2);
        case TOK.greaterOrEqual:
            return new CmpExp(EXP.greaterOrEqual, loc, e1, e2);
        case TOK.lessThan:
            return new CmpExp(EXP.lessThan, loc, e1, e2);
        case TOK.lessOrEqual:
            return new CmpExp(EXP.lessOrEqual, loc, e1, e2);
        default:
            assert(0, "unknown integer operation");
    }
}

Expression createExpression(const ref Loc loc, EXP op) { return new Expression(loc, op); }
DsymbolExp createDsymbolExp(const ref Loc loc, Dsymbol s) { return new DsymbolExp(loc, s, /*hasOverloads=*/false); }
AddrExp createAddrExp(const ref Loc loc, Expression e) { return new AddrExp(loc, e); }
CommaExp createCommaExp(const ref Loc loc, Expression e1, Expression e2, bool generated = true) { return new CommaExp(loc, e1, e2, generated); }
