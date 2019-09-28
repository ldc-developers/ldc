/**
 * Compiler implementation of the
 * $(LINK2 http://www.dlang.org, D programming language).
 *
 * Copyright:   Copyright (C) 1999-2019 by The D Language Foundation, All Rights Reserved
 * Authors:     $(LINK2 http://www.digitalmars.com, Walter Bright)
 * License:     $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Source:      $(LINK2 https://github.com/dlang/dmd/blob/master/src/dmd/staticcond.d, _staticcond.d)
 * Documentation:  https://dlang.org/phobos/dmd_staticcond.html
 * Coverage:    https://codecov.io/gh/dlang/dmd/src/master/src/dmd/staticcond.d
 */

module dmd.staticcond;

import dmd.aliasthis;
import dmd.arraytypes;
import dmd.dmodule;
import dmd.dscope;
import dmd.dsymbol;
import dmd.errors;
import dmd.expression;
import dmd.expressionsem;
import dmd.globals;
import dmd.identifier;
import dmd.mtype;
import dmd.root.array;
import dmd.root.outbuffer;
import dmd.tokens;
import dmd.utils;



/********************************************
 * Semantically analyze and then evaluate a static condition at compile time.
 * This is special because short circuit operators &&, || and ?: at the top
 * level are not semantically analyzed if the result of the expression is not
 * necessary.
 * Params:
 *      sc  = instantiating scope
 *      original = original expression, for error messages
 *      e =  resulting expression
 *      errors = set to `true` if errors occurred
 *      negatives = array to store negative clauses
 * Returns:
 *      true if evaluates to true
 */
bool evalStaticCondition(Scope* sc, Expression original, Expression e, out bool errors, Expressions* negatives = null)
{
    if (negatives)
        negatives.setDim(0);

    bool impl(Expression e)
    {
        if (e.op == TOK.not)
        {
            NotExp ne = cast(NotExp)e;
            return !impl(ne.e1);
        }

        if (e.op == TOK.andAnd || e.op == TOK.orOr)
        {
            LogicalExp aae = cast(LogicalExp)e;
            bool result = impl(aae.e1);
            if (errors)
                return false;
            if (e.op == TOK.andAnd)
            {
                if (!result)
                    return false;
            }
            else
            {
                if (result)
                    return true;
            }
            result = impl(aae.e2);
            return !errors && result;
        }

        if (e.op == TOK.question)
        {
            CondExp ce = cast(CondExp)e;
            bool result = impl(ce.econd);
            if (errors)
                return false;
            Expression leg = result ? ce.e1 : ce.e2;
            result = impl(leg);
            return !errors && result;
        }

        Expression before = e;
        const uint nerrors = global.errors;

        sc = sc.startCTFE();
        sc.flags |= SCOPE.condition;

        e = e.expressionSemantic(sc);
        e = resolveProperties(sc, e);
        e = e.toBoolean(sc);

        sc = sc.endCTFE();
        e = e.optimize(WANTvalue);

        if (nerrors != global.errors ||
            e.op == TOK.error ||
            e.type.toBasetype() == Type.terror)
        {
            errors = true;
            return false;
        }

        e = resolveAliasThis(sc, e);

        if (!e.type.isBoolean())
        {
            original.error("expression `%s` of type `%s` does not have a boolean value",
                original.toChars(), e.type.toChars());
            errors = true;
            return false;
        }

        e = e.ctfeInterpret();

        if (e.isBool(true))
            return true;
        else if (e.isBool(false))
        {
            if (negatives)
                negatives.push(before);
            return false;
        }

        e.error("expression `%s` is not constant", e.toChars());
        errors = true;
        return false;
    }
    return impl(e);
}

/********************************************
 * Format a static condition as a tree-like structure, marking failed and
 * bypassed expressions.
 * Params:
 *      original = original expression
 *      instantiated = instantiated expression
 *      negatives = array with negative clauses from `instantiated` expression
 *      full = controls whether it shows the full output or only failed parts
 *      itemCount = returns the number of written clauses
 * Returns:
 *      formatted string or `null` if the expressions were `null`, or if the
 *      instantiated expression is not based on the original one
 */
const(char)* visualizeStaticCondition(Expression original, Expression instantiated,
    const Expression[] negatives, bool full, ref uint itemCount)
{
    if (!original || !instantiated || original.loc !is instantiated.loc)
        return null;

    OutBuffer buf;

    if (full)
        itemCount = visualizeFull(original, instantiated, negatives, buf);
    else
        itemCount = visualizeShort(original, instantiated, negatives, buf);

    return buf.extractChars();
}

private uint visualizeFull(Expression original, Expression instantiated,
    const Expression[] negatives, ref OutBuffer buf)
{
    // tree-like structure; traverse and format simultaneously
    uint count;
    uint indent;

    static void printOr(uint indent, ref OutBuffer buf)
    {
        buf.reserve(indent * 4 + 8);
        foreach (i; 0 .. indent)
            buf.writestring("    ");
        buf.writestring("    or:\n");
    }

    // returns true if satisfied
    bool impl(Expression orig, Expression e, bool inverted, bool orOperand, bool unreached)
    {
        TOK op = orig.op;

        // lower all 'not' to the bottom
        // !(A && B) -> !A || !B
        // !(A || B) -> !A && !B
        if (inverted)
        {
            if (op == TOK.andAnd)
                op = TOK.orOr;
            else if (op == TOK.orOr)
                op = TOK.andAnd;
        }

        if (op == TOK.not)
        {
            NotExp no = cast(NotExp)orig;
            NotExp ne = cast(NotExp)e;
            assert(ne);
            return impl(no.e1, ne.e1, !inverted, orOperand, unreached);
        }
        else if (op == TOK.andAnd)
        {
            BinExp bo = cast(BinExp)orig;
            BinExp be = cast(BinExp)e;
            assert(be);
            const r1 = impl(bo.e1, be.e1, inverted, false, unreached);
            const r2 = impl(bo.e2, be.e2, inverted, false, unreached || !r1);
            return r1 && r2;
        }
        else if (op == TOK.orOr)
        {
            if (!orOperand) // do not indent A || B || C twice
                indent++;
            BinExp bo = cast(BinExp)orig;
            BinExp be = cast(BinExp)e;
            assert(be);
            const r1 = impl(bo.e1, be.e1, inverted, true, unreached);
            printOr(indent, buf);
            const r2 = impl(bo.e2, be.e2, inverted, true, unreached);
            if (!orOperand)
                indent--;
            return r1 || r2;
        }
        else if (op == TOK.question)
        {
            CondExp co = cast(CondExp)orig;
            CondExp ce = cast(CondExp)e;
            assert(ce);
            if (!inverted)
            {
                // rewrite (A ? B : C) as (A && B || !A && C)
                if (!orOperand)
                    indent++;
                const r1 = impl(co.econd, ce.econd, inverted, false, unreached);
                const r2 = impl(co.e1, ce.e1, inverted, false, unreached || !r1);
                printOr(indent, buf);
                const r3 = impl(co.econd, ce.econd, !inverted, false, unreached);
                const r4 = impl(co.e2, ce.e2, inverted, false, unreached || !r3);
                if (!orOperand)
                    indent--;
                return r1 && r2 || r3 && r4;
            }
            else
            {
                // rewrite !(A ? B : C) as (!A || !B) && (A || !C)
                if (!orOperand)
                    indent++;
                const r1 = impl(co.econd, ce.econd, inverted, false, unreached);
                printOr(indent, buf);
                const r2 = impl(co.e1, ce.e1, inverted, false, unreached);
                const r12 = r1 || r2;
                const r3 = impl(co.econd, ce.econd, !inverted, false, unreached || !r12);
                printOr(indent, buf);
                const r4 = impl(co.e2, ce.e2, inverted, false, unreached || !r12);
                if (!orOperand)
                    indent--;
                return (r1 || r2) && (r3 || r4);
            }
        }
        else // 'primitive' expression
        {
            buf.reserve(indent * 4 + 4);
            foreach (i; 0 .. indent)
                buf.writestring("    ");

            // find its value; it may be not computed, if there was a short circuit,
            // but we handle this case with `unreached` flag
            bool value = true;
            if (!unreached)
            {
                foreach (fe; negatives)
                {
                    if (fe is e)
                    {
                        value = false;
                        break;
                    }
                }
            }
            // write the marks first
            const satisfied = inverted ? !value : value;
            if (!satisfied && !unreached)
                buf.writestring("  > ");
            else if (unreached)
                buf.writestring("  - ");
            else
                buf.writestring("    ");
            // then the expression itself
            if (inverted)
                buf.writeByte('!');
            buf.writestring(orig.toChars);
            buf.writenl();
            count++;
            return satisfied;
        }
    }

    impl(original, instantiated, false, true, false);
    return count;
}

private uint visualizeShort(Expression original, Expression instantiated,
    const Expression[] negatives, ref OutBuffer buf)
{
    // simple list; somewhat similar to long version, so no comments
    // one difference is that it needs to hold items to display in a stack

    static struct Item
    {
        Expression orig;
        bool inverted;
    }

    Array!Item stack;

    bool impl(Expression orig, Expression e, bool inverted)
    {
        TOK op = orig.op;

        if (inverted)
        {
            if (op == TOK.andAnd)
                op = TOK.orOr;
            else if (op == TOK.orOr)
                op = TOK.andAnd;
        }

        if (op == TOK.not)
        {
            NotExp no = cast(NotExp)orig;
            NotExp ne = cast(NotExp)e;
            assert(ne);
            return impl(no.e1, ne.e1, !inverted);
        }
        else if (op == TOK.andAnd)
        {
            BinExp bo = cast(BinExp)orig;
            BinExp be = cast(BinExp)e;
            assert(be);
            bool r = impl(bo.e1, be.e1, inverted);
            r = r && impl(bo.e2, be.e2, inverted);
            return r;
        }
        else if (op == TOK.orOr)
        {
            BinExp bo = cast(BinExp)orig;
            BinExp be = cast(BinExp)e;
            assert(be);
            const lbefore = stack.length;
            bool r = impl(bo.e1, be.e1, inverted);
            r = r || impl(bo.e2, be.e2, inverted);
            if (r)
                stack.setDim(lbefore); // purge added positive items
            return r;
        }
        else if (op == TOK.question)
        {
            CondExp co = cast(CondExp)orig;
            CondExp ce = cast(CondExp)e;
            assert(ce);
            if (!inverted)
            {
                const lbefore = stack.length;
                bool a = impl(co.econd, ce.econd, inverted);
                a = a && impl(co.e1, ce.e1, inverted);
                bool b;
                if (!a)
                {
                    b = impl(co.econd, ce.econd, !inverted);
                    b = b && impl(co.e2, ce.e2, inverted);
                }
                const r = a || b;
                if (r)
                    stack.setDim(lbefore);
                return r;
            }
            else
            {
                bool a;
                {
                    const lbefore = stack.length;
                    a = impl(co.econd, ce.econd, inverted);
                    a = a || impl(co.e1, ce.e1, inverted);
                    if (a)
                        stack.setDim(lbefore);
                }
                bool b;
                if (a)
                {
                    const lbefore = stack.length;
                    b = impl(co.econd, ce.econd, !inverted);
                    b = b || impl(co.e2, ce.e2, inverted);
                    if (b)
                        stack.setDim(lbefore);
                }
                return a && b;
            }
        }
        else // 'primitive' expression
        {
            bool value = true;
            foreach (fe; negatives)
            {
                if (fe is e)
                {
                    value = false;
                    break;
                }
            }
            const satisfied = inverted ? !value : value;
            if (!satisfied)
                stack.push(Item(orig, inverted));
            return satisfied;
        }
    }

    impl(original, instantiated, false);

    foreach (i; 0 .. stack.length)
    {
        // write the expression only
        buf.writestring("       ");
        if (stack[i].inverted)
            buf.writeByte('!');
        buf.writestring(stack[i].orig.toChars);
        // here with no trailing newline
        if (i + 1 < stack.length)
            buf.writenl();
    }
    return cast(uint)stack.length;
}
