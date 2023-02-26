//===-- gen/ldctraits.d - LDC-specific __traits handling ----------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module gen.ldctraits;

import dmd.arraytypes;
import dmd.dscope;
import dmd.dtemplate;
import dmd.expression;
import dmd.expressionsem;
import dmd.errors;
import dmd.id;
import dmd.mtype;

extern(C++) struct Dstring
{
    size_t length;
    const(char)* ptr;
}

extern(C++) Dstring traitsGetTargetCPU();
extern(C++) bool traitsTargetHasFeature(Dstring feature);

Expression semanticTraitsLDC(TraitsExp e, Scope* sc)
{
    size_t arg_count = e.args ? e.args.length : 0;

    if (e.ident == Id.targetCPU)
    {
        if (arg_count != 0)
        {
            e.warning("ignoring arguments for __traits %s", e.ident.toChars());
        }

        auto cpu = traitsGetTargetCPU();
        auto se = new StringExp(e.loc, cpu.ptr[0 .. cpu.length]);
        return se.expressionSemantic(sc);
    }
    if (e.ident == Id.targetHasFeature)
    {
        if (arg_count != 1)
        {
            e.error("__traits %s expects one argument, not %u", e.ident.toChars(), cast(uint)arg_count);
            return ErrorExp.get();
        }

        auto ex = isExpression((*e.args)[0]);
        if (!ex)
        {
            e.error("expression expected as argument of __traits %s", e.ident.toChars());
            return ErrorExp.get();
        }
        ex = ex.ctfeInterpret();

        StringExp se = ex.toStringExp();
        if (!se || se.len == 0)
        {
            e.error("string expected as argument of __traits %s instead of %s", e.ident.toChars(), ex.toChars());
            return ErrorExp.get();
        }

        se = se.toUTF8(sc);
        auto str = se.peekString();
        auto featureFound = traitsTargetHasFeature(Dstring(str.length, str.ptr));
        return new IntegerExp(e.loc, featureFound ? 1 : 0, Type.tbool);
    }
    return null;
}
