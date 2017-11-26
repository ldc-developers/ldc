//===-- gen/ldctraits.d - LDC-specific __traits handling ----------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module gen.ldctraits;

import ddmd.arraytypes;
import ddmd.dscope;
import ddmd.dtemplate;
import ddmd.expression;
import ddmd.expressionsem;
import ddmd.errors;
import ddmd.id;
import ddmd.mtype;

extern(C++) struct Dstring
{
    size_t length;
    const(char)* ptr;
}

extern(C++) Dstring traitsGetTargetCPU();
extern(C++) bool traitsTargetHasFeature(Dstring feature);

Expression semanticTraitsLDC(TraitsExp e, Scope* sc)
{
    size_t arg_count = e.args ? e.args.dim : 0;

    if (e.ident == Id.targetCPU)
    {
        if (arg_count != 0)
        {
            e.warning("ignoring arguments for __traits %s", e.ident.toChars());
        }

        auto cpu = traitsGetTargetCPU();
        auto se = new StringExp(e.loc, cast(void*)cpu.ptr, cpu.length);
        return se.expressionSemantic(sc);
    }
    if (e.ident == Id.targetHasFeature)
    {
        if (arg_count != 1)
        {
            e.error("__traits %s expects one argument, not %u", e.ident.toChars(), cast(uint)arg_count);
            return new ErrorExp();
        }

        auto ex = isExpression((*e.args)[0]);
        if (!ex)
        {
            e.error("expression expected as argument of __traits %s", e.ident.toChars());
            return new ErrorExp();
        }
        ex = ex.ctfeInterpret();

        StringExp se = ex.toStringExp();
        if (!se || se.len == 0)
        {
            e.error("string expected as argument of __traits %s instead of %s", e.ident.toChars(), ex.toChars());
            return new ErrorExp();
        }

        se = se.toUTF8(sc);
        auto featureFound = traitsTargetHasFeature(Dstring(se.len, se.toPtr()));
        return new IntegerExp(e.loc, featureFound ? 1 : 0, Type.tbool);
    }
    return null;
}
