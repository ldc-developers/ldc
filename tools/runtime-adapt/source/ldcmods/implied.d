//===-- tools/runtime-adapt/source/ldcmods/implied.d --------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Stock modules the frontend imports (dmd/imphint.d) that official druntime
// may lack: core.interpolation, __importc_builtins, core.stdc.stdatomic.
//
//===----------------------------------------------------------------------===//

module ldcmods.implied;

import compilerparse;
import ldcmods.common;

import std.array : appender;

GeneratedFile[] generateImpliedModules(const CompilerModel model, bool[string] have,
    string tag)
{
    GeneratedFile[] outp;
    foreach (mod; model.implicitModules)
    {
        if (mod.length >= 4 && (mod[0 .. 4] == "ldc." || mod[0 .. 4] == "std."))
            continue;
        auto rel = moduleToRel(mod);
        if (!rel.length || rel in have)
            continue;
        GeneratedFile f;
        f.rel = rel;
        f.principleId = "emit.implied-" ~ mod;
        if (mod == "core.interpolation")
            f.body = renderInterpolation(model, tag);
        else if (mod == "__importc_builtins" || mod == "importc_builtins"
            || rel == "__importc_builtins.di")
            f.body = banner(tag) ~ "module __importc_builtins;\n";
        else if (mod == "core.stdc.stdatomic")
            f.body = banner(tag) ~ "module core.stdc.stdatomic;\n";
        else
            continue;
        outp ~= f;
    }
    return outp;
}

private string renderInterpolation(const CompilerModel model, string tag)
{
    auto buf = appender!string();
    buf.put(banner(tag));
    buf.put("module core.interpolation;\n\n");
    bool[string] have;
    foreach (sym, mod; model.implicitSymbols)
    {
        if (mod != "core.interpolation" || sym in have)
            continue;
        have[sym] = true;
        buf.put("struct ");
        buf.put(sym);
        if (sym == "InterpolatedLiteral" || sym == "InterpolatedExpression")
            buf.put("(string s)");
        buf.put(" { }\n");
    }
    if (!have.length)
    {
        buf.put("struct InterpolationHeader { }\n");
        buf.put("struct InterpolationFooter { }\n");
        buf.put("struct InterpolatedLiteral(string s) { }\n");
        buf.put("struct InterpolatedExpression(string s) { }\n");
    }
    return buf.data;
}
