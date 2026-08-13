//===-- tools/runtime-adapt/source/ldcmods/attributes.d -----------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ldc/attributes.d ← gen/uda.cpp (Id::uda* / attribute structs).
// Typical LDC commit: add a UDA in uda.cpp, then a struct here.
//
//===----------------------------------------------------------------------===//

module ldcmods.attributes;

import compilerparse;
import ldcmods.common;

import std.algorithm : canFind;
import std.array : appender;
import std.format : format;

bool wantAttributes(const CompilerModel m)
{
    return m.attrNames.length > 0 || hasLdcModule(m, "attributes");
}

string renderAttributes(const CompilerModel model, string tag)
{
    auto buf = appender!string();
    buf.put(banner(tag));
    buf.put("module ldc.attributes;\n\n");
    buf.put("private template AliasSeq(TList...)\n{\n    alias AliasSeq = TList;\n}\n\n");
    bool[string] emittedPub;
    foreach (n; model.attrNames)
    {
        if (!n.length || n == "gnuAbiTag" || n == "selector" || n == "optional"
            || n == "mustuse" || n == "standalone" || n == "swift"
            || n == "compute" || n == "kernel" || n == "_kernel")
            continue;
        auto pub = (n.length && n[0] == '_') ? n[1 .. $] : n;
        if (pub in emittedPub)
            continue;
        emittedPub[pub] = true;
        buf.put(attrDecl(n, model));
        buf.put("\n\n");
    }
    if (buf.data.canFind("struct llvmAttr"))
    {
        buf.put("immutable naked = llvmAttr(\"naked\");\n");
        buf.put("immutable restrict = llvmAttr(\"noalias\");\n");
        buf.put("immutable cold = llvmAttr(\"cold\");\n");
        buf.put("immutable noplt = llvmAttr(\"nonlazybind\");\n");
    }
    return buf.data;
}

void correctAttributes(ref GeneratedFile f, const CompilerModel model)
{
    foreach (n; model.attrNames)
    {
        if (!n.length || n == "gnuAbiTag" || n == "selector" || n == "optional"
            || n == "mustuse" || n == "standalone" || n == "swift"
            || n == "compute" || n == "kernel" || n == "_kernel")
            continue;
        auto pub = (n.length && n[0] == '_') ? n[1 .. $] : n;
        if (f.body.canFind(n) || f.body.canFind("enum " ~ pub)
            || f.body.canFind("struct " ~ pub))
            continue;
        f.body ~= "\n" ~ attrDecl(n, model) ~ "\n";
        f.corrections ~= "add-attr:" ~ n;
    }
}

private string attrDecl(string name, const CompilerModel model)
{
    string[] fields;
    foreach (sh; model.attrShapes)
        if (sh.name == name || sh.name == "_" ~ name)
            fields = sh.fieldTypes.dup;
    if (!fields.length)
    {
        if (name == "callingConvention" || name == "llvmFastMathFlag"
            || name == "noSanitize" || name == "optStrategy" || name == "section"
            || name == "target")
            fields = ["string"];
        else if (name == "llvmAttr")
            fields = ["string", "string"];
        else if (name == "allocSize")
            fields = ["int", "int"];
    }
    if (name.length && name[0] == '_')
        return "immutable " ~ name[1 .. $] ~ " = " ~ name ~ "();\nprivate struct " ~ name ~ " { }";
    auto names = attrFieldNames(name, fields.length);
    if (!fields.length)
        return "struct " ~ name ~ " { }";
    auto buf = appender!string();
    buf.put("struct ");
    buf.put(name);
    buf.put("\n{\n");
    foreach (i, ty; fields)
    {
        buf.put("    ");
        buf.put(ty);
        buf.put(" ");
        buf.put(names[i]);
        if (name == "allocSize" && names[i] == "numArgIdx")
            buf.put(" = int.min");
        buf.put(";\n");
    }
    buf.put("}");
    return buf.data;
}

private string[] attrFieldNames(string attr, size_t n)
{
    string[] names;
    if (attr == "allocSize")
        names = ["sizeArgIdx", "numArgIdx"];
    else if (attr == "callingConvention")
        names = ["convention"];
    else if (attr == "llvmAttr")
        names = ["key", "value"];
    else if (attr == "llvmFastMathFlag")
        names = ["flag"];
    else if (attr == "noSanitize")
        names = ["sanitizerName"];
    else if (attr == "optStrategy")
        names = ["strategy"];
    else if (attr == "section")
        names = ["name"];
    else if (attr == "target")
        names = ["specifier"];
    else
        foreach (i; 0 .. n)
            names ~= format("_%s", i);
    while (names.length < n)
        names ~= format("_%s", names.length);
    return names[0 .. n];
}
