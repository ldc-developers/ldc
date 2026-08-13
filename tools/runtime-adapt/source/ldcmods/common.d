//===-- tools/runtime-adapt/source/ldcmods/common.d ---------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Shared types for one ldc/* emitter. Mirrors runtime/druntime/src/ldc/.
//
//===----------------------------------------------------------------------===//

module ldcmods.common;

import compilerparse;

import std.algorithm : canFind, endsWith;
import std.array : appender;
import std.format : format;

struct GeneratedFile
{
    string rel;
    string body;
    string principleId;
    string[] corrections;
}

/// One row in the ldc/* table — same idea as runtime/CMakeLists.txt listing
/// a druntime file, plus the gen/driver locus an LDC commit would touch.
struct LdcEmitter
{
    string rel;     /// runtime/druntime/src/ldc/<file>
    string name;    /// ldc.<name>
    string locus;   /// compiler file an LDC patch usually edits first
    bool function(const CompilerModel) want;
    string function(const CompilerModel, string) render;
}

string banner(string tag)
{
    return "// runtime-adapt " ~ tag
        ~ ": generated from parsed LDC compiler sources (dmd/, gen/, driver/).\n"
        ~ "// Goal runtime is not copied.\n\n";
}

string sanitizeIdent(string s)
{
    auto buf = appender!string();
    foreach (ch; s)
    {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
            || (ch >= '0' && ch <= '9'))
            buf.put(ch);
        else
            buf.put('_');
    }
    return buf.data;
}

bool wantAlways(const CompilerModel)
{
    return true;
}

bool wantIfNamed(const CompilerModel m, string name)
{
    return hasLdcModule(m, name);
}

string renderCStub(string tag, string why)
{
    return "/* runtime-adapt " ~ tag ~ ": " ~ why ~ "\n"
        ~ " * Generated from compiler references; not copied from LDC runtime.\n */\n";
}

string renderStubModule(string ldcName, string tag)
{
    return banner(tag) ~ "module ldc." ~ ldcName ~ ";\n\n"
        ~ "// Compiler referenced ldc." ~ ldcName ~ "; fill from gen/driver.\n";
}
