//===-- tools/runtime-adapt/tests/adapt_ast.d ---------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module tests.adapt_ast;

import adapt;
import compilerparse;
import parseutil;
import versions;

import std.algorithm : canFind;
import std.array : join;
import std.file : mkdirRecurse, tempDir, write;
import std.path : buildPath;

unittest
{
    enum obj = q"D
module object;
class Object { }
D";
    auto r = adaptSource(obj, "object.d", "v1.36.0", AdaptMode.overlay);
    assert(r.parsedIn && r.parsedOut, r.error);
    assert(r.applied.canFind("object.crt-gate"));
    assert(r.outFacts.moduleName == "object");
    assert(r.output.canFind("CRuntime_OVERLAY"));
}

unittest
{
    enum life = q"D
module rt.lifetime;
extern (C) void* _d_allocmemory(size_t sz)
{
    return GC.malloc(sz);
}
D";
    auto r = adaptSource(life, "rt/lifetime.d", "v1.36.0", AdaptMode.overlay);
    if (!r.parsedOut)
    {
        mkdirRecurse(buildPath(tempDir, "runtime-adapt-ut"));
        write(buildPath(tempDir, "runtime-adapt-ut", "adapt-fail.txt"),
            "applied=[" ~ r.applied.join(",") ~ "]\nerr=" ~ r.error ~ "\nout=\n" ~ r.output);
        assert(0);
    }
    assert(r.parsedIn);
    assert(r.applied.canFind("hook._d_allocmemory"), r.applied.join(","));
    assert(r.outFacts.moduleName == "rt.lifetime");
    assert(r.output.canFind("return null"));
}

unittest
{
    enum src = q"D
module sample;
version (DigitalMars) { enum int x = 1; }
extern (C) void already();
D";
    CompilerModel m;
    m.ok = true;
    m.definesLdc = true;
    m.runtimeHooks = ["_d_throw_exception"];
    // Empty presence lists: treat as unknown (unit test / single file).
    auto r = adaptSource(src, "object.d", "v1.36.0", AdaptMode.fromCompiler, &m);
    assert(r.parsedOut, r.error);
    // LDC does not predefine DigitalMars; do not assign it (std.compiler.Vendor,
    // DMD-only workarounds). Goal runtime leaves those version branches false.
    assert(!r.output.canFind("version = DigitalMars"));
    assert(!r.applied.canFind("version.LDC-defines-DigitalMars"));
    assert(r.output.canFind("_d_throw_exception"));
    assert(r.applied.canFind("hook-decl._d_throw_exception"));
}

unittest
{
    foreach (tag; consecutiveTags)
        assert(versioningHolds(tag) || tag == consecutiveTags[$ - 1]);
    assert(versioningHolds("v1.30.0"));
    assert(versioningHolds("v1.41.0"));
    // last tag has no successor; still must match itself
    auto last = matchConstraint(minorLadder, "v1.42.0");
    assert(last !is null && appliesTo(*last, "v1.42.0"));
    assert(!appliesTo(*last, "v1.41.0"));
}
