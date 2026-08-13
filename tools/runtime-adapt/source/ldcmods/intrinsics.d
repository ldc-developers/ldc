//===-- tools/runtime-adapt/source/ldcmods/intrinsics.d -----------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ldc/intrinsics.di ← gen/pragma.cpp (LDC_intrinsic) + LLVM version ladder.
// Typical LDC commits:
//   * "Add support for LLVM N to intrinsics.di" → llvmLo/llvmHi from CMakeLists
//   * "Add llvm_X to ldc.intrinsics" → scanned from gen/*.cpp "llvm.…" strings
//
//===----------------------------------------------------------------------===//

module ldcmods.intrinsics;

import compilerparse;
import ldcmods.common;

import std.algorithm : canFind, endsWith;
import std.array : appender;
import std.format : format;

bool wantIntrinsics(const CompilerModel m)
{
    return hasPragma(m, "LDC_intrinsic") || hasLdcModule(m, "intrinsics");
}

string renderIntrinsics(const CompilerModel model, string tag)
{
    auto buf = appender!string();
    buf.put(banner(tag));
    buf.put("module ldc.intrinsics;\n\n");
    buf.put("version (LDC) {}\nelse static assert(false, \"This module is only valid for LDC\");\n\n");
    buf.put(renderVersionIdents(model));
    buf.put("\nnothrow:\n@nogc:\n\n");
    buf.put("enum AtomicOrdering {\n");
    buf.put("    NotAtomic = 0, Unordered = 1, Monotonic = 2, Consume = 3,\n");
    buf.put("    Acquire = 4, Release = 5, AcquireRelease = 6, SequentiallyConsistent = 7\n}\n");
    buf.put("alias DefaultOrdering = AtomicOrdering.SequentiallyConsistent;\n");
    buf.put("enum SynchronizationScope { SingleThread = 0, CrossThread = 1, Default = CrossThread }\n");
    buf.put("struct OverflowRet(T) { T result; bool overflow; }\n\n");

    string[] names = model.llvmIntrinsics.dup;
    foreach (f; publicFallbackIntrinsics)
        if (!names.canFind(f))
            names ~= f;

    bool[string] emittedD;
    foreach (it; names)
    {
        if (!looksLikePublicIntrinsic(it))
            continue;
        auto dname = intrinsicIdent(it);
        if (dname in emittedD)
            continue;
        emittedD[dname] = true;
        buf.put("pragma(LDC_intrinsic, \"");
        buf.put(it);
        buf.put("\")\n    ");
        buf.put(intrinsicSignature(it, dname));
        buf.put(";\n\n");
    }

    if (hasPragma(model, "LDC_fence"))
        buf.put("pragma(LDC_fence)\n    void llvm_memory_fence(AtomicOrdering ordering = DefaultOrdering,\n"
            ~ "        SynchronizationScope syncScope = SynchronizationScope.Default);\n\n");
    if (hasPragma(model, "LDC_atomic_load"))
    {
        buf.put("pragma(LDC_atomic_load)\n    T llvm_atomic_load(T)(in shared T* ptr, AtomicOrdering ordering = DefaultOrdering);\n\n");
        buf.put("pragma(LDC_atomic_store)\n    void llvm_atomic_store(T)(T val, shared T* ptr, AtomicOrdering ordering = DefaultOrdering);\n\n");
        buf.put("pragma(LDC_atomic_cmp_xchg)\n    CmpXchgResult!T llvm_atomic_cmp_xchg(T)(shared T* ptr, T cmp, T val,\n"
            ~ "        AtomicOrdering successOrdering = DefaultOrdering,\n"
            ~ "        AtomicOrdering failureOrdering = DefaultOrdering, bool weak = false);\n\n");
        buf.put("struct CmpXchgResult(T) { T previousValue; bool exchanged; }\n\n");
        foreach (op; ["xchg", "add", "sub", "and", "nand", "or", "xor", "max", "min", "umax", "umin"])
        {
            buf.put("pragma(LDC_atomic_rmw, \"");
            buf.put(op);
            buf.put("\")\n    T llvm_atomic_rmw_");
            buf.put(op);
            buf.put("(T)(shared T* ptr, T val, AtomicOrdering ordering = DefaultOrdering);\n\n");
        }
    }
    if (model.hasWasm)
    {
        buf.put("pragma(LDC_intrinsic, \"llvm.wasm.memory.grow.i32\")\n    int llvm_wasm_memory_grow(int mem, int delta);\n\n");
        buf.put("pragma(LDC_intrinsic, \"llvm.wasm.memory.size.i32\")\n    int llvm_wasm_memory_size(int mem);\n\n");
        buf.put("pragma(LDC_intrinsic, \"llvm.wasm.throw\")\n    void llvm_wasm_throw(uint tag, void* exn);\n\n");
    }
    return buf.data;
}

void correctIntrinsics(ref GeneratedFile f, const CompilerModel model)
{
    foreach (name; model.llvmIntrinsics)
    {
        if (f.body.canFind(name) || !looksLikePublicIntrinsic(name))
            continue;
        f.body ~= format("\npragma(LDC_intrinsic, \"%s\")\n    void %s();\n",
            name, intrinsicIdent(name));
        f.corrections ~= "add-intrinsic:" ~ name;
    }
    foreach (name; model.ldcIntrinsics)
    {
        if (f.body.canFind(name))
            continue;
        f.body ~= format("\npragma(LDC_intrinsic, \"ldc.%s\")\n    T llvm_%s(T, S)(S src);\n",
            name, sanitizeIdent(name));
        f.corrections ~= "add-ldc-intrinsic:" ~ name;
    }
}

bool looksLikePublicIntrinsic(string name)
{
    if (name.length < 6 || name[0 .. 5] != "llvm.")
        return false;
    if (name.canFind(" ") || name.canFind("%") || name.canFind("("))
        return false;
    if (name[$ - 1] == '.')
        return false;
    if (name.canFind("llvm.ldc.") || name == "llvm.compiler.used"
        || name == "llvm.ident" || name == "llvm.linker.options"
        || name == "llvm.metadata" || name == "llvm.dbg" || name == "llvm.dbg.cu"
        || name == "llvm.used")
        return false;
    if ((name.endsWith(".f32") || name.endsWith(".f64") || name.endsWith(".f16")
            || name.endsWith(".i8") || name.endsWith(".i16"))
        && !name.canFind("wasm"))
        return false;
    return true;
}

string intrinsicIdent(string llvmName)
{
    auto s = llvmName;
    if (s.length >= 5 && s[0 .. 5] == "llvm.")
        s = s[5 .. $];
    static immutable suffixes = [".p0", ".p1", ".p2", ".i#", ".f#", ".i32", ".i64"];
    bool changed = true;
    while (changed)
    {
        changed = false;
        foreach (suf; suffixes)
            if (s.length > suf.length && s[$ - suf.length .. $] == suf)
            {
                s = s[0 .. $ - suf.length];
                changed = true;
            }
    }
    return "llvm_" ~ sanitizeIdent(s);
}

private string renderVersionIdents(const CompilerModel m)
{
    auto buf = appender!string();
    buf.put("     ");
    bool first = true;
    foreach (maj; m.llvmLo .. m.llvmHi + 1)
    {
        void one(int verId, string enumName, int enumVal)
        {
            if (!first)
                buf.put("else ");
            first = false;
            buf.put(format("version (LDC_LLVM_%s) enum %s = %s;\n", verId, enumName, enumVal));
        }
        if (m.llvmIdentIsMajor)
            one(maj, "LLVM_major", maj);
        else
        {
            one(maj * 100, "LLVM_version", maj * 100);
            one(maj * 100 + 1, "LLVM_version", maj * 100 + 1);
        }
    }
    buf.put("else static assert(false, \"LDC LLVM version not supported\");\n\n");
    if (m.llvmIdentIsMajor)
        buf.put("enum LLVM_atleast(int major) = LLVM_major >= major;\n");
    else
        buf.put("enum LLVM_atleast(int major) = (LLVM_version >= major * 100);\n");
    return buf.data;
}

private string intrinsicSignature(string llvmName, string dname)
{
    if (llvmName.canFind("memcpy") || llvmName.canFind("memmove"))
        return "void " ~ dname ~ "(T)(void* dst, const(void)* src, T len, bool volatile_ = false)";
    if (llvmName.canFind("memset"))
        return "void " ~ dname ~ "(T)(void* dst, ubyte val, T len, bool volatile_ = false)";
    if (llvmName.canFind("returnaddress") || llvmName.canFind("frameaddress"))
        return "void* " ~ dname ~ "(uint level)";
    if (llvmName.canFind("stacksave") || llvmName.canFind("stackaddress")
        || llvmName.canFind("thread.pointer"))
        return "void* " ~ dname ~ "()";
    if (llvmName.canFind("stackrestore"))
        return "void " ~ dname ~ "(void* ptr)";
    if (llvmName.canFind("prefetch"))
        return "void " ~ dname ~ "(const(void)* ptr, uint rw, uint locality, uint cachetype)";
    if (llvmName.canFind("readcyclecounter"))
        return "ulong " ~ dname ~ "()";
    if (llvmName.canFind("trap") || llvmName.canFind("debugtrap") || llvmName.canFind("sideeffect"))
        return "void " ~ dname ~ "()";
    if (llvmName.canFind("assume"))
        return "void " ~ dname ~ "(bool cond)";
    if (llvmName.canFind("expect"))
        return "T " ~ dname ~ "(T)(T val, T expected_val)";
    if (llvmName.canFind("ctlz") || llvmName.canFind("cttz"))
        return "T " ~ dname ~ "(T)(T val, bool is_zero_undef)";
    if (llvmName.canFind("fshl") || llvmName.canFind("fshr") || llvmName.canFind("fma")
        || llvmName.canFind("fmuladd"))
        return "T " ~ dname ~ "(T)(T a, T b, T c)";
    if (llvmName.canFind("pow") || llvmName.canFind("copysign") || llvmName.canFind("minnum")
        || llvmName.canFind("maxnum") || llvmName.canFind("minimum") || llvmName.canFind("maximum"))
        return "T " ~ dname ~ "(T)(T a, T b)";
    if (llvmName.canFind(".f#") || llvmName.canFind(".i#") || llvmName.canFind("bswap")
        || llvmName.canFind("ctpop") || llvmName.canFind("bitreverse")
        || llvmName.canFind("fabs") || llvmName.canFind("sqrt") || llvmName.canFind("floor"))
        return "T " ~ dname ~ "(T)(T val)";
    if (llvmName.canFind("with.overflow") || llvmName.canFind(".sat."))
        return "OverflowRet!(T) " ~ dname ~ "(T)(T a, T b)";
    return "void " ~ dname ~ "()";
}

/// Public names LDC always exposes even if a given tag's gen/ scan is thin.
private static immutable publicFallbackIntrinsics = [
    "llvm.returnaddress", "llvm.frameaddress.p0", "llvm.stacksave.p0",
    "llvm.stackrestore.p0", "llvm.prefetch.p0", "llvm.readcyclecounter",
    "llvm.memcpy.p0.p0.i#", "llvm.memmove.p0.p0.i#", "llvm.memset.p0.i#",
    "llvm.sqrt.f#", "llvm.sin.f#", "llvm.cos.f#", "llvm.pow.f#", "llvm.powi.f#",
    "llvm.exp.f#", "llvm.exp2.f#", "llvm.log.f#", "llvm.log2.f#", "llvm.log10.f#",
    "llvm.fabs.f#", "llvm.floor.f#", "llvm.ceil.f#", "llvm.trunc.f#",
    "llvm.rint.f#", "llvm.nearbyint.f#", "llvm.round.f#", "llvm.copysign.f#",
    "llvm.fma.f#", "llvm.fmuladd.f#", "llvm.minnum.f#", "llvm.maxnum.f#",
    "llvm.minimum.f#", "llvm.maximum.f#",
    "llvm.bswap.i#", "llvm.ctpop.i#", "llvm.ctlz.i#", "llvm.cttz.i#",
    "llvm.bitreverse.i#", "llvm.fshl.i#", "llvm.fshr.i#",
    "llvm.trap", "llvm.debugtrap", "llvm.expect.i#", "llvm.assume",
    "llvm.clear_cache", "llvm.pcmarker", "llvm.sideeffect",
    "llvm.thread.pointer", "llvm.stacksave", "llvm.stackrestore",
    "llvm.sadd.with.overflow.i#", "llvm.uadd.with.overflow.i#",
    "llvm.ssub.with.overflow.i#", "llvm.usub.with.overflow.i#",
    "llvm.smul.with.overflow.i#", "llvm.umul.with.overflow.i#",
    "llvm.sadd.sat.i#", "llvm.uadd.sat.i#", "llvm.ssub.sat.i#", "llvm.usub.sat.i#",
];
