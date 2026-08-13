//===-- tools/runtime-adapt/source/kernel.d -----------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module kernel;

enum KernelGroup
{
    none,
    gc,
    thread,
    bootstrap,
    eh,
    os,
    libc,
    phobosIo,
    phobosConc,
    phobosMath,
    phobosOther,
    ctfeKeep,
    wasmNative, /// files LDC grew for stock WASI (sections_wasm, …)
}

KernelGroup classifyPath(string relPosix)
{
    import std.algorithm : canFind, endsWith, startsWith;
    auto p = relPosix;
    if (p == "rt/sections_wasm.d" || p.canFind("sections_wasm"))
        return KernelGroup.wasmNative;
    if (p.startsWith("gc/") || p == "core/memory.d" || p.startsWith("core/internal/gc/")
        || p.startsWith("core/gc/"))
        return KernelGroup.gc;
    if (p.startsWith("core/thread") || p.startsWith("core/sync/") || p == "core/thread.d")
        return KernelGroup.thread;
    if (p.canFind("/deh") || p.canFind("dwarfeh") || p.canFind("ehalloc")
        || p.canFind("eh_msvc") || p.canFind("backtrace/"))
        return KernelGroup.eh;
    if (p.startsWith("rt/dmain") || p.startsWith("rt/minfo") || p.startsWith("rt/sections")
        || p.startsWith("rt/dso") || p.startsWith("rt/monitor") || p.startsWith("rt/critical")
        || p == "rt/memory.d" || p == "core/runtime.d" || p == "core/internal/entrypoint.d")
        return KernelGroup.bootstrap;
    if (p.startsWith("core/sys/wasi/"))
        return KernelGroup.none;
    if (p.startsWith("core/sys/"))
        return KernelGroup.os;
    if (p.startsWith("core/stdc/") || p.startsWith("core/stdcpp/"))
        return KernelGroup.libc;
    if (p.startsWith("std/stdio") || p.startsWith("std/file") || p.startsWith("std/process")
        || p.startsWith("std/socket") || p.startsWith("std/net/") || p == "std/mmfile.d")
        return KernelGroup.phobosIo;
    if (p.startsWith("std/concurrency") || p.startsWith("std/parallelism"))
        return KernelGroup.phobosConc;
    if (p.startsWith("std/numeric") || p.startsWith("std/complex")
        || p.startsWith("std/mathspecial") || p.startsWith("std/internal/math/"))
        return KernelGroup.phobosMath;
    if (p.startsWith("std/algorithm") || p.startsWith("std/traits") || p.startsWith("std/meta")
        || p.startsWith("std/format") || p.startsWith("std/range") || p.startsWith("std/array")
        || p.startsWith("std/conv") || p.startsWith("std/typecons") || p.startsWith("std/utf")
        || p.startsWith("std/uni") || p.startsWith("std/functional") || p.startsWith("std/exception")
        || p.startsWith("std/ascii") || p.startsWith("std/bitmanip") || p.startsWith("std/math/")
        || p == "std/math.d" || p.startsWith("std/variant") || p.startsWith("std/typetuple")
        || p.startsWith("std/system"))
        return KernelGroup.ctfeKeep;
    if (p.startsWith("std/"))
        return KernelGroup.phobosOther;
    return KernelGroup.none;
}

string groupName(KernelGroup g)
{
    final switch (g)
    {
    case KernelGroup.none: return "none";
    case KernelGroup.gc: return "gc";
    case KernelGroup.thread: return "thread";
    case KernelGroup.bootstrap: return "bootstrap";
    case KernelGroup.eh: return "eh";
    case KernelGroup.os: return "os";
    case KernelGroup.libc: return "libc";
    case KernelGroup.phobosIo: return "phobos-io";
    case KernelGroup.phobosConc: return "phobos-conc";
    case KernelGroup.phobosMath: return "phobos-math";
    case KernelGroup.phobosOther: return "phobos-other";
    case KernelGroup.ctfeKeep: return "ctfe-keep";
    case KernelGroup.wasmNative: return "wasm-native";
    }
}
