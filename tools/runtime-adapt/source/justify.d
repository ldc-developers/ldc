//===-- tools/runtime-adapt/source/justify.d ----------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Line-cited LDC concepts that *prompt* a druntime/phobos adaptation.
// Reconstruct mode copies stock files from --reference and overlays the
// adapted bodies; these notes explain why a file is copied, omitted, or
// kept as an overlay instead of taking stock LDC.
//
// Line numbers are against this checkout (master / v1.43.0-beta1-3).
//
//===----------------------------------------------------------------------===//

module justify;

import kernel;

struct Locus
{
    string path;
    uint line;
    uint lineEnd; /// inclusive; 0 means same as line
    string symbol;
    string snippet;
    string why;
}

struct FileHint
{
    string rel; /// adapted/reference relative path
    string prompt;
}

struct Justification
{
    KernelGroup group;
    string conceptName;
    string summary;
    Locus[] loci;
    FileHint[] fileHints;
}

immutable Justification[] catalog = [
    Justification(KernelGroup.eh, "DtoThrow / wasm exception model",
        "Every D `throw` becomes a call to `_d_throw_exception` and then "
        ~ "`unreachable`. There is no in-tree wasm personality; dwarf/msvc EH "
        ~ "modules are the wrong ABI. An adapted runtime therefore *replaces* "
        ~ "`_d_throw_exception` (host/JS) and *omits* `rt/dwarfeh.d`, "
        ~ "`rt/deh_*.d`, `rt/ehalloc.d`. LLVM 21–22 also forces "
        ~ "`ExceptionModel = Wasm` and `+exception-handling`, which is why "
        ~ "the overlay keeps `--wasm-enable-eh` consumers and assert-shaped "
        ~ "JS abort instead of Itanium unwind.",
        [
            Locus("gen/llvmhelpers.cpp", 333, 341, "DtoThrow",
                "getRuntimeFunction(..., \"_d_throw_exception\"); CreateCallOrInvoke; CreateUnreachable",
                "Frontend throw lowering — the only EH hook an adapted tree must define"),
            Locus("gen/runtime.cpp", 712, 714, "_d_throw_exception",
                "createFwdDecl(..., {\"_d_throw_exception\"}, {throwableTy}, {}, Attr_Cold_NoReturn)",
                "Compiler-rt forward decl; missing symbol is a link error, not a silent no-op"),
            Locus("driver/cl_options.cpp", 1031, 1031, "wasm-enable-eh",
                "\"wasm-enable-eh\" in the LLVM option forward list",
                "User-facing switch that asks LLVM for wasm EH opcodes"),
            Locus("driver/targetmachine.cpp", 630, 640, "triple.isWasm",
                "features.push_back(\"+exception-handling\"); ExceptionModel = Wasm (LLVM 21–22)",
                "This checkout already enables wasm EH on every wasm triple"),
            Locus("runtime/druntime/src/ldc/intrinsics.di", 739, 740, "llvm_wasm_throw",
                "pragma(LDC_intrinsic, \"llvm.wasm.throw\") void llvm_wasm_throw(uint tag, void* ex)",
                "Newest-LDC intrinsic for a real wasm throw; overlay may call this later"),
            Locus("runtime/druntime/src/core/exception.d", 665, 670, "onAssertErrorMsg",
                "extern (C) void onAssertErrorMsg(string file, size_t line, string msg)",
                "Stock body throws AssertError; adapted abort() under WASI imports this from JS"),
            Locus("runtime/druntime/src/core/exception.d", 897, 900, "_d_assert_msg",
                "void _d_assert_msg(...) { onAssertErrorMsg(file, line, msg); }",
                "Assert lowering — the model JS EH stubs must copy (decode + abort)"),
        ],
        [
            FileHint("object.d", "Keep overlay object.d when it asserts a non-stock CRT; do not take stock object.d"),
            FileHint("rt/lifetime.d", "Keep overlay: nothrow / no BlkInfo GC"),
            FileHint("ldc/eh_msvc.d", "version (CRuntime_Microsoft) — unused on wasm; copy overlay or omit"),
            FileHint("rt/dwarfeh.d", "OMIT — dwarf personality, not wasm EH"),
            FileHint("rt/deh_win64_posix.d", "OMIT — Win64 SEH"),
        ]),
    Justification(KernelGroup.bootstrap, "ModuleInfo / _start vs sections_wasm",
        "Stock LDC wasm emits `__moduleRef` into section `__minfo` and "
        ~ "`rt.sections_wasm.initSections` walks `__start___minfo` .. "
        ~ "`__stop___minfo`. An adapted SPA supplies `mixin Spa!_start` and "
        ~ "compiles with `-fno-moduleinfo`, so `rt/minfo.d`, `rt/dmain2.d`, "
        ~ "and `rt/sections*.d` must be omitted or they pull a GC scan of "
        ~ "`__global_base` .. `__data_end`.",
        [
            Locus("runtime/druntime/src/rt/sections.d", 72, 73, "version (WebAssembly)",
                "else version (WebAssembly) public import rt.sections_wasm;",
                "Stock selector — adapted trees never import this file"),
            Locus("runtime/druntime/src/rt/sections_wasm.d", 38, 47, "initSections",
                "ModuleInfo** __start___minfo .. __stop___minfo; GC range __global_base .. __data_end",
                "Native wasm ModuleInfo + conservative GC roots"),
            Locus("tests/codegen/wasm.d", 21, 23, "__moduleRef",
                "CHECK: section \"__minfo\"",
                "Frontend proof that ModuleInfo is emitted unless -fno-moduleinfo"),
        ],
        [
            FileHint("rt/sections_wasm.d", "OMIT while -fno-moduleinfo remains"),
            FileHint("rt/minfo.d", "OMIT — ModuleGroup registry"),
            FileHint("rt/dmain2.d", "OMIT — _d_run_main; overlay uses exported _start"),
            FileHint("core/runtime.d", "OMIT — rt_init / module ctors"),
        ]),
    Justification(KernelGroup.wasmNative, "sections_wasm is LDC-native, not an overlay",
        "`rt/sections_wasm.d` exists only in newer LDC (this checkout). It is "
        ~ "not in a v1.36.0 reference and must not be copied into an overlay "
        ~ "that still uses `-fno-moduleinfo`. Reconstruct from reference "
        ~ "therefore omits it automatically.",
        [
            Locus("runtime/druntime/src/rt/sections_wasm.d", 1, 3, "rt.sections_wasm",
                "module rt.sections_wasm; version (WebAssembly):",
                "Target-only file; never produced from a v1.36.0 reference"),
        ],
        [
            FileHint("rt/sections_wasm.d", "OMIT in reconstruct; stub-or-port only if overlay drops -fno-moduleinfo"),
        ]),
    Justification(KernelGroup.gc, "No wasm GC; frontend still calls GC hooks",
        "`_d_allocmemory` is `GC.malloc`. WasmPointersSpill exists because a "
        ~ "conservative GC cannot see wasm locals. An adapted tree stubs "
        ~ "`gc_malloc` / `_d_allocmemory` to a bump/`memory.grow` allocator "
        ~ "and leaves JS `gc_*` as no-ops. Reconstruct keeps the overlay "
        ~ "`core/memory.d` and `rt/lifetime.d`; omits `gc/` and "
        ~ "`core/internal/gc/`.",
        [
            Locus("runtime/druntime/src/rt/lifetime.d", 69, 72, "_d_allocmemory",
                "extern (C) void* _d_allocmemory(size_t sz) @weak { return GC.malloc(sz); }",
                "Closure/class allocation the frontend emits — must not call a real GC"),
            Locus("runtime/druntime/src/rt/lifetime.d", 81, 84, "_d_allocmemoryT",
                "GC.malloc(ti.tsize(), ... BlkAttr.NO_SCAN)",
                "LDC-only POD allocate; overlay redirects the same symbol"),
            Locus("gen/passes/WasmPointersSpill.cpp", 15, 32, "WasmPointersSpill",
                "conservative GC blocked on wasm locals / value stack; spill live pointers across calls",
                "Why stock GC is not a drop-in; overlay uses a non-scanning bump instead"),
        ],
        [
            FileHint("core/memory.d", "Keep overlay (GC types + hooks, no collector)"),
            FileHint("rt/lifetime.d", "Keep overlay"),
            FileHint("gc/gc.d", "OMIT entire gc/"),
        ]),
    Justification(KernelGroup.os, "CRuntime_WASI vs overlay CRT",
        "A `wasm32-unknown-wasi` triple predefines `WebAssembly`, `WASI`, "
        ~ "`CRuntime_WASI`. Stock `object.d` follows that CRT. An overlay "
        ~ "that `static assert`s a different `CRuntime_*` is a different "
        ~ "object module: reconstruct must copy the overlay `object.d`, never "
        ~ "the reference one, and must omit `core/sys/posix/**` and "
        ~ "`core/sys/windows/**`.",
        [
            Locus("driver/main.cpp", 745, 748, "wasm32/wasm64",
                "VersionCondition::addPredefinedGlobalIdent(\"WebAssembly\")",
                "Always on for the wasm arch, independent of CRT"),
            Locus("driver/main.cpp", 936, 939, "Triple::WASI",
                "WASI, WASIp1, Posix, CRuntime_WASI",
                "Stock CRT id — overlay must override with -d-version=<its CRuntime_*>"),
            Locus("dmd/target.d", 278, 278, "TargetC.Runtime.WASI",
                "case WASI: return predef(\"CRuntime_WASI\")",
                "Frontend CRT switch"),
            Locus("dmd/cond.d", 377, 377, "CRuntime_WASI",
                "case \"CRuntime_WASI\":",
                "Recognized version identifier"),
            Locus("ldc2.conf.header", 43, 50, "^wasm(32|64)-",
                "switches ~= [ \"-defaultlib=\" ]; lib-dirs = []",
                "Stock package unhooks defaultlibs; it does *not* unhook import/object.d"),
        ],
        [
            FileHint("object.d", "Keep overlay — non-stock CRuntime_* static assert"),
            FileHint("core/sys/wasi/config.d", "Keep extra — not in stock LDC 1.36"),
            FileHint("core/sys/posix/pthread.d", "OMIT"),
        ]),
    Justification(KernelGroup.thread, "No osthread on wasm",
        "`core.thread.osthread` is pthread/Win32. Wasm overlay is "
        ~ "single-threaded; reconstruct omits the whole group.",
        [
            Locus("runtime/druntime/src/core/thread/osthread.d", 1, 1, "osthread",
                "OS thread attach / TLS",
                "No wasm implementation in-tree"),
        ],
        [
            FileHint("core/thread/osthread.d", "OMIT"),
            FileHint("core/sync/mutex.d", "OMIT"),
        ]),
    Justification(KernelGroup.libc, "core.stdc is the C ABI the frontend assumes",
        "Keep the `core/stdc` subset that is identical to the reference; "
        ~ "keep overlay bodies where errno/math/time were patched. Do not "
        ~ "pull `core/stdcpp`.",
        [
            Locus("runtime/druntime/src/core/stdc/config.d", 1, 1, "c_long",
                "ABI integers",
                "Frontend and CTFE size_t/c_long"),
        ],
        [
            FileHint("core/stdc/errno.d", "Keep overlay if it differs"),
            FileHint("core/stdc/math.d", "Keep overlay if it differs"),
        ]),
    Justification(KernelGroup.phobosIo, "stdio/file/socket need a kernel",
        "Stock Phobos already versions some socket paths out under "
        ~ "`CRuntime_WASI`. The overlay omits stdio/file/process/socket "
        ~ "entirely. Reconstruct omits them so the produced tree matches.",
        [
            Locus("runtime/phobos/std/socket.d", 321, 342, "CRuntime_WASI",
                "getaddrinfo support is rather incomplete",
                "Even stock WASI Phobos is not a full socket story"),
        ],
        [
            FileHint("std/stdio.d", "OMIT"),
            FileHint("std/file.d", "OMIT"),
            FileHint("std/socket.d", "OMIT"),
        ]),
    Justification(KernelGroup.phobosConc, "concurrency needs threads + GC",
        "Omit `std.concurrency` / `std.parallelism` so reconstruct matches "
        ~ "the overlay (they are not present there).",
        [
            Locus("runtime/phobos/std/concurrency.d", 1, 1, "Tid",
                "message passing on OS threads",
                "No wasm stop-the-world"),
        ],
        [
            FileHint("std/concurrency.d", "OMIT"),
        ]),
    Justification(KernelGroup.phobosMath, "Accidental runtime import, not a frontend hook",
        "LDC does not lower anything to `std.numeric`. The overlay currently "
        ~ "omits `std.complex` / `std.numeric` / `std.mathspecial` — "
        ~ "reconstruct must omit them too so the tree stays nearly exact. "
        ~ "Do not copy them from the reference to “help” a later compile.",
        [
            Locus("runtime/phobos/std/numeric.d", 1, 1, "std.numeric",
                "pulls std.internal.math.gammafunction",
                "Not a frontend hook; still omit in reconstruct if the overlay lacks it"),
        ],
        [
            FileHint("std/numeric.d", "OMIT — not in overlay"),
            FileHint("std/complex.d", "OMIT — not in overlay"),
            FileHint("std/mathspecial.d", "OMIT — not in overlay"),
        ]),
    Justification(KernelGroup.ctfeKeep, "CTFE Phobos must come from the reference when identical",
        "`std.algorithm` / `traits` / `meta` / `format` / `range` are needed "
        ~ "at compile time. If the overlay file matches the reference, "
        ~ "reconstruct emits the *reference* body (same bytes after newline "
        ~ "normalize). That is how the produced tree is “from” v1.36.0.",
        [
            Locus("runtime/phobos/std/traits.d", 1, 1, "std.traits",
                "imported by frontend CTFE and user mixins",
                "Rebase-from-reference when identical"),
        ],
        [
            FileHint("std/traits.d", "copy-reference if identical; else keep overlay"),
        ]),
    Justification(KernelGroup.none, "Allocator intrinsics + extras",
        "`llvm_wasm_memory_grow` / `memory_size` are why a bump allocator "
        ~ "can exist without a kernel. Extras (`core/sys/wasi`, gccbuiltins) "
        ~ "are overlay-only — copy them verbatim so reconstruct stays exact.",
        [
            Locus("runtime/druntime/src/ldc/intrinsics.di", 716, 737, "llvm_wasm_memory_grow/size",
                "pragma(LDC_intrinsic, \"llvm.wasm.memory.grow.i32\")",
                "Page grow the overlay WasmAllocator calls"),
            Locus("gen/abi/wasm.cpp", 10, 10, "BasicCABI",
                "WebAssembly tool-conventions BasicCABI.md",
                "Why size_t is 32-bit on wasm32 — overlay object.d aliases"),
            Locus("driver/linker.cpp", 106, 109, "extension wasm",
                "wasm32/wasm64 output extension is .wasm",
                "Not a runtime file, but the emit contract the overlay links against"),
        ],
        [
            FileHint("core/sys/wasi/config.d", "copy-adapted extra"),
            FileHint("ldc/gccbuiltins_x86.di", "copy-adapted extra (frontend implicit import)"),
        ]),
    Justification(KernelGroup.phobosOther, "Unreferenced Phobos stays omitted",
        "Reconstruct omits any `std/*` that the overlay never shipped. "
        ~ "Copying them from the reference would *not* reproduce the adapted tree.",
        [
            Locus("runtime/phobos/std/", 0, 0, "std.*",
                "stock Phobos",
                "No wasm lowering"),
        ],
        [
            FileHint("std/random.d", "OMIT unless present in overlay"),
        ]),
];

const(Justification)* forGroup(KernelGroup g)
{
    foreach (ref j; catalog)
        if (j.group == g)
            return &j;
    return null;
}

string justifyText(KernelGroup g)
{
    auto j = forGroup(g);
    return j is null ? "" : j.summary;
}

string justifyLociMarkdown(KernelGroup g)
{
    import std.array : appender;
    import std.format : format;
    auto j = forGroup(g);
    if (j is null)
        return "";
    auto buf = appender!string();
    foreach (loc; j.loci)
    {
        auto end = loc.lineEnd ? loc.lineEnd : loc.line;
        buf.put(format("- `%s:%s-%s` **%s**: %s\n  - %s\n",
            loc.path, loc.line, end, loc.symbol, loc.snippet, loc.why));
    }
    return buf.data;
}

string filePrompt(string rel, KernelGroup g)
{
    auto j = forGroup(g);
    if (j !is null)
    {
        foreach (h; j.fileHints)
            if (h.rel == rel)
                return h.prompt;
    }
    return justifyText(g);
}
