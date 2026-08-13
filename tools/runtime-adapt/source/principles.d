//===-- tools/runtime-adapt/source/principles.d -------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Main goal: the official reference druntime/phobos is the input; the
// complete LDC druntime/phobos is the output. Equivalence is produced by
// *this source*, not by a one-off dump. Each principle names an LDC
// frontend/runtime locus and the action generate must take. Closing a
// residual means adding or tightening a rule here (and the splice in
// adapt.d), then re-running iterate.
//
//===----------------------------------------------------------------------===//

module principles;

import kernel;
import parseutil;

enum EmitAction
{
    omit,
    takeReference, /// stock body at --reference
    takeLdc,       /// unused on native emit (LDC runtime is the goal, not a source)
    astOverlay,    /// adapt.d splices (CRT gate, DtoThrow hook, …)
    generateFromCompiler, /// ldc/*.di from driver/ + gen/ at the workspace tag
}

/// native = produce the complete LDC runtime (default generate).
/// strip  = honor omit.kernel / overlay stubs (optional shrink, not the product).
enum EmitProfile
{
    native,
    strip,
}

struct Principle
{
    string id;
    string locus;     /// path:line in this LDC tree
    string symbol;
    string why;
    bool function(string rel, KernelGroup g, ParseFacts refF, ParseFacts ldcF) match;
    EmitAction action;
}

private bool isHookFile(string rel)
{
    return rel == "object.d" || rel == "rt/lifetime.d" || rel == "core/exception.d"
        || rel == "core/memory.d";
}

private bool kernelOmit(string rel, KernelGroup g, ParseFacts, ParseFacts)
{
    if (isHookFile(rel))
        return false;
    return g == KernelGroup.thread || g == KernelGroup.os
        || g == KernelGroup.eh || g == KernelGroup.gc
        || g == KernelGroup.phobosIo || g == KernelGroup.phobosConc
        || g == KernelGroup.bootstrap;
}

private bool frontendHook(string rel, KernelGroup, ParseFacts, ParseFacts)
{
    return isHookFile(rel);
}

private bool ctfeRebase(string rel, KernelGroup g, ParseFacts, ParseFacts ldcF)
{
    return g == KernelGroup.ctfeKeep && ldcF.parsed;
}

private bool wasmNativeTake(string rel, KernelGroup g, ParseFacts, ParseFacts)
{
    return g == KernelGroup.wasmNative || rel == "rt/sections_wasm.d";
}

private bool defaultTakeRef(string, KernelGroup, ParseFacts, ParseFacts)
{
    return true;
}

private bool frontendNewFile(string rel, KernelGroup, ParseFacts refF, ParseFacts)
{
    // Present only on the LDC side; frontend started emitting these.
    if (refF.parsed)
        return false;
    return rel == "core/interpolation.d" || rel == "__importc_builtins.di"
        || rel == "core/stdc/stdatomic.d" || rel == "rt/invariant_.d"
        || rel == "core/internal/cast_.d";
}

private bool remapFiber(string rel, KernelGroup, ParseFacts, ParseFacts)
{
    return rel == "core/thread/fiber.d";
}

/// Reference path → LDC path after known in-tree moves.
string remapRel(string refRel)
{
    if (refRel == "core/thread/fiber.d")
        return "core/thread/fiber/package.d";
    if (refRel == "rt/invariant.d")
        return "rt/invariant_.d";
    if (refRel == "__builtins.di")
        return "__importc_builtins.di";
    if (refRel == "core/internal/vararg/aarch64.d")
        return "core/internal/vararg/aapcs64.d";
    return refRel;
}

/// LDC path → reference path (inverse of remapRel).
string inverseRemap(string ldcRel)
{
    if (ldcRel == "core/thread/fiber/package.d")
        return "core/thread/fiber.d";
    if (ldcRel == "rt/invariant_.d")
        return "rt/invariant.d";
    if (ldcRel == "__importc_builtins.di")
        return "__builtins.di";
    if (ldcRel == "core/internal/vararg/aapcs64.d")
        return "core/internal/vararg/aarch64.d";
    return ldcRel;
}

/// Ordered. First match wins. Edit this table to change generated output.
immutable Principle[] principleTable = [
    Principle("omit.kernel",
        "runtime/druntime/src/core/thread/osthread.d:1; rt/dmain2.d; rt/minfo.d",
        "osthread/_d_run_main/ModuleInfo",
        "Frontend does not require OS threads or _d_run_main when -fno-moduleinfo; omit.",
        &kernelOmit, EmitAction.omit),
    Principle("take.sections-wasm",
        "runtime/druntime/src/rt/sections_wasm.d:38-47; rt/sections.d:72-73",
        "initSections/__minfo",
        "LDC wasm ModuleInfo. Native emit still writes the reference body; this names the locus.",
        &wasmNativeTake, EmitAction.takeReference),
    Principle("take.frontend-new",
        "dmd/cond.d interpolation; importc builtins; core/stdc/stdatomic.d",
        "core.interpolation/__importc_builtins",
        "Newer frontend implicit imports. Only applies when the reference already has the file.",
        &frontendNewFile, EmitAction.takeReference),
    Principle("remap.fiber",
        "core/thread/fiber.d → core/thread/fiber/package.d (post-1.36 split)",
        "fiber",
        "LDC split fiber.d; omit the old path (kernel/thread) and take new files via take.frontend-new/omit.",
        &remapFiber, EmitAction.omit),
    Principle("ast.frontend-hooks",
        "gen/llvmhelpers.cpp:333-341 DtoThrow; rt/lifetime.d:69-72; core/exception.d:897-900; driver/main.cpp:936-939",
        "_d_throw_exception/_d_allocmemory/onAssertErrorMsg/CRuntime_WASI",
        "Hooks the frontend emits. Splice via adapt.d (source of truth).",
        &frontendHook, EmitAction.astOverlay),
    Principle("take.ctfe",
        "runtime/phobos/std/traits.d:1",
        "std.traits",
        "CTFE/frontend imports; native emit writes the reference body.",
        &ctfeRebase, EmitAction.takeReference),
    Principle("take.reference",
        "runtime/druntime/src/object.d",
        "stock",
        "Unmatched keepers come from the reference tag.",
        &defaultTakeRef, EmitAction.takeReference),
];

const(Principle)* selectPrinciple(string rel, KernelGroup g,
    ParseFacts refF, ParseFacts ldcF, EmitProfile profile = EmitProfile.native)
{
    foreach (ref p; principleTable)
    {
        // Native product is the full LDC runtime: do not strip kernel
        // files and do not apply overlay-stub AST splices.
        if (profile == EmitProfile.native
            && (p.action == EmitAction.omit || p.action == EmitAction.astOverlay))
            continue;
        if (p.match(rel, g, refF, ldcF))
            return &p;
    }
    return null;
}

EmitAction selectAction(string rel, KernelGroup g, ParseFacts refF, ParseFacts ldcF,
    EmitProfile profile = EmitProfile.native)
{
    auto p = selectPrinciple(rel, g, refF, ldcF, profile);
    return p is null ? EmitAction.takeReference : p.action;
}
