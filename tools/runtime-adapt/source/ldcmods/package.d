//===-- tools/runtime-adapt/source/ldcmods/package.d --------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Registry of ldc/* emitters. One row per runtime/druntime/src/ldc/ file.
// Adding a file LDC just added: drop source/ldcmods/<name>.d and append a
// row here. See EXTENDING.md.
//
//===----------------------------------------------------------------------===//

module ldcmods;

public import ldcmods.common;
public import ldcmods.implied : generateImpliedModules;

import compilerparse;
import ldcmods.attributes;
import ldcmods.dcompute;
import ldcmods.eh_msvc;
import ldcmods.eh_wasm;
import ldcmods.intrinsics;
import ldcmods.libfuzzer;
import ldcmods.llvmasm;
import ldcmods.native;
import ldcmods.opencl;
import ldcmods.profile;
import ldcmods.sanitizers;
import ldcmods.simd;

import std.algorithm : canFind;

/// Back-compat name used by emit.d.
alias GeneratedDi = GeneratedFile;

/// Kept so existing call sites compile; prefer parseCompiler.
struct GuideFacts
{
    bool ok;
    bool llvmIdentIsMajor;
    int llvmLo = 15;
    int llvmHi = 24;
    bool hasIntrinsic;
    bool hasInlineAsm;
    bool hasInlineIr;
    bool hasFence;
    bool hasAtomic;
    bool hasConvertVector;
    bool hasDcompute;
    bool hasProfile;
    bool hasFuzzer;
    bool hasWasm;
    string[] pragmas;
}

GuideFacts scanGuide(string guideRoot)
{
    auto m = parseCompiler(guideRoot);
    GuideFacts g;
    g.ok = m.ok;
    g.llvmIdentIsMajor = m.llvmIdentIsMajor;
    g.llvmLo = m.llvmLo;
    g.llvmHi = m.llvmHi;
    g.hasIntrinsic = hasPragma(m, "LDC_intrinsic");
    g.hasInlineAsm = hasPragma(m, "LDC_inline_asm");
    g.hasInlineIr = hasPragma(m, "LDC_inline_ir");
    g.hasFence = hasPragma(m, "LDC_fence");
    g.hasAtomic = hasPragma(m, "LDC_atomic_load");
    g.hasConvertVector = m.ldcIntrinsics.canFind("convertvector");
    g.hasDcompute = hasLdcModule(m, "dcompute");
    g.hasProfile = hasLdcModule(m, "profile") || hasPragma(m, "LDC_profile_instr");
    g.hasFuzzer = m.hasFuzzer || hasLdcModule(m, "libfuzzer");
    g.hasWasm = m.hasWasm;
    g.pragmas = m.pragmas;
    return g;
}

/// Table order matches a typical `ls runtime/druntime/src/ldc`.
static immutable LdcEmitter[] ldcEmitters = [
    LdcEmitter("ldc/attributes.d", "attributes", "gen/uda.cpp",
        &wantAttributes, &renderAttributes),
    LdcEmitter("ldc/dcompute.d", "dcompute", "driver/main.cpp",
        &wantDcompute, &renderDcompute),
    LdcEmitter("ldc/intrinsics.di", "intrinsics", "gen/pragma.cpp",
        &wantIntrinsics, &renderIntrinsics),
    LdcEmitter("ldc/llvmasm.di", "llvmasm", "gen/pragma.cpp",
        &wantLlvmAsm, &renderLlvmAsm),
    LdcEmitter("ldc/simd.di", "simd", "gen/pragma.cpp",
        &wantSimd, &renderSimd),
    LdcEmitter("ldc/opencl.di", "opencl", "gen/dcompute",
        &wantOpencl, &renderOpencl),
    LdcEmitter("ldc/profile.di", "profile", "gen/pragma.cpp",
        &wantProfile, &renderProfile),
    LdcEmitter("ldc/eh_msvc.d", "eh_msvc", "gen/runtime.cpp",
        &wantEhMsvc, &renderEhMsvc),
    LdcEmitter("ldc/eh_wasm.d", "eh_wasm", "gen/trycatchfinally.cpp",
        &wantEhWasm, &renderEhWasm),
    LdcEmitter("ldc/libfuzzer.di", "libfuzzer", "driver/cl_options_sanitizers.cpp",
        &wantLibfuzzer, &renderLibfuzzer),
    LdcEmitter("ldc/asan.d", "asan", "driver/cl_options_sanitizers.cpp",
        &wantAsan, &renderAsan),
    LdcEmitter("ldc/sanitizer_common.d", "sanitizer_common",
        "driver/cl_options_sanitizers.cpp", &wantSanitizerCommon, &renderSanitizerCommon),
    LdcEmitter("ldc/sanitizers_optionally_linked.d", "sanitizers_optionally_linked",
        "driver/cl_options_sanitizers.cpp", &wantSanitizersOpt, &renderSanitizersOpt),
    LdcEmitter("ldc/arm_unwind.c", "arm_unwind", "gen/runtime.cpp",
        &wantNative, &renderArmUnwind),
    LdcEmitter("ldc/eh_asm.S", "eh_asm", "runtime/CMakeLists.txt",
        &wantNative, &renderEhAsm),
    LdcEmitter("ldc/msvc.c", "msvc", "gen/runtime.cpp",
        &wantNative, &renderMsvcC),
];

GeneratedFile[] generateLdcDi(string guideRoot, string tag)
{
    return generateFromCompiler(guideRoot, tag);
}

/// Emit every ldc.* module the compiler model requires, then correct gaps.
GeneratedFile[] generateFromCompiler(string guideRoot, string tag)
{
    auto model = parseCompiler(guideRoot);
    if (!model.ok)
        return null;

    GeneratedFile[] outp;
    bool[string] seen;
    foreach (e; ldcEmitters)
    {
        if (e.rel in seen || !e.want(model))
            continue;
        seen[e.rel] = true;
        outp ~= GeneratedFile(e.rel, e.render(model, tag), "emit.ldc-" ~ e.name);
    }
    correctAgainstModel(outp, model, tag);
    return outp;
}

void correctAgainstModel(ref GeneratedFile[] files, const CompilerModel model, string tag)
{
    foreach (ref f; files)
    {
        if (f.rel == "ldc/attributes.d")
            correctAttributes(f, model);
        else if (f.rel == "ldc/intrinsics.di")
            correctIntrinsics(f, model);
        else if (f.rel == "ldc/dcompute.d")
            correctDcompute(f, model);
    }
}
