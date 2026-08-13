//===-- tools/runtime-adapt/source/ldcmods/llvmasm.d --------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ldc/llvmasm.di ← gen/pragma.cpp LDC_inline_asm / LDC_inline_ir.
//
//===----------------------------------------------------------------------===//

module ldcmods.llvmasm;

import compilerparse;
import ldcmods.common;

import std.array : appender;

bool wantLlvmAsm(const CompilerModel m)
{
    return hasPragma(m, "LDC_inline_asm") || hasPragma(m, "LDC_inline_ir")
        || hasLdcModule(m, "llvmasm");
}

string renderLlvmAsm(const CompilerModel model, string tag)
{
    auto buf = appender!string();
    buf.put(banner(tag));
    buf.put("module ldc.llvmasm;\n\nstruct __asmtuple_t(T...) { T v; }\n\n");
    if (hasPragma(model, "LDC_inline_asm"))
    {
        buf.put("pragma(LDC_inline_asm)\n{\n");
        buf.put("    void __asm()(const(char)[] asmcode, const(char)[] constraints, ...) pure nothrow @nogc;\n");
        buf.put("    T __asm(T)(const(char)[] asmcode, const(char)[] constraints, ...) pure nothrow @nogc;\n");
        buf.put("    void __asm_trusted()(const(char)[] asmcode, const(char)[] constraints, ...) @trusted pure nothrow @nogc;\n");
        buf.put("    T __asm_trusted(T)(const(char)[] asmcode, const(char)[] constraints, ...) @trusted pure nothrow @nogc;\n");
        buf.put("    template __asmtuple(T...)\n    {\n");
        buf.put("        __asmtuple_t!(T) __asmtuple(const(char)[] asmcode, const(char)[] constraints, ...);\n");
        buf.put("    }\n}\n\n");
    }
    if (hasPragma(model, "LDC_inline_ir"))
    {
        buf.put("pragma(LDC_inline_ir)\n    R __ir(string s, R, P...)(P params) @trusted nothrow @nogc;\n");
        buf.put("pragma(LDC_inline_ir)\n    R __ir_pure(string s, R, P...)(P params) @trusted nothrow @nogc pure;\n");
        buf.put("pragma(LDC_inline_ir)\n    R __irEx(string prefix, string code, string suffix, R, P...)(P) @trusted nothrow @nogc;\n");
        buf.put("pragma(LDC_inline_ir)\n    R __irEx_pure(string prefix, string code, string suffix, R, P...)(P) @trusted nothrow @nogc pure;\n");
    }
    return buf.data;
}
