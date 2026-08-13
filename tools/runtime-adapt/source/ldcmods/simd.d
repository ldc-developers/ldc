//===-- tools/runtime-adapt/source/ldcmods/simd.d -----------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// ldc/simd.di ← gen/pragma.cpp LDC_inline_ir (shuffle/extract/insert).
//
//===----------------------------------------------------------------------===//

module ldcmods.simd;

import compilerparse;
import ldcmods.common;

bool wantSimd(const CompilerModel m)
{
    return hasPragma(m, "LDC_inline_ir") || hasLdcModule(m, "simd");
}

string renderSimd(const CompilerModel, string tag)
{
    return banner(tag) ~ q"D
module ldc.simd;

import core.simd;
import ldc.llvmasm;

pure:
nothrow:
@nogc:
@trusted:

private template llvmType(T)
{
    static if (is(T == float)) enum llvmType = "float";
    else static if (is(T == double)) enum llvmType = "double";
    else static if (is(T == byte) || is(T == ubyte) || is(T == void)) enum llvmType = "i8";
    else static if (is(T == short) || is(T == ushort)) enum llvmType = "i16";
    else static if (is(T == int) || is(T == uint)) enum llvmType = "i32";
    else static if (is(T == long) || is(T == ulong)) enum llvmType = "i64";
    else static assert(0, "Can't determine llvm type for D type " ~ T.stringof);
}

private template isFloatingPoint(T)
{
    enum isFloatingPoint = is(T == float) || is(T == double) || is(T == real);
}
private template isIntegral(T)
{
    enum isIntegral = is(T == byte) || is(T == ubyte) || is(T == short)
        || is(T == ushort) || is(T == int) || is(T == uint)
        || is(T == long) || is(T == ulong);
}
private template isSigned(T)
{
    enum isSigned = is(T == byte) || is(T == short) || is(T == int) || is(T == long);
}
private template IntOf(T)
if (isIntegral!T || isFloatingPoint!T)
{
    static if (T.sizeof == 1) alias byte IntOf;
    else static if (T.sizeof == 2) alias short IntOf;
    else static if (T.sizeof == 4) alias int IntOf;
    else static if (T.sizeof == 8) alias long IntOf;
    else static assert(0, "Type not supported");
}
private template BaseType(V) { alias typeof(V.array[0]) BaseType; }
private template numElements(V) { enum numElements = V.sizeof / BaseType!(V).sizeof; }
private template llvmVecType(V)
{
    static if (is(V == void16)) enum llvmVecType = "<16 x i8>";
    else static if (is(V == void32)) enum llvmVecType = "<32 x i8>";
    else enum llvmVecType = "<" ~ numElements!V.stringof ~ " x " ~ llvmType!(BaseType!V) ~ ">";
}

pragma(LDC_inline_ir)
    R inlineIR(string s, R, P...)(P);

template extractelement(V, int i)
if (is(typeof(llvmVecType!V)) && i < numElements!V)
{
    enum ir = `
        %r = extractelement ` ~ llvmVecType!V ~ ` %0, i32 ` ~ i.stringof ~ `
        ret ` ~ llvmType!(BaseType!V) ~ ` %r`;
    alias __ir_pure!(ir, BaseType!V, V) extractelement;
}

template insertelement(V, int i)
if (is(typeof(llvmVecType!V)) && i < numElements!V)
{
    enum ir = `
        %r = insertelement ` ~ llvmVecType!V ~ ` %0, ` ~ llvmType!(BaseType!V) ~ ` %1, i32 ` ~ i.stringof ~ `
        ret ` ~ llvmVecType!V ~ ` %r`;
    alias __ir_pure!(ir, V, V, BaseType!V) insertelement;
}

template shufflevector(V, mask...)
if (is(typeof(llvmVecType!V)) && mask.length == numElements!V)
{
    enum int n = mask.length;
    enum llvmV = llvmVecType!V;
    template genMaskIr(string ir, m...)
    {
        static if (m.length == 0)
            enum genMaskIr = ir;
        else
            enum genMaskIr = genMaskIr!(ir ~ ", i32 " ~ m[0].stringof, m[1 .. $]);
    }
    enum maskIr = genMaskIr!("", mask)[2 .. $];
    enum ir = `
        %r = shufflevector ` ~ llvmV ~ ` %0, ` ~ llvmV ~ ` %1, <` ~ n.stringof ~ ` x i32> <` ~ maskIr ~ `>
        ret ` ~ llvmV ~ ` %r`;
    alias __ir_pure!(ir, V, V, V) shufflevector;
}

template loadUnaligned(V)
if (is(typeof(llvmVecType!V)))
{
    enum llvmV = llvmVecType!V;
    enum ir = `
        %p = bitcast ` ~ llvmType!(BaseType!V) ~ `* %0 to ` ~ llvmV ~ `*
        %r = load ` ~ llvmV ~ `, ` ~ llvmV ~ `* %p, align 1
        ret ` ~ llvmV ~ ` %r`;
    private alias impl = __ir_pure!(ir, V, const(BaseType!V)*);
    pragma(inline, true)
    V loadUnaligned(const(BaseType!V)* p) { return impl(p); }
}

template storeUnaligned(V)
if (is(typeof(llvmVecType!V)))
{
    enum llvmV = llvmVecType!V;
    enum ir = `
        %p = bitcast ` ~ llvmType!(BaseType!V) ~ `* %1 to ` ~ llvmV ~ `*
        store ` ~ llvmV ~ ` %0, ` ~ llvmV ~ `* %p, align 1`;
    private alias impl = __ir_pure!(ir, void, V, BaseType!V*);
    pragma(inline, true)
    void storeUnaligned(BaseType!V* p, V value) { impl(value, p); }
}

private enum Cond { eq, ne, gt, ge }

private template cmpMask(Cond cond)
{
    template cmpMask(V)
    if (is(IntOf!(BaseType!V)))
    {
        alias IntOf!(BaseType!V) Relem;
        enum int n = numElements!V;
        alias __vector(Relem[n]) R;
        enum sign = (cond == Cond.eq || cond == Cond.ne) ? "" : (isSigned!(BaseType!V) ? "s" : "u");
        enum condStr = cond == Cond.eq ? "eq" : cond == Cond.ne ? "ne" : cond == Cond.ge ? "ge" : "gt";
        enum op = isFloatingPoint!(BaseType!V) ? "fcmp o" ~ condStr : "icmp " ~ sign ~ condStr;
        enum ir = `
            %cmp = ` ~ op ~ ` ` ~ llvmVecType!V ~ ` %0, %1
            %r = sext <` ~ n.stringof ~ ` x i1> %cmp to ` ~ llvmVecType!R ~ `
            ret ` ~ llvmVecType!R ~ ` %r`;
        alias __ir_pure!(ir, R, V, V) cmpMask;
    }
}

alias cmpMask!(Cond.eq) equalMask;
alias cmpMask!(Cond.ne) notEqualMask;
alias cmpMask!(Cond.gt) greaterMask;
alias cmpMask!(Cond.ge) greaterOrEqualMask;
D";
}
