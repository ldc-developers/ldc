module ldc.simd;

import core.simd;
import ldc.llvmasm;

pure:
nothrow:
@nogc:
@trusted:

private template isFloatingPoint(T)
{
    enum isFloatingPoint =
        is(T == float) ||
        is(T == double) ||
        is(T == real);
}

private template isIntegral(T)
{
    enum isIntegral =
        is(T == byte) ||
        is(T == ubyte) ||
        is(T == short) ||
        is(T == ushort) ||
        is(T == int) ||
        is(T == uint) ||
        is(T == long) ||
        is(T == ulong);
}

private template isSigned(T)
{
    enum isSigned =
        is(T == byte) ||
        is(T == short) ||
        is(T == int) ||
        is(T == long);
}

private template IntOf(T)
if(isIntegral!T || isFloatingPoint!T)
{
    enum n = T.sizeof;
    static if(n == 1)
        alias byte IntOf;
    else static if(n == 2)
        alias short IntOf;
    else static if(n == 4)
        alias int IntOf;
    else static if(n == 8)
        alias long IntOf;
    else
        static assert(0, "Type not supported");
}

private template BaseType(V)
{
    alias typeof(V.array[0]) BaseType;
}

private template numElements(V)
{
    enum numElements = V.sizeof / BaseType!(V).sizeof;
}

private template llvmType(T)
{
    static if(is(T == float))
        enum llvmType = "float";
    else static if(is(T == double))
        enum llvmType = "double";
    else static if(is(T == byte) || is(T == ubyte) || is(T == void))
        enum llvmType = "i8";
    else static if(is(T == short) || is(T == ushort))
        enum llvmType = "i16";
    else static if(is(T == int) || is(T == uint))
        enum llvmType = "i32";
    else static if(is(T == long) || is(T == ulong))
        enum llvmType = "i64";
    else
        static assert(0,
            "Can't determine llvm type for D type " ~ T.stringof);
}

private template llvmVecType(V)
{
    static if(is(V == void16))
        enum llvmVecType =  "<16 x i8>";
    else static if(is(V == void32))
        enum llvmVecType =  "<32 x i8>";
    else
    {
        alias BaseType!V T;
        enum int n = numElements!V;
        enum llvmT = llvmType!T;
        enum llvmVecType = "<"~n.stringof~" x "~llvmT~">";
    }
}

deprecated("Use `ldc.llvmasm.__ir` or `ldc.llvmasm.__ir_pure` instead of `ldc.simd.inlineIR`.")
pragma(LDC_inline_ir)
    R inlineIR(string s, R, P...)(P);

/**
This template provides access to
$(LINK2 http://llvm.org/docs/LangRef.html#i_shufflevector,
LLVM's shufflevector instruction).

Example:
---
int4 a = [0, 10, 20, 30];
int4 b = [40, 50, 60, 70];
int4 c = shufflevector!(int4, 0, 2, 4, 6)(a, b);
assert(c.array == [0, 20, 40, 60]);
---
*/

template shufflevector(V, mask...)
if(is(typeof(llvmVecType!V)) && mask.length == numElements!V)
{
    enum int n = mask.length;
    enum llvmV = llvmVecType!V;

    template genMaskIr(string ir, m...)
    {
        static if(m.length == 0)
            enum genMaskIr = ir;
        else
        {
            enum int mfront = m[0];

            enum genMaskIr =
                genMaskIr!(ir ~ ", i32 " ~ mfront.stringof, m[1 .. $]);
        }
    }
    enum maskIr = genMaskIr!("", mask)[2 .. $];
    enum ir = `
        %r = shufflevector `~llvmV~` %0, `~llvmV~` %1, <`~n.stringof~` x i32> <`~maskIr~`>
        ret `~llvmV~` %r`;

    alias __ir_pure!(ir, V, V, V) shufflevector;
}

/**
This template provides access to
$(LINK2 http://llvm.org/docs/LangRef.html#i_extractelement,
LLVM's extractelement instruction).

Example:
---
int4 a = [0, 10, 20, 30];
int k = extractelement!(int4, 2)(a);
assert(k == 20);
---
*/

template extractelement(V, int i)
if(is(typeof(llvmVecType!V)) && i < numElements!V)
{
    enum llvmT = llvmType!(BaseType!V);
    enum llvmV = llvmVecType!V;
    enum ir = `
        %r = extractelement `~llvmV~` %0, i32 `~i.stringof~`
        ret `~llvmT~` %r`;

    alias __ir_pure!(ir, BaseType!V, V) extractelement;
}

/**
This template provides access to
$(LINK2 http://llvm.org/docs/LangRef.html#i_insertelement,
LLVM's insertelement instruction).

Example:
---
int4 a = [0, 10, 20, 30];
int b = insertelement!(int4, 2)(a, 50);
assert(b.array == [0, 10, 50, 30]);
---
*/

template insertelement(V, int i)
if(is(typeof(llvmVecType!V)) && i < numElements!V)
{
    enum llvmT = llvmType!(BaseType!V);
    enum llvmV = llvmVecType!V;
    enum ir = `
        %r = insertelement `~llvmV~` %0, `~llvmT~` %1, i32 `~i.stringof~`
        ret `~llvmV~` %r`;

    alias __ir_pure!(ir, V, V, BaseType!V) insertelement;
}

/**
loadUnaligned: Loads a vector from an unaligned pointer.
Example:
---
int[4] a = [0, 10, 20, 30];
int4 v = loadUnaligned!int4(a.ptr);
assert(v.array == a);
---
*/
template loadUnaligned(V)
if(is(typeof(llvmVecType!V)))
{
    enum llvmElementType = llvmType!(BaseType!V);
    enum llvmV = llvmVecType!V;
    enum ir = `
        %p = bitcast `~llvmElementType~`* %0 to `~llvmV~`*
        %r = load `~llvmV~`, `~llvmV~`* %p, align 1
        ret `~llvmV~` %r`;
    private alias impl = __ir_pure!(ir, V, const(BaseType!V)*);

    pragma(inline, true):

    V loadUnaligned(const(BaseType!V)* p)
    {
        return impl(p);
    }

    /// Deprecated: This is the DMD interface, use it only for DMD compatibility. Otherwise, please use the LDC interface.
    V loadUnaligned(const V* p)
    {
        return impl(cast(const(BaseType!V)*) p);
    }
}

/**
storeUnaligned: Stores a vector to an unaligned pointer.
Example:
---
int[4] a;
int4 v = [0, 10, 20, 30];
storeUnaligned!int4(v, a.ptr);
assert(v.array == a);
---
*/
template storeUnaligned(V)
if(is(typeof(llvmVecType!V)))
{
    enum llvmElementType = llvmType!(BaseType!V);
    enum llvmV = llvmVecType!V;
    enum ir = `
        %p = bitcast `~llvmElementType~`* %1 to `~llvmV~`*
        store `~llvmV~` %0, `~llvmV~`* %p, align 1`;
    private alias impl = __ir_pure!(ir, void, V, BaseType!V*);

    pragma(inline, true):

    void storeUnaligned(V value, BaseType!V* p)
    {
        impl(value, p);
    }

    /// Deprecated: This is the DMD interface, use it only for DMD compatibility. Otherwise, please use the LDC interface.
    V storeUnaligned(V* p, V value)
    {
        impl(value, cast(BaseType!V*) p);
        return value;
    }
}

private enum Cond{ eq, ne, gt, ge }

private template cmpMask(Cond cond)
{
    template cmpMask(V)
    if(is(IntOf!(BaseType!V)))
    {
        alias BaseType!V T;
        enum llvmT = llvmType!T;

        alias IntOf!T Relem;

        enum int n = numElements!V;
        alias __vector(Relem[n]) R;

        enum llvmV = llvmVecType!V;
        enum llvmR = llvmVecType!R;
        enum sign =
            (cond == Cond.eq || cond == Cond.ne) ? "" :
            isSigned!T ? "s" : "u";
        enum condStr =
            cond == Cond.eq ? "eq" :
            cond == Cond.ne ? "ne" :
            cond == Cond.ge ? "ge" : "gt";
        enum op =
            isFloatingPoint!T ? "fcmp o"~condStr : "icmp "~sign~condStr;

        enum ir = `
            %cmp = `~op~` `~llvmV~` %0, %1
            %r = sext <`~n.stringof~` x i1> %cmp to `~llvmR~`
            ret `~llvmR~` %r`;

        alias __ir_pure!(ir, R, V, V) cmpMask;
    }
}

/**
equalMask, notEqualMask, greaterMask and greaterOrEqualMask perform an
element-wise comparison between two vectors and return a vector with
signed integral elements. The number of elements in the returned vector
and their size is the same as in parameter vectors. If the condition in
the name of the function holds for elements of the parameter vectors at
a given index, all bits of the element of the result at that index are
set to 1, otherwise the element of the result is zero.

Example:
---
float4 a = [1, 3, 5, 7];
float4 b = [2, 3, 4, 5];
int4 c = greaterMask!float4(a, b);
writeln(c.array);
assert(c.array == [0, 0, 0xffff_ffff, 0xffff_ffff]);
---
*/
alias cmpMask!(Cond.eq) equalMask;
alias cmpMask!(Cond.ne) notEqualMask; /// Ditto
alias cmpMask!(Cond.gt) greaterMask; /// Ditto
alias cmpMask!(Cond.ge) greaterOrEqualMask; /// Ditto

