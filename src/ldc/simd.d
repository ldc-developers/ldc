module ldc.simd;

import core.simd;

pure:
nothrow:
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

pragma(llvm_inline_ir)
    R inlineIR(string s, R, P...)(P);

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

    alias inlineIR!(ir, V, V, V) shufflevector;
}

template extractelement(V, int i)
if(is(typeof(llvmVecType!V)) && i < numElements!V)
{
    enum llvmT = llvmType!(BaseType!V);
    enum llvmV = llvmVecType!V;
    enum ir = `
        %r = extractelement `~llvmV~` %0, i32 `~i.stringof~`
        ret `~llvmT~` %r`;

    alias inlineIR!(ir, BaseType!V, V) extractelement; 
}

template insertelement(V, int i)
if(is(typeof(llvmVecType!V)) && i < numElements!V)
{
    enum llvmT = llvmType!(BaseType!V);
    enum llvmV = llvmVecType!V;
    enum ir = `
        %r = insertelement `~llvmV~` %0, `~llvmT~` %1, i32 `~i.stringof~`
        ret `~llvmV~` %r`;

    alias inlineIR!(ir, V, V, BaseType!V) insertelement; 
}

template loadUnaligned(V)
if(is(typeof(llvmVecType!V)))
{
    alias BaseType!V T;
    enum llvmT = llvmType!T;
    enum llvmV = llvmVecType!V;
    enum ir = `
        %p = bitcast `~llvmT~`* %0 to `~llvmV~`*
        %r = load `~llvmV~`* %p, align 1
        ret `~llvmV~` %r`;

    alias inlineIR!(ir, V, T*) loadUnaligned; 
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
            cond == Cond.ge ? "gt" : "ge";
        enum op = 
            isFloatingPoint!T ? "fcmp o"~condStr : "icmp "~sign~condStr;

        enum ir = `
            %cmp = `~op~` `~llvmV~` %0, %1
            %r = sext <`~n.stringof~` x i1> %cmp to `~llvmR~`
            ret `~llvmR~` %r`;

        alias inlineIR!(ir, R, V, V) cmpMask;
    }
}

alias cmpMask!(Cond.eq) equalMask;
alias cmpMask!(Cond.ne) notEqualMask;
alias cmpMask!(Cond.gt) greaterMask;
alias cmpMask!(Cond.gt) greaterOrEqualMask;

