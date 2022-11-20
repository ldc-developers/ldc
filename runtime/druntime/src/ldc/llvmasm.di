module ldc.llvmasm;

struct __asmtuple_t(T...)
{
    T v;
}

pragma(LDC_inline_asm)
{
    void __asm()(const(char)[] asmcode, const(char)[] constraints, ...) pure nothrow @nogc;
    T __asm(T)(const(char)[] asmcode, const(char)[] constraints, ...) pure nothrow @nogc;

    void __asm_trusted()(const(char)[] asmcode, const(char)[] constraints, ...) @trusted pure nothrow @nogc;
    T __asm_trusted(T)(const(char)[] asmcode, const(char)[] constraints, ...) @trusted pure nothrow @nogc;

    template __asmtuple(T...)
    {
        __asmtuple_t!(T) __asmtuple(const(char)[] asmcode, const(char)[] constraints, ...);
    }
}


/++
 + Calls a function whose body is written in the LLVM assembly language (IR).
 +
 + Template params:
 +   s = the LLVM IR function body code
 +   R = the function return type
 +   P... = the types of the function arguments
 +
 + If the return type is `void` then `ret void` is automatically added at the
 + end of the function body by LDC.
 +
 + Example:
 + ---
 + import ldc.llvmasm;
 + int add(int a, int b)
 + {
 +     return __ir!(`%r = add i32 %0, %1
 +                   ret i32 %r`, int)(a, b);
 + }
 + ---
 +/
pragma(LDC_inline_ir)
    R __ir(string s, R, P...)(P params) @trusted nothrow @nogc;

/// Ditto
pragma(LDC_inline_ir)
    R __ir_pure(string s, R, P...)(P params) @trusted nothrow @nogc pure;


/++
 + Extended inline IR, see __ir.
 + Calls a function whose body is written in the LLVM assembly language (IR).
 +
 + Template params:
 +   prefix = LLVM IR to be put _before_ the IR function definition
 +   code = the LLVM IR function body code
 +   suffix = LLVM IR to be put _after_ the IR function definition
 +   R = the function return type
 +   P... = the types of the function arguments
 +
 + Example:
 + ---
 + import ldc.llvmasm;
 + alias __irEx!("", "store i32 %1, i32* %0, !nontemporal !0", "!0 = !{i32 1}", void, int*, int) nontemporalStore;
 + ---
 +/
pragma(LDC_inline_ir)
    R __irEx(string prefix, string code, string suffix, R, P...)(P) @trusted nothrow @nogc;

/// Ditto
pragma(LDC_inline_ir)
    R __irEx_pure(string prefix, string code, string suffix, R, P...)(P) @trusted nothrow @nogc pure;

