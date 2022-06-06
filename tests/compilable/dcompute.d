// Tests that
//  - we don't try to link with one file on the commandline that is @compute
//  - turning on debugging doesn't ICE
//  - don't analyse uninstantiated templates
//  - typeid generated for hashing of struct (typeid(const(T))) is ignored and does not error
//  - if (__ctfe) and if (!__ctfe) don't cause errors

// REQUIRES: target_NVPTX
// RUN: %ldc -mdcompute-targets=cuda-350 -g %s

@compute(CompileFor.deviceOnly) module dcompute;
import ldc.dcompute;

@kernel void foo()
{
    if (__ctfe)
    {
        auto a = new int;
    }

    if (__ctfe)
    {
        auto a = new int;
    }
    else {}

    if (!__ctfe)
    {
    }
    else
    {
        auto a = new int;
    }
}

struct AutoIndexed(T)
{
    T p = void;
    alias U = typeof(*T);

    @property U index()
    {
        return p[0];
    }

    @property void index(U t)
    {
        p[0] = t;
    }
    @disable this();
    void opAssign(U u) { index = u; }
    alias index this;
}
alias aagf = AutoIndexed!(GlobalPointer!(float));

@kernel void auto_index_test(aagf a,
                             aagf b,
                             aagf c)
{
    a = b + c;
}

