// Tests that
//  - we dont try to link with one file on the commandline that is @compute
//  - truning on debugging doesn't ICE
//  - dont analyse uninstansiated templates
//  - typeid generated for hashing of struct (typeid(const(T))) is ignored and does not error

// REQUIRES: target_NVPTX
// RUN: %ldc -mdcompute-targets=cuda-350 -g

@compute(Compilefor.deviceOnly) module dcompute;
import ldc.dcompute;

@kernel void foo()
{

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
    alias index this;
}
alias aagf = AutoIndexed!(GlobalPointer!(float));

@kernel void auto_index_test(aagf a,
                             aagf b,
                             aagf c)
{
    a = b + c;
}

