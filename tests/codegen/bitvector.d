// RUN: %ldc -output-ll %s

import core.simd;

struct __VecBitBool
{
    bool b;
    alias b this;
}

alias V = Vector!(__VecBitBool[4]);

V foo1(V v1)
{
    v1[2] = true;
    return v1;
}

V foo2()
{
    V v1;
    v1[2] = true;
    return v1;
}