
// Check object files are identical regardless -output-s option
// RUN: %ldc -c -O3 -output-s -output-o -of=%t1.o %s && %ldc -c -O3 -output-o -of=%t2.o %s && %diff_binary %t1.o %t2.o

import core.simd;
import ldc.simd;

alias Vec = Vector!(float[4]);

extern void foo(float*);

// Just some random sufficiently complex code
Vec bar(Vec v)
{
    float[4] val;
    Vec ret;
    storeUnaligned!Vec(v,val.ptr);
    foo(val.ptr);
    ret = loadUnaligned!Vec(val.ptr);
    return ret;
}

Vec baz(Vec v)
{
    return bar(bar(bar(bar(bar(bar(v))))));
}
