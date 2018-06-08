// RUN: %ldc -run %s

ulong[2] foo(ulong a, ulong b) @nogc nothrow pure @safe
{
    import ldc.simd;
    return inlineIR!(`
        %agg1 = insertvalue [2 x i64] undef, i64 %0, 0
        %agg2 = insertvalue [2 x i64] %agg1, i64 %1, 1
        ret [2 x i64] %agg2`, ulong[2])(a, b);
}

void main()
{
    assert(foo(123, 456) == [ 123, 456 ]);
}
