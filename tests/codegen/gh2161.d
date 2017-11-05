// Verify some minimum inlining.

// UNSUPPORTED: llvm308

// RUN: %ldc %s -output-ll -of=%t_safeonly.ll -O3 -release
// RUN: %ldc %s -output-ll -of=%t_off.ll      -O3 -release -boundscheck=off
// RUN: FileCheck %s < %t_safeonly.ll
// RUN: FileCheck %s < %t_off.ll

import std.algorithm;
import std.range;

// CHECK-LABEL: define {{.*}}@{{.*}}_D6gh2161__T13insertionSort
void insertionSort(Range, Less)(Range r, Less l)
if (hasLength!Range && isRandomAccessRange!Range && hasSlicing!Range)
{
    // no calls/invokes to any function in this module or in Phobos
    // CHECK-NOT: {{(call|invoke) .*@.*(_D6gh2161|_D3std)}}

    foreach (immutable i; 1 .. r.length)
    {
        bringToFront(
            r[0 .. i].assumeSorted!((a,b) => l(a, b)).upperBound(r[i]),
            r[i .. i + 1]);
    }

    // CHECK: {{^\}$}}
}

struct Pair(T, U = T)
{
    T f;
    U s;

    this(const T a, const U b)
    {
        f = a;
        s = b;
    }

    bool opEquals(ref const Pair r) const
    {
        return f == r.f && s == r.s;
    }
}

alias IntPair = Pair!int;
alias IntPairPair = Pair!IntPair;

bool lt(int l, int r)
{
    return l < r;
}

bool lt(T, U)(ref const Pair!(T, U) l, ref const Pair!(T, U) r)
{
    if (l.f != r.f) return lt(l.f, r.f);
    return lt(l.s, r.s);
}

struct La(T)
{
    bool opCall(ref const T l, ref const T r) const
    {
        return lt(l, r);
    }
}


struct Ld(T)
{
    bool opCall(ref const T l, ref const T r) const
    {
        return lt(r, l);
    }
}

void main()
{
    enum N = 10;
    enum S = 1000;

    IntPairPair[] v;
    v.reserve(24 * S);
    for (auto i = 0; i < S; i++)
    {
        v ~= IntPairPair(IntPair(i + 0, i + 1), IntPair(i + 2, i + 3));
        v ~= IntPairPair(IntPair(i + 0, i + 1), IntPair(i + 3, i + 2));
        v ~= IntPairPair(IntPair(i + 0, i + 2), IntPair(i + 1, i + 3));
        v ~= IntPairPair(IntPair(i + 0, i + 2), IntPair(i + 3, i + 1));
        v ~= IntPairPair(IntPair(i + 0, i + 3), IntPair(i + 1, i + 2));
        v ~= IntPairPair(IntPair(i + 0, i + 3), IntPair(i + 2, i + 1));
    }

    La!IntPairPair la;
    Ld!IntPairPair ld;
    for (auto i = 0; i < N; i++)
    {
        insertionSort(v, la);
        insertionSort(v, ld);
    }
}
