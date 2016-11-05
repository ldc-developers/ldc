// Explicitly target Linux x86_64.
// REQUIRES: target_X86
// RUN: %ldc -mtriple=x86_64-pc-linux-gnu -c %s %S/inputs/gh1741b.d

struct S
{
    C c;
}

class C
{
    @property s() { return S(); }
}
