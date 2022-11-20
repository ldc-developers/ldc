struct S { uint x; }

template MakeS(uint x) { const MakeS = S(x); }

struct S2 { alias .MakeS MakeS; }

void f() {
    S2 s2;
    auto n = s2.MakeS!(0); //////////// XXX
}