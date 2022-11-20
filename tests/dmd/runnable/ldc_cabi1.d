// EXTRA_CPP_SOURCES: ldc_cabi2.cpp
// disable deprecations (for imaginary and complex types)
// REQUIRED_ARGS: -d

import core.stdc.stdarg;
import core.stdc.stdio;

extern (C)
{
  __gshared byte a = 1, b = -42, c = 3, d = -10, e = 0, f = -50, g = 20, h = -77;
  __gshared uint errors;
}

void test(bool b, string file = __FILE__, size_t line = __LINE__)
{
    if (!b)
    {
        printf("%.*s:%zu: failed check\n", cast(int) file.length, file.ptr, line);
        ++errors;
    }
}

// private doesn't work like static here - bug?
private bool testar(byte[] a, byte a0)
{
    for (size_t i = 0; i < a.length; ++i) {
        if (a[i] != cast(byte)(a0+i)) {
          return false;
        }
    }
    return true;
}

int main()
{
    enum a = D4.alignof;
    static assert(is(typeof(a) == size_t));
    static assert(D4.alignof == double.alignof);
    static assert(F4.alignof == float.alignof);
    static assert(S9.alignof == double.alignof);
    dcall();
    ccall();
    return (errors != 0) ? 1 : 0;
}

extern(C):
struct EMPTY {};
struct B1 {byte a;}
struct B2 {byte a, b;}
struct I1 {int a;}
struct I2 {int a, b;}
union UI1 {int a; short b; byte c;}
union UI1a {short a; byte b; int c;}
struct NI1 {I1 a;}
struct NUI1 {UI1 a;}
union UNI1 {UI1 a; NI1 b; int c;}
struct S3 {char a; short b;}
struct S6 {char a; int b; char c;}
struct S9 {char a; double b;}
struct S19 {char a; double b, c;}
struct F2 {float a, b;}
struct F2i {ifloat a, b;}
struct F2ir {ifloat a; float b;}
struct F2ri {float a; ifloat b;}
struct F4 {float a, b, c, d;}
//enum DD : double {x=0.0, b=1.0};
//struct Dx {DD a;}
struct D1 {double a;}
struct D2 {double a, b;}
struct D4 {double a, b, c, d;}
struct D5 {double a, b, c, d, e;}
struct D8 {double a, b, c, d, e, f, g, h;}
union UD4 {D1 a; D2 b; D4 c; double e;}
struct CX1 {creal a;}
struct CX2 {creal a, b;}
struct CX3 {creal a, b, c;}
struct CX1D2 {creal a; D2 b;}
struct DA0 {double[0] a;}
struct DA2 {double[2] a;}
struct DA3 {double[3] a;}
struct DA4 {double[4] a;}
struct DA5 {double[5] a;}
struct DA8 {double[8] a;}
struct CA4 {char[4] a;}
struct DHFA1 {EMPTY a; double d;}
//struct DHFA1 {EMPTY a; EMPTY b; double[0] c; double d;}
struct DHFA2 {double a; D1 b;}
struct DHFA2a {D1 a; double b;}
struct DHFA4 {D1 a; double b; D2 c;}
struct DHFA4a {D1 a; real b; D2 c;}
struct DHFA4b {DHFA2 a; DHFA2a b;}
struct DHFA4c {DHFA2 a; double[2] b;}
struct DHFA4d {DHFA2 a; DA2 b;}
struct DHFA4e {DA2 a; double b, c;}
struct DHFA4x {D2[2] a;};
struct DHFAx {D1 a; F2 b; double d;}
struct DHFA5 {D1 a; double b; D2 c; double d;}
struct S40 {D2 a; double b; D2 c;}
struct S1 {
    byte a;
    ~this() {}
}
struct S2 {
    byte a;
    this(this) {}
}
struct SA64 {byte[64] a;}
struct SA65 {byte[65] a;}


void cvfun(int s, ...);

B1 cretb1(B1 x);
B1 dretb1(B1 x)
{
    test(x.a == a);
    B1 r = {++a};
    return r;
}

B2 cretb2(B2 x);
B2 dretb2(B2 x)
{
    test(x.a == a);
    test(x.b == b);
    B2 r = {++a, ++b};
    return r;
}

I1 creti1(I1 x);
I1 dreti1(I1 x)
{
    test(x.a == a);
    I1 r = {++a};
    return r;
}


I2 creti2(I2 x);
I2 dreti2(I2 x)
{
    test(x.a == a);
    test(x.b == b);
    I2 r = {++a, ++b};
    return r;
}

UNI1 cretuni1(UNI1 x);
UNI1 dretuni1(UNI1 x)
{
    test(x.a.a == a);
    UNI1 r = {{++a}};
    return r;
}

F2 cretf2(F2 x);
F2 dretf2(F2 x)
{
    test(x.a == a);
    test(x.b == b);
    F2 r = {++a, ++b};
    return r;
}

F2ir cretf2ir(F2ir x);
F2ir dretf2ir(F2ir x)
{
    test(x.a == a*1i);
    test(x.b == b);
    F2ir r = {++a*1i, ++b};
    return r;
}

F2ri cretf2ri(F2ri x);
F2ri dretf2ri(F2ri x)
{
    test(x.a == a);
    test(x.b == b*1i);
    F2ri r = {++a, ++b*1i};
    return r;
}

F4 cretf4(F4 x);
F4 dretf4(F4 x)
{
    test(x.a == a);
    test(x.b == b);
    test(x.c == c);
    test(x.d == d);
    F4 r = {++a, ++b, ++c, ++d};
    return r;
}

// Dx cretdx(Dx x);
// Dx dretdx(Dx x)
// {
//     test(x.a == a);
//     Dx r;
//     return r;
// }

D4 cretd4(D4 x);
D4 dretd4(D4 x)
{
    test(x.a == a);
    test(x.b == b);
    test(x.c == c);
    test(x.d == d);
    D4 r = {++a, ++b, ++c, ++d};
    return r;
}

D5 cretd5(D5 x);
D5 dretd5(D5 x)
{
    test(x.a == a);
    test(x.b == b);
    test(x.c == c);
    test(x.d == d);
    test(x.e == e);
    D5 r = {++a, ++b, ++c, ++d, ++e};
    return r;
}

D8 cretd8(D8 x);
D8 dretd8(D8 x)
{
    test(x.a == a);
    test(x.b == b);
    test(x.c == c);
    test(x.d == d);
    test(x.e == e);
    test(x.f == f);
    test(x.g == g);
    test(x.h == h);
    D8 r = {++a, ++b, ++c, ++d, ++e, ++f, ++g, ++h};
    return r;
}

UD4 cretud4(UD4 x);
UD4 dretud4(UD4 x)
{
    test(x.c.a == a);
    test(x.c.b == b);
    test(x.c.c == c);
    test(x.c.d == d);
    UD4 r;
    r.c = D4(++a, ++b, ++c, ++d);
    return r;
}

DA0 cretda0(DA0 x);
DA0 dretda0(DA0 x)
{
    DA0 r;
    return r;
}

DA4 cretda4(DA4 x);
DA4 dretda4(DA4 x)
{
    test(x.a[0] == a);
    test(x.a[1] == b);
    test(x.a[2] == c);
    test(x.a[3] == d);
    DA4 r = {[++a, ++b, ++c, ++d]};
    return r;
}

DA5 cretda5(DA5 x);
DA5 dretda5(DA5 x)
{
    test(x.a[0] == a);
    test(x.a[1] == b);
    test(x.a[2] == c);
    test(x.a[3] == d);
    test(x.a[4] == e);
    DA5 r = {[++a, ++b, ++c, ++d, ++e]};
    return r;
}

DA8 cretda8(DA8 x);
DA8 dretda8(DA8 x)
{
    test(x.a[0] == a);
    test(x.a[1] == b);
    test(x.a[2] == c);
    test(x.a[3] == d);
    test(x.a[4] == e);
    test(x.a[5] == f);
    test(x.a[6] == g);
    test(x.a[7] == h);
    DA8 r = {[++a, ++b, ++c, ++d, ++e, ++f, ++g, ++h]};
    return r;
}

CX1 cretcx1(CX1 x);
CX1 dretcx1(CX1 x)
{
    test(x.a == a + b*1i);
    CX1 r = {++a + ++b*1i};
    return r;
}

CX2 cretcx2(CX2 x);
CX2 dretcx2(CX2 x)
{
    test(x.a == a + b*1i);
    test(x.b == c + d*1i);
    CX2 r = {++a + ++b*1i, ++c + ++d*1i};
    return r;
}

CX3 cretcx3(CX3 x);
CX3 dretcx3(CX3 x)
{
    test(x.a == a + b*1i);
    test(x.b == c + d*1i);
    test(x.c == e + f*1i);
    CX3 r = {++a + ++b*1i, ++c + ++d*1i, ++e + ++f*1i};
    return r;
}

CX1D2 cretcx1d2(CX1D2 x);
CX1D2 dretcx1d2(CX1D2 x)
{
    test(x.a == a + b*1i);
    test(x.b.a == c);
    test(x.b.b == d);
    CX1D2 r = {++a + ++b*1i, {++c, ++d}};
    return r;
}

DHFA1 cretdhfa1(DHFA1 x);
DHFA1 dretdhfa1(DHFA1 x)
{
    test(x.d == a);
    DHFA1 r;
    r.d = ++a;
    return r;
}

DHFA2 cretdhfa2(DHFA2 x);
DHFA2 dretdhfa2(DHFA2 x)
{
    test(x.a == a);
    test(x.b.a == b);
    DHFA2 r = {++a, {++b}};
    return r;
}

DHFA2a cretdhfa2a(DHFA2a x);
DHFA2a dretdhfa2a(DHFA2a x)
{
    test(x.a.a == a);
    test(x.b == b);
    DHFA2a r = {{++a}, ++b};
    return r;
}

DHFA4 cretdhfa4(DHFA4 x);
DHFA4 dretdhfa4(DHFA4 x)
{
    test(x.a.a == a);
    test(x.b == b);
    test(x.c.a == c);
    test(x.c.b == d);
    DHFA4 r = {{++a}, ++b, {++c, ++d}};
    return r;
}

DHFA4a cretdhfa4a(DHFA4a x);
DHFA4a dretdhfa4a(DHFA4a x)
{
    test(x.a.a == a);
    test(x.b == b);
    test(x.c.a == c);
    test(x.c.b == d);
    DHFA4a r = {{++a}, ++b, {++c, ++d}};
    return r;
}

DHFA4b cretdhfa4b(DHFA4b x);
DHFA4b dretdhfa4b(DHFA4b x)
{
    test(x.a.a == a);
    test(x.a.b.a == b);
    test(x.b.a.a == c);
    test(x.b.b == d);
    DHFA4b r = {{++a, {++b}}, {{++c}, ++d}};
    return r;
}

DHFA4c cretdhfa4c(DHFA4c x);
DHFA4c dretdhfa4c(DHFA4c x)
{
    test(x.a.a == a);
    test(x.a.b.a == b);
    test(x.b[0] == c);
    test(x.b[1] == d);
    DHFA4c r = {{++a, {++b}}, [++c, ++d]};
    return r;
}

DHFA4d cretdhfa4d(DHFA4d x);
DHFA4d dretdhfa4d(DHFA4d x)
{
    test(x.a.a == a);
    test(x.a.b.a == b);
    test(x.b.a[0] == c);
    test(x.b.a[1] == d);
    DHFA4d r = {{++a, {++b}}, {[++c, ++d]}};
    return r;
}

DHFA4e cretdhfa4e(DHFA4e x);
DHFA4e dretdhfa4e(DHFA4e x)
{
    test(x.a.a[0] == a);
    test(x.a.a[1] == b);
    test(x.b == c);
    test(x.c == d);
    DHFA4e r = {{[++a, ++b]}, ++c, ++d};
    return r;
}

DHFA4x cretdhfa4x(DHFA4x x);
DHFA4x dretdhfa4x(DHFA4x x)
{
    test(x.a[0].a == a);
    test(x.a[0].b == b);
    test(x.a[1].a == c);
    test(x.a[1].b == d);
    DHFA4x r = {[{++a, ++b}, {++c, ++d}]};
    return r;
}

DHFA5 cretdhfa5(DHFA5 x);
DHFA5 dretdhfa5(DHFA5 x)
{
    test(x.a.a == a);
    test(x.b == b);
    test(x.c.a == c);
    test(x.c.b == d);
    test(x.d == 42);
    DHFA5 r = {{++a}, ++b, {++c, ++d}, 42.0};
    return r;
}

S1 crets1(S1 x);
S1 drets1(S1 x)
{
    test(x.a == a);
    S1 r = S1(++a);
    return r;
}

S2 crets2(S2 x);
S2 drets2(S2 x)
{
    test(x.a == a);
    S2 r = S2(++a);
    return r;
}

SA64 cretsa64(SA64 x);
SA64 dretsa64(SA64 x)
{
    test(testar(x.a, a));

    SA64 r;
    ++a;
    for (size_t i = 0; i < 64; ++i) {
        r.a[i] = cast(byte)(a+i);
    }
    return r;
}

SA65 cretsa65(SA65 x);
SA65 dretsa65(SA65 x)
{
    test(testar(x.a, a));

    SA65 r;
    ++a;
    for (size_t i = 0; i < 65; ++i) {
        r.a[i] = cast(byte)(a+i);
    }
    return r;
}

void dvfun(int s, ...)
{
    va_list args;
    va_start(args, s);
    final switch (s) {
    case 0:
        dretb1(va_arg!B1(args));
        break;
    case 1:
        dretb2(va_arg!B2(args));
        break;
    case 2:
        dreti2(va_arg!I2(args));
        break;
    case 3:
        dretf4(va_arg!F4(args));
        break;
    case 4:
        dretd4(va_arg!D4(args));
        break;
    case 5:
        dretdhfa2(va_arg!DHFA2(args));
        break;
    case 6:
        dretdhfa2a(va_arg!DHFA2a(args));
        break;
    case 7:
        dretuni1(va_arg!UNI1(args));
        break;
    }
}

version(none){
struct CR1 {cdouble a;}
cdouble cretcd(cdouble x);
CR1 cretcr1(CR1 x);
}

version (none) {
struct Foo {
    double a;
    this(double x) {a = x;}
}

class Bar {
    double a;
    this(double x) {a = x;}
}
}

extern (D) void xvfun(...);

void ccall();
void dcall()
{
    //xvfun(2.0f);
    
    version (none) {
    cdouble cd = cretcd(4.5+2i);
    test(cd == 0);

    CR1 cr1 = cretcr1(CR1(4.5+2i));
    test(cr1.a == 0);
    }

    version (none) {
    Foo f = Foo(1.0);
    Bar z = new Bar(1.0);
    cvfun(1, f, z);
    }
    
    B1 b1 = cretb1(B1(++a));
    test(b1.a == a);

    B2 b2 = cretb2(B2(++a, ++b));
    test(b2.a == a);
    test(b2.b == b);

    I2 i2 = creti2(I2(++a, ++b));
    test(i2.a == a);
    test(i2.b == b);

    UNI1 uni1i = {{++a}};
    UNI1 uni1 = cretuni1(uni1i);
    test(uni1.a.a == a);

    F4 f4 = cretf4(F4(++a, ++b, ++c, ++d));
    test(f4.a == a);
    test(f4.b == b);
    test(f4.c == c);
    test(f4.d == d);    

    D4 d4 = cretd4(D4(++a, ++b, ++c, ++d));
    test(d4.a == a);
    test(d4.b == b);
    test(d4.c == c);
    test(d4.d == d);    

    D5 d5 = cretd5(D5(++a, ++b, ++c, ++d, ++e));
    test(d5.a == a);
    test(d5.b == b);
    test(d5.c == c);
    test(d5.d == d);
    test(d5.e == e);

    D8 d8 = cretd8(D8(++a, ++b, ++c, ++d, ++e, ++f, ++g, ++h));
    test(d8.a == a);
    test(d8.b == b);
    test(d8.c == c);
    test(d8.d == d);
    test(d8.e == e);
    test(d8.f == f);
    test(d8.g == g);
    test(d8.h == h);

    UD4 ud4;
    ud4.c = D4(++a, ++b, ++c, ++d);
    UD4 ud4r = cretud4(ud4);
    test(ud4r.c.a == a);
    test(ud4r.c.b == b);
    test(ud4r.c.c == c);
    test(ud4r.c.d == d);    

    DA4 da4 = cretda4(DA4([++a, ++b, ++c, ++d]));
    test(da4.a[0] == a);
    test(da4.a[1] == b);
    test(da4.a[2] == c);
    test(da4.a[3] == d);

    DA5 da5 = cretda5(DA5([++a, ++b, ++c, ++d, ++e]));
    test(da5.a[0] == a);
    test(da5.a[1] == b);
    test(da5.a[2] == c);
    test(da5.a[3] == d);
    test(da5.a[4] == e);

    DA8 da8 = cretda8(DA8([++a, ++b, ++c, ++d, ++e, ++f, ++g, ++h]));
    test(da8.a[0] == a);
    test(da8.a[1] == b);
    test(da8.a[2] == c);
    test(da8.a[3] == d);
    test(da8.a[4] == e);
    test(da8.a[5] == f);
    test(da8.a[6] == g);
    test(da8.a[7] == h);

    DHFA2 dhfa2 = cretdhfa2(DHFA2(++a, D1(++b)));
    test(dhfa2.a == a);
    test(dhfa2.b.a == b);    

    DHFA2a dhfa2a = cretdhfa2a(DHFA2a(D1(++a), ++b));
    test(dhfa2a.a.a == a);
    test(dhfa2a.b == b);    

    DHFA4x dhfa4xi = {[{++a, ++b}, {++c, ++d}]};
    DHFA4x dhfa4x = cretdhfa4x(dhfa4xi);
    test(dhfa4x.a[0].a == a);
    test(dhfa4x.a[0].b == b);
    test(dhfa4x.a[1].a == c);
    test(dhfa4x.a[1].b == d);

    // structs with postblit or dtor may not be passed like a similar POD
    // struct.  Depends on target, although would be undefined behavior since
    // C code can't obey struct life cycle.
    // going to be obey
    version (none) {
    S1 s1 = crets1(S1(++a));
    test(s1.a == a);

    S2 s2 = crets2(S2(++a));
    test(s2.a == a);
    }

    SA64 s64;
    ++a;
    for (size_t i = 0; i < 64; ++i) {
        s64.a[i] = cast(byte)(a+i);
    }
    SA64 s64r = cretsa64(s64);
    test(testar(s64r.a, a));

    SA65 s65;
    ++a;
    for (size_t i = 0; i < 65; ++i) {
        s65.a[i] = cast(byte)(a+i);
    }
    SA65 s65r = cretsa65(s65);
    test(testar(s65r.a, a));
        
    b1.a = ++a;
    cvfun(0, b1);

    b2.a = ++a;
    b2.b = ++b;
    cvfun(1, b2);

    i2.a = ++a;
    i2.b = ++b;
    cvfun(2, i2);

    uni1.a.a = ++a;
    cvfun(7, uni1);

    f4.a = ++a;
    f4.b = ++b;
    f4.c = ++c;
    f4.d = ++d;
    cvfun(3, f4);

    d4.a = ++a;
    d4.b = ++b;
    d4.c = ++c;
    d4.d = ++d;
    cvfun(4, d4);

    dhfa2.a = ++a;
    dhfa2.b.a = ++b;
    cvfun(5, dhfa2);    

    dhfa2a.a.a = ++a;
    dhfa2a.b = ++b;
    cvfun(6, dhfa2a);    
}
