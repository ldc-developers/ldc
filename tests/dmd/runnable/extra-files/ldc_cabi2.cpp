#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>

// ldc_cabi1 defines these
extern "C"
{
  extern int8_t a, b, c, d, e, f, g, h;
  extern uint32_t errors;
}

#define TEST(b) \
  if (!(b)) (++errors, printf("%s:%u: failed check\n", __FILE__, __LINE__))

static bool testar(int8_t* a, size_t len, int8_t a0)
{
    for (size_t i = 0; i < len; ++i) {
        if (a[i] != (int8_t)(a0+i)) {
            return false;
        }
    }
    return true;
}

extern "C" {

struct EMPTY {};
struct B1 {int8_t a;};
struct B2 {int8_t a, b;};
struct I1 {int a;};
struct I2 {int a, b;};
union UI1 {int a; short b; int8_t c;};
union UI1a {short a; int8_t b; int c;};
struct NI1 {I1 a;};
struct NUI1 {UI1 a;};
union UNI1 {UI1 a; NI1 b; int c;};
struct S3 {char a; short b;};
struct S6 {char a; int b; char c;};
struct S9 {char a; double b;};
struct S19 {char a; double b, c;};
struct F2 {float a, b;};
struct F4 {float a, b, c, d;};
struct D1 {double a;};
struct D2 {double a, b;};
struct D4 {double a, b, c, d;};
struct D5 {double a, b, c, d, e;};
struct D8 {double a, b, c, d, e, f, g, h;};
union UD4 {D1 a; D2 b; D4 c; double e;};
struct DA0 {double a[0];};
struct DA4 {double a[4];};
struct DA5 {double a[5];};
struct DA8 {double a[8];};
struct CA4 {char a[4];};
struct DHFA1 {EMPTY a; double d;};
//struct DHFA1 {EMPTY a; EMPTY b; double c[0]; double d;};
struct DHFA2 {double a; D1 b;};
struct DHFA2a {D1 a; double b;};
struct DHFA4x {D2 a[2];};
struct S1 {int8_t a;};
struct S2 {int8_t a;};
struct SA64 {int8_t a[64];};
struct SA65 {int8_t a[65];};

void dvfun(int s, ...);

B1 dretb1(B1 x);
B1 cretb1(B1 x)
{
    TEST(x.a == a);
    B1 r = {++a};
    return r;
}

B2 dretb2(B2 x);
B2 cretb2(B2 x)
{
    TEST(x.a == a);
    TEST(x.b == b);
    B2 r = {++a, ++b};
    return r;
}

I1 dreti1(I1 x);
I1 creti1(I1 x)
{
    TEST(x.a == a);
    I1 r = {++a};
    return r;
}


I2 dreti2(I2 x);
I2 creti2(I2 x)
{
    TEST(x.a == a);
    TEST(x.b == b);
    I2 r = {++a, ++b};
    return r;
}

UI1a dretui1a(UI1a x);
UI1a cretui1a(UI1a x)
{
    TEST(x.c == a);
    UI1a r;
    r.c = ++a;
    return r;
}

UNI1 dretuni1(UNI1 x);
UNI1 cretuni1(UNI1 x)
{
    TEST(x.a.a == a);
    UNI1 r = {{++a}};
    return r;
}

F4 dretf4(F4 x);
F4 cretf4(F4 x)
{
    TEST(x.a == a);
    TEST(x.b == b);
    TEST(x.c == c);
    TEST(x.d == d);
    F4 r = {(float)++a, (float)++b, (float)++c, (float)++d};
    return r;
}

D4 dretd4(D4 x);
D4 cretd4(D4 x)
{
    TEST(x.a == a);
    TEST(x.b == b);
    TEST(x.c == c);
    TEST(x.d == d);
    D4 r = {(double)++a, (double)++b, (double)++c, (double)++d};
    return r;
}

D5 dretd5(D5 x);
D5 cretd5(D5 x)
{
    TEST(x.a == a);
    TEST(x.b == b);
    TEST(x.c == c);
    TEST(x.d == d);
    TEST(x.e == e);
    D5 r = {(double)++a, (double)++b, (double)++c, (double)++d, (double)++e};
    return r;
}

D8 dretd8(D8 x);
D8 cretd8(D8 x)
{
    TEST(x.a == a);
    TEST(x.b == b);
    TEST(x.c == c);
    TEST(x.d == d);
    TEST(x.e == e);
    TEST(x.f == f);
    TEST(x.g == g);
    TEST(x.h == h);
    D8 r = {(double)++a, (double)++b, (double)++c, (double)++d,
            (double)++e, (double)++f, (double)++g, (double)++h};
    return r;
}

UD4 dretud4(UD4 x);
UD4 cretud4(UD4 x)
{
    TEST(x.c.a == a);
    TEST(x.c.b == b);
    TEST(x.c.c == c);
    TEST(x.c.d == d);
    UD4 r;
    D4 d4 = {(double)++a, (double)++b, (double)++c, (double)++d};
    r.c = d4;
    return r;
}

DA0 dretda0(DA0 x);
DA0 cretda0(DA0 x)
{
    DA0 r;
    return r;
}

DA4 dretda4(DA4 x);
DA4 cretda4(DA4 x)
{
    TEST(x.a[0] == a);
    TEST(x.a[1] == b);
    TEST(x.a[2] == c);
    TEST(x.a[3] == d);
    DA4 r = {(double)++a, (double)++b, (double)++c, (double)++d};
    return r;
}

DA5 dretda5(DA5 x);
DA5 cretda5(DA5 x)
{
    TEST(x.a[0] == a);
    TEST(x.a[1] == b);
    TEST(x.a[2] == c);
    TEST(x.a[3] == d);
    TEST(x.a[4] == e);
    DA5 r = {(double)++a, (double)++b, (double)++c, (double)++d, (double)++e};
    return r;
}

DA8 dretda8(DA8 x);
DA8 cretda8(DA8 x)
{
    TEST(x.a[0] == a);
    TEST(x.a[1] == b);
    TEST(x.a[2] == c);
    TEST(x.a[3] == d);
    TEST(x.a[4] == e);
    TEST(x.a[5] == f);
    TEST(x.a[6] == g);
    TEST(x.a[7] == h);
    DA8 r = {(double)++a, (double)++b, (double)++c, (double)++d,
             (double)++e, (double)++f, (double)++g, (double)++h};
    return r;
}

DHFA1 dretdhfa1(DHFA1 x);
DHFA1 cretdhfa1(DHFA1 x)
{
    TEST(x.d == a);
    DHFA1 r;
    r.d = ++a;
    return r;
}

DHFA2 dretdhfa2(DHFA2 x);
DHFA2 cretdhfa2(DHFA2 x)
{
    TEST(x.a == a);
    TEST(x.b.a == b);
    DHFA2 r = {(double)++a, {(double)++b}};
    return r;
}

DHFA2a dretdhfa2a(DHFA2a x);
DHFA2a cretdhfa2a(DHFA2a x)
{
    TEST(x.a.a == a);
    TEST(x.b == b);
    DHFA2a r = {{(double)++a}, (double)++b};
    return r;
}

DHFA4x dretdhfa4x(DHFA4x x);
DHFA4x cretdhfa4x(DHFA4x x)
{
    TEST(x.a[0].a == a);
    TEST(x.a[0].b == b);
    TEST(x.a[1].a == c);
    TEST(x.a[1].b == d);
    DHFA4x r = {{{(double)++a, (double)++b}, {(double)++c, (double)++d}}};
    return r;
}

S1 drets1(S1 x);
S1 crets1(S1 x)
{
    TEST(x.a == a);
    S1 r = {++a};
    return r;
}

S2 drets2(S2 x);
S2 crets2(S2 x)
{
    TEST(x.a == a);
    S2 r = {++a};
    return r;
}

double dretdouble(double x);
double cretdouble(double x)
{
    TEST(x == a);
    double r = ++a;
    return r;
}

long long dretlonglong(long long x);
long long cretlonglong(long long x)
{
    TEST(x == a);
    long long r = ++a;
    return r;
}

S9 drets9(S9 x);
S9 crets9(S9 x)
{
    TEST(x.a == a);
    TEST(x.b == b);
    S9 r = {(char)(++a), (double)(++b)};
    return r;
}

SA64 dretsa64(SA64 x);
SA64 cretsa64(SA64 x)
{
    TEST(testar(x.a, 64, a));

    SA64 r;
    ++a;
    for (int i = 0; i < 64; ++i) {
        r.a[i] = a+i;
    }
    return r;
}

SA65 dretsa65(SA65 x);
SA65 cretsa65(SA65 x)
{
    TEST(testar(x.a, 65, a));

    SA65 r;
    ++a;
    for (int i = 0; i < 65; ++i) {
        r.a[i] = a+i;
    }
    return r;
}

void cvfun(int s, ...)
{
    va_list args;
    va_start(args, s);
    switch (s) {
    case 0:
        cretb1(va_arg(args,B1));
        break;
    case 1:
        cretb2(va_arg(args,B2));
        break;
    case 2:
        creti2(va_arg(args,I2));
        break;
    case 3:
        cretf4(va_arg(args,F4));
        break;
    case 4:
        cretd4(va_arg(args,D4));
        break;
    case 5:
        cretdhfa2(va_arg(args,DHFA2));
        break;
    case 6:
        cretdhfa2a(va_arg(args,DHFA2a));
        break;
    case 7:
        cretuni1(va_arg(args,UNI1));
        break;
    case 8:
        cretdouble(va_arg(args,double));
        break;
    case 9:
        cretlonglong(va_arg(args,long long));
        break;
    case 10:
        crets9(va_arg(args,S9));
        break;
    }
}

void ccall()
{
    B1 b1 = {++a};
    B1 b1r = dretb1(b1);
    TEST(b1r.a == a);

    B2 b2 = {++a,++b};
    B2 b2r = dretb2(b2);
    TEST(b2r.a == a);
    TEST(b2r.b == b);

    I2 i2 = {++a,++b};
    I2 i2r = dreti2(i2);
    TEST(i2r.a == a);
    TEST(i2r.b == b);

    UNI1 uni1i = {{++a}};
    UNI1 uni1 = dretuni1(uni1i);
    TEST(uni1.a.a == a);

    F4 f4 = {(float)++a, (float)++b, (float)++c, (float)++d};
    F4 f4r = dretf4(f4);
    TEST(f4r.a == a);
    TEST(f4r.b == b);
    TEST(f4r.c == c);
    TEST(f4r.d == d);

    D4 d4 = {(double)++a, (double)++b, (double)++c, (double)++d};
    D4 d4r = dretd4(d4);
    TEST(d4r.a == a);
    TEST(d4r.b == b);
    TEST(d4r.c == c);
    TEST(d4r.d == d);

    D5 d5 = {(double)++a, (double)++b, (double)++c, (double)++d, (double)++e};
    D5 d5r = dretd5(d5);
    TEST(d5r.a == a);
    TEST(d5r.b == b);
    TEST(d5r.c == c);
    TEST(d5r.d == d);
    TEST(d5r.e == e);

    D8 d8 = {(double)++a, (double)++b, (double)++c, (double)++d,
             (double)++e, (double)++f, (double)++g, (double)++h};
    D8 d8r = dretd8(d8);
    TEST(d8r.a == a);
    TEST(d8r.b == b);
    TEST(d8r.c == c);
    TEST(d8r.d == d);
    TEST(d8r.e == e);
    TEST(d8r.f == f);
    TEST(d8r.g == g);
    TEST(d8r.h == h);

    UD4 ud4;
    D4 d4x = {(double)++a, (double)++b, (double)++c, (double)++d};
    ud4.c = d4x;
    UD4 ud4r = dretud4(ud4);
    TEST(ud4r.c.a == a);
    TEST(ud4r.c.b == b);
    TEST(ud4r.c.c == c);
    TEST(ud4r.c.d == d);

    DA4 da4 = {(double)++a, (double)++b, (double)++c, (double)++d};
    DA4 da4r = dretda4(da4);
    TEST(da4r.a[0] == a);
    TEST(da4r.a[1] == b);
    TEST(da4r.a[2] == c);
    TEST(da4r.a[3] == d);

    DA5 da5 = {(double)++a, (double)++b, (double)++c, (double)++d, (double)++e};
    DA5 da5r = dretda5(da5);
    TEST(da5r.a[0] == a);
    TEST(da5r.a[1] == b);
    TEST(da5r.a[2] == c);
    TEST(da5r.a[3] == d);
    TEST(da5r.a[4] == e);

    DA8 da8 = {(double)++a, (double)++b, (double)++c, (double)++d,
               (double)++e, (double)++f, (double)++g, (double)++h};
    DA8 da8r = dretda8(da8);
    TEST(da8r.a[0] == a);
    TEST(da8r.a[1] == b);
    TEST(da8r.a[2] == c);
    TEST(da8r.a[3] == d);
    TEST(da8r.a[4] == e);
    TEST(da8r.a[5] == f);
    TEST(da8r.a[6] == g);
    TEST(da8r.a[7] == h);

    DHFA2 dhfa2i = {(double)++a, {(double)++b}};
    DHFA2 dhfa2 = dretdhfa2(dhfa2i);
    TEST(dhfa2.a == a);
    TEST(dhfa2.b.a == b);

    DHFA2a dhfa2ai = {{(double)++a}, (double)++b};
    DHFA2a dhfa2a = dretdhfa2a(dhfa2ai);
    TEST(dhfa2a.a.a == a);
    TEST(dhfa2a.b == b);

    DHFA4x dhfa4xi = {{{(double)++a, (double)++b}, {(double)++c, (double)++d}}};
    DHFA4x dhfa4x = dretdhfa4x(dhfa4xi);
    TEST(dhfa4x.a[0].a == a);
    TEST(dhfa4x.a[0].b == b);
    TEST(dhfa4x.a[1].a == c);
    TEST(dhfa4x.a[1].b == d);

    // structs with postblit or dtor may not be passed like a similar POD
    // struct.
#if 0
    S1 s1 = {++a};
    S1 s1r = drets1(s1);
    TEST(s1r.a == a);

    S2 s2 = {++a};
    S2 s2r = drets2(s2);
    TEST(s2r.a == a);
#endif

    SA64 s64;
    ++a;
    for (int i = 0; i < 64; ++i) {
        s64.a[i] = a+i;
    }
    SA64 s64r = dretsa64(s64);
    TEST(testar(s64r.a, 64, a));

    SA65 s65;
    ++a;
    for (int i = 0; i < 65; ++i) {
        s65.a[i] = a+i;
    }
    SA65 s65r = dretsa65(s65);
    TEST(testar(s65r.a, 65, a));

    b1.a = ++a;
    dvfun(0, b1);

    b2.a = ++a;
    b2.b = ++b;
    dvfun(1, b2);

    i2.a = ++a;
    i2.b = ++b;
    dvfun(2, i2);

    uni1.a.a = ++a;
    cvfun(7, uni1); // TODO: type-oh?  Should be dvfun?

    f4.a = ++a;
    f4.b = ++b;
    f4.c = ++c;
    f4.d = ++d;
    dvfun(3, f4);

    d4.a = ++a;
    d4.b = ++b;
    d4.c = ++c;
    d4.d = ++d;
    dvfun(4, d4);

    dhfa2.a = ++a;
    dhfa2.b.a = ++b;
    dvfun(5, dhfa2);

    dhfa2a.a.a = ++a;
    dhfa2a.b = ++b;
    dvfun(6, dhfa2a);
}
} // extern "C"
