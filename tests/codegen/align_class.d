// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -run %s

struct S16 { align(16) short a; }

class D     { align(32) short d = 0xDD; }
// CHECK: %align_class.D = type <{ ptr, ptr, [{{(16|24)}} x i8], i16 }>
class E : D { S16 s16 = S16(0xEE); }
// CHECK: %align_class.E = type { ptr, ptr, [{{(16|24)}} x i8], i16, [14 x i8], %align_class.S16 }
class F : D { align(64) short f = 0xFF; }
// CHECK: %align_class.F = type <{ ptr, ptr, [{{(16|24)}} x i8], i16, [30 x i8], i16 }>

extern(C++) class CppClass { align(32) short a = 0xAA; }
// CHECK: %align_class.CppClass = type <{ ptr, [{{(24|28)}} x i8], i16 }>

void main()
{
    scope d = new D;
    // CHECK: = alloca %align_class.D, align 32
    static assert(D.d.offsetof == 32);
    assert(d.d == 0xDD);

    scope e = new E;
    // CHECK: = alloca %align_class.E, align 32
    static assert(E.d.offsetof == 32);
    assert(e.d == 0xDD);
    static assert(E.s16.offsetof == 48);
    assert(e.s16.a == 0xEE);

    scope f = new F;
    // CHECK: = alloca %align_class.F, align 64
    static assert(F.d.offsetof == 32);
    assert(f.d == 0xDD);
    static assert(F.f.offsetof == 64);
    assert(f.f == 0xFF);

    scope cppClass = new CppClass;
    // CHECK: = alloca %align_class.CppClass, align 32
    static assert(CppClass.a.offsetof == 32);
    assert(cppClass.a == 0xAA);
}
