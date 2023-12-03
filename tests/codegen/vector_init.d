// Make sure vector initializer llvm::Constants are generated correctly (GitHub #2101).
// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

alias D2 = __vector(double[2]);

// CHECK: @{{.*}}_D11vector_init12ImplicitInit6__initZ{{\"?}} =
// CHECK-SAME: { <2 x double> <double 0x7FF8000000000000, double 0x7FF8000000000000> }
struct ImplicitInit { D2 a; }

// CHECK: @{{.*}}_D11vector_init12ExplicitInit6__initZ{{\"?}} =
// CHECK-SAME: { <2 x double> <double 0x7FF8000000000000, double 0x7FF8000000000000> }
struct ExplicitInit { D2 a = D2.init; }

// CHECK: @{{.*}}_D11vector_init10SplatValue6__initZ{{\"?}} =
// CHECK-SAME: { <2 x double> <double 1.000000e+00, double 1.000000e+00> }
struct SplatValue { D2 a = 1.0; }

// CHECK: @{{.*}}_D11vector_init13ElementValues6__initZ{{\"?}} =
// CHECK-SAME: { <2 x double> <double 1.000000e+00, double 2.000000e+00> }
struct ElementValues { D2 a = [1.0, 2.0]; }


// CHECK: define {{.*}}_D11vector_init3foo
void foo()
{
    alias void16 = __vector(void[16]);
    alias short8 = __vector(short[8]);

    // CHECK-NEXT: %v16 = alloca <16 x i8>
    // CHECK-NEXT: %s8 = alloca <8 x i16>
    // CHECK-NEXT: %d2 = alloca <2 x double>
    // CHECK-NEXT: store <16 x i8> zeroinitializer, {{<16 x i8>\*|ptr}} %v16
    // CHECK-NEXT: store <8 x i16> <i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8>, {{<8 x i16>\*|ptr}} %s8
    // CHECK-NEXT: store <2 x double> <double 1.500000e+00, double 1.500000e+00>, {{<2 x double>\*|ptr}} %d2
    // CHECK-NEXT: ret void
    void16 v16;
    short8 s8 = [1, 2, 3, 4, 5, 6, 7, 8];
    D2 d2 = 1.5;
}

// https://github.com/ldc-developers/ldc/issues/3418
// CHECK: define {{.*}}_D11vector_init3bar
void bar(const ref float[4] floats, const ref int[4] ints)
{
    alias float4 = __vector(float[4]);
    alias int4 = __vector(int[4]);

    // CHECK: call void @llvm.memcpy{{.*}}({{i8\*|ptr}}{{[^,]*}} %{{.*}}
    auto f = cast(float4) floats;

    // CHECK: call void @llvm.memcpy{{.*}}({{i8\*|ptr}}{{[^,]*}} %{{.*}}
    auto i = cast(int4) ints;

    // CHECK: fptosi
    // CHECK: fptosi
    // CHECK: fptosi
    // CHECK: fptosi
    auto converted = cast(int4) floats;
}
